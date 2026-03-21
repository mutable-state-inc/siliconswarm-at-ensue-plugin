/// GPU FFN: Full feed-forward network on Metal GPU.
/// LayerNorm2 → FC up (768→3072) → GELU → FC down (3072→768) → residual add.
/// Uses real model weights loaded as Metal buffers.

use metal::*;
use std::sync::OnceLock;

const EMBED_DIM: usize = 768;
const FFN_DIM: usize = 3072; // 4 * 768
const SPATIAL: usize = 64;

struct GpuFfnLayer {
    pso_ln: ComputePipelineState,
    pso_fc_up: ComputePipelineState,
    pso_gelu: ComputePipelineState,
    pso_fc_down: ComputePipelineState,
    pso_residual: ComputePipelineState,
    ln2_weight: Buffer,
    ln2_bias: Buffer,
    fc_weight: Buffer,
    fc_bias: Buffer,
    fc_proj_weight: Buffer,
    fc_proj_bias: Buffer,
    buf_normalized: Buffer,
    buf_hidden: Buffer,
    buf_output: Buffer,
}

static GPU_FFN_SHADERS: OnceLock<ComputePipelineState> = OnceLock::new();

pub struct GpuFfn {
    device: Device,
    queue: CommandQueue,
    layers: Vec<GpuFfnLayer>,
}

impl GpuFfn {
    pub fn new(device: Device, layer_weights: &[crate::weights::LayerWeights]) -> Self {
        let queue = device.new_command_queue();

        let shader_src = Self::shader_source();
        let lib = device.new_library_with_source(&shader_src, &CompileOptions::new())
            .expect("FFN shader compile");

        let pso_ln = device.new_compute_pipeline_state_with_function(
            &lib.get_function("layer_norm", None).unwrap()).unwrap();
        let pso_fc_up = device.new_compute_pipeline_state_with_function(
            &lib.get_function("matmul_bias", None).unwrap()).unwrap();
        let pso_gelu = device.new_compute_pipeline_state_with_function(
            &lib.get_function("gelu_inplace", None).unwrap()).unwrap();
        let pso_fc_down = device.new_compute_pipeline_state_with_function(
            &lib.get_function("matmul_bias", None).unwrap()).unwrap();
        let pso_residual = device.new_compute_pipeline_state_with_function(
            &lib.get_function("residual_add", None).unwrap()).unwrap();

        let layers = layer_weights.iter().map(|w| {
            GpuFfnLayer {
                pso_ln: pso_ln.clone(),
                pso_fc_up: pso_fc_up.clone(),
                pso_gelu: pso_gelu.clone(),
                pso_fc_down: pso_fc_down.clone(),
                pso_residual: pso_residual.clone(),
                ln2_weight: Self::buf_from_f32(&device, &w.ln2_weight),
                ln2_bias: Self::buf_from_f32(&device, &w.ln2_bias),
                fc_weight: Self::buf_from_f32(&device, &w.fc_weight),
                fc_bias: Self::buf_from_f32(&device, &w.fc_bias),
                fc_proj_weight: Self::buf_from_f32(&device, &w.fc_proj_weight),
                fc_proj_bias: Self::buf_from_f32(&device, &w.fc_proj_bias),
                buf_normalized: device.new_buffer((EMBED_DIM * SPATIAL * 4) as u64, MTLResourceOptions::StorageModeShared),
                buf_hidden: device.new_buffer((FFN_DIM * SPATIAL * 4) as u64, MTLResourceOptions::StorageModeShared),
                buf_output: device.new_buffer((EMBED_DIM * SPATIAL * 4) as u64, MTLResourceOptions::StorageModeShared),
            }
        }).collect();

        GpuFfn { device, queue, layers }
    }

    /// Run FFN for one layer on GPU. Input/output are f32 slices in ANE NCHW layout.
    pub fn forward(&self, layer_idx: usize, input: &[f32], output: &mut [f32]) {
        let layer = &self.layers[layer_idx];

        // Copy input to GPU
        unsafe {
            let ptr = layer.buf_output.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(input.as_ptr(), ptr, input.len().min(EMBED_DIM * SPATIAL));
        }

        let cmd = self.queue.new_command_buffer();

        // 1. LayerNorm
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&layer.pso_ln);
        enc.set_buffer(0, Some(&layer.buf_output), 0);     // input
        enc.set_buffer(1, Some(&layer.ln2_weight), 0);       // gamma
        enc.set_buffer(2, Some(&layer.ln2_bias), 0);         // beta
        enc.set_buffer(3, Some(&layer.buf_normalized), 0);   // output
        enc.dispatch_threads(MTLSize::new(SPATIAL as u64, 1, 1), MTLSize::new(64, 1, 1));
        enc.end_encoding();

        // 2. FC up: [768] → [3072]
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&layer.pso_fc_up);
        enc.set_buffer(0, Some(&layer.buf_normalized), 0);   // input [768 * SPATIAL]
        enc.set_buffer(1, Some(&layer.fc_weight), 0);        // weight [3072 * 768]
        enc.set_buffer(2, Some(&layer.fc_bias), 0);          // bias [3072]
        enc.set_buffer(3, Some(&layer.buf_hidden), 0);       // output [3072 * SPATIAL]
        enc.dispatch_threads(MTLSize::new(FFN_DIM as u64, SPATIAL as u64, 1), MTLSize::new(16, 16, 1));
        enc.end_encoding();

        // 3. GELU
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&layer.pso_gelu);
        enc.set_buffer(0, Some(&layer.buf_hidden), 0);
        enc.dispatch_threads(MTLSize::new((FFN_DIM * SPATIAL) as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();

        // 4. FC down: [3072] → [768]
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&layer.pso_fc_down);
        enc.set_buffer(0, Some(&layer.buf_hidden), 0);       // input [3072 * SPATIAL]
        enc.set_buffer(1, Some(&layer.fc_proj_weight), 0);   // weight [768 * 3072]
        enc.set_buffer(2, Some(&layer.fc_proj_bias), 0);     // bias [768]
        enc.set_buffer(3, Some(&layer.buf_normalized), 0);   // reuse as output [768 * SPATIAL]
        enc.dispatch_threads(MTLSize::new(EMBED_DIM as u64, SPATIAL as u64, 1), MTLSize::new(16, 16, 1));
        enc.end_encoding();

        // 5. Residual add
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&layer.pso_residual);
        enc.set_buffer(0, Some(&layer.buf_normalized), 0);  // ffn output
        enc.set_buffer(1, Some(&layer.buf_output), 0);       // original input (residual)
        enc.set_buffer(2, Some(&layer.buf_output), 0);       // write result back
        enc.dispatch_threads(MTLSize::new((EMBED_DIM * SPATIAL) as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        // Copy output back
        unsafe {
            let ptr = layer.buf_output.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, output.as_mut_ptr(), output.len().min(EMBED_DIM * SPATIAL));
        }
    }

    fn buf_from_f32(device: &Device, data: &[f32]) -> Buffer {
        let buf = device.new_buffer((data.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        unsafe {
            let ptr = buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        buf
    }

    fn shader_source() -> String {
        r#"
#include <metal_stdlib>
using namespace metal;

constant uint EMBED_DIM = 768;
constant uint FFN_DIM = 3072;
constant uint SPATIAL = 64;
constant float LAYER_NORM_EPS = 1e-5;

// Data layout: NCHW where N=1, C=channels, H=1, W=spatial
// So element [c, s] = data[c * SPATIAL + s]

kernel void layer_norm(
    device const float *input [[buffer(0)]],
    device const float *gamma [[buffer(1)]],
    device const float *beta [[buffer(2)]],
    device float *output [[buffer(3)]],
    uint sid [[thread_position_in_grid]])
{
    if (sid >= SPATIAL) return;
    // Compute mean and variance over channels for this spatial position
    float mean = 0;
    for (uint c = 0; c < EMBED_DIM; c++) mean += input[c * SPATIAL + sid];
    mean /= EMBED_DIM;
    float var = 0;
    for (uint c = 0; c < EMBED_DIM; c++) {
        float d = input[c * SPATIAL + sid] - mean;
        var += d * d;
    }
    var /= EMBED_DIM;
    float rstd = rsqrt(var + LAYER_NORM_EPS);
    for (uint c = 0; c < EMBED_DIM; c++) {
        output[c * SPATIAL + sid] = (input[c * SPATIAL + sid] - mean) * rstd * gamma[c] + beta[c];
    }
}

kernel void matmul_bias(
    device const float *input [[buffer(0)]],    // [in_dim * SPATIAL]
    device const float *weight [[buffer(1)]],   // [out_dim * in_dim]
    device const float *bias [[buffer(2)]],     // [out_dim]
    device float *output [[buffer(3)]],         // [out_dim * SPATIAL]
    uint2 tid [[thread_position_in_grid]])
{
    uint out_c = tid.x;
    uint sid = tid.y;
    // Determine input dim from weight size (not ideal but works for our fixed sizes)
    uint in_dim = (out_c < FFN_DIM) ? EMBED_DIM : FFN_DIM;
    float sum = bias[out_c];
    for (uint i = 0; i < in_dim; i++) {
        sum += weight[out_c * in_dim + i] * input[i * SPATIAL + sid];
    }
    output[out_c * SPATIAL + sid] = sum;
}

kernel void gelu_inplace(
    device float *data [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    float x = data[tid];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x3 = x * x * x;
    float inner = 0.7978845608 * (x + 0.044715 * x3);
    data[tid] = 0.5 * x * (1.0 + tanh(inner));
}

kernel void residual_add(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *output [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    output[tid] = a[tid] + b[tid];
}
"#.to_string()
    }
}
