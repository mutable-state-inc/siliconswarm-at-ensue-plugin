/// Phi-1.5 Instruct (1.4B) on Apple GPU via Metal compute shaders.
/// All computation on GPU — no CPU fallback for RoPE, LayerNorm, etc.
/// Architecture: partial RoPE (50%), LayerNorm, GELU, MHA, parallel attn+MLP.
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use half::{bf16, f16};
use hf_hub::api::sync::ApiBuilder;
use metal::*;
use rand::{Rng, RngExt};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use tokenizers::Tokenizer;

const REPO_ID: &str = "rasyosef/Phi-1_5-Instruct-v0.1";
const MAX_NEW_TOKENS: usize = 100;
const MAX_SEQ: usize = 128;
const TEMPERATURE: f32 = 0.7;
const TOP_P: f32 = 0.9;
const REPETITION_PENALTY: f32 = 1.1;

#[derive(Debug, Clone, Deserialize)]
struct Config {
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    vocab_size: usize,
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
    #[serde(default = "default_eps")]
    layer_norm_eps: f64,
    #[serde(default = "default_rotary")]
    partial_rotary_factor: f64,
}
fn default_rope_theta() -> f64 { 10000.0 }
fn default_eps() -> f64 { 1e-5 }
fn default_rotary() -> f64 { 0.5 }
impl Config {
    fn head_dim(&self) -> usize { self.hidden_size / self.num_attention_heads }
    fn rotary_dim(&self) -> usize { (self.head_dim() as f64 * self.partial_rotary_factor) as usize }
}

// ─── Weight loading (same as phi_ane.rs) ─────────────────────────────────

struct LayerWeights {
    ln_weight: Box<[f32]>, ln_bias: Box<[f32]>,
    q_proj_w: Box<[f32]>, q_proj_b: Box<[f32]>,
    k_proj_w: Box<[f32]>, k_proj_b: Box<[f32]>,
    v_proj_w: Box<[f32]>, v_proj_b: Box<[f32]>,
    dense_w: Box<[f32]>, dense_b: Box<[f32]>,
    fc1_w: Box<[f32]>, fc1_b: Box<[f32]>,
    fc2_w: Box<[f32]>, fc2_b: Box<[f32]>,
}

struct ModelWeights {
    embed_tokens: Box<[f32]>,
    layers: Box<[LayerWeights]>,
    final_ln_w: Box<[f32]>, final_ln_b: Box<[f32]>,
    lm_head_w: Box<[f32]>, lm_head_b: Box<[f32]>,
}

fn tensor_f32(st: &SafeTensors, name: &str) -> Box<[f32]> {
    let t = st.tensor(name).unwrap_or_else(|_| panic!("missing: {name}"));
    let b = t.data();
    match t.dtype() {
        Dtype::BF16 => b.chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0],c[1]])).to_f32()).collect(),
        Dtype::F16 => b.chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0],c[1]])).to_f32()).collect(),
        Dtype::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect(),
        other => panic!("unsupported dtype: {other:?}"),
    }
}

fn download(repo_id: &str) -> Result<(Config, PathBuf, Vec<u8>), Box<dyn std::error::Error>> {
    let api = ApiBuilder::new().with_progress(true).build()?;
    let repo = api.model(repo_id.to_string());
    eprint_status("Downloading config.json");
    let config: Config = serde_json::from_reader(std::fs::File::open(repo.get("config.json")?)?)?;
    eprint_status("Downloading tokenizer");
    let tok_path = repo.get("tokenizer.json")?;
    eprint_status("Downloading model.safetensors");
    let sb = std::fs::read(repo.get("model.safetensors")?)?;
    eprint_done("Downloaded model files");
    Ok((config, tok_path, sb))
}

fn load_weights(st: &SafeTensors, cfg: &Config) -> ModelWeights {
    let d = cfg.hidden_size;
    let layers: Box<[LayerWeights]> = (0..cfg.num_hidden_layers).map(|i| {
        let p = format!("model.layers.{i}");
        LayerWeights {
            ln_weight: tensor_f32(st, &format!("{p}.input_layernorm.weight")),
            ln_bias: tensor_f32(st, &format!("{p}.input_layernorm.bias")),
            q_proj_w: tensor_f32(st, &format!("{p}.self_attn.q_proj.weight")),
            q_proj_b: tensor_f32(st, &format!("{p}.self_attn.q_proj.bias")),
            k_proj_w: tensor_f32(st, &format!("{p}.self_attn.k_proj.weight")),
            k_proj_b: tensor_f32(st, &format!("{p}.self_attn.k_proj.bias")),
            v_proj_w: tensor_f32(st, &format!("{p}.self_attn.v_proj.weight")),
            v_proj_b: tensor_f32(st, &format!("{p}.self_attn.v_proj.bias")),
            dense_w: tensor_f32(st, &format!("{p}.self_attn.dense.weight")),
            dense_b: tensor_f32(st, &format!("{p}.self_attn.dense.bias")),
            fc1_w: tensor_f32(st, &format!("{p}.mlp.fc1.weight")),
            fc1_b: tensor_f32(st, &format!("{p}.mlp.fc1.bias")),
            fc2_w: tensor_f32(st, &format!("{p}.mlp.fc2.weight")),
            fc2_b: tensor_f32(st, &format!("{p}.mlp.fc2.bias")),
        }
    }).collect();
    ModelWeights {
        embed_tokens: tensor_f32(st, "model.embed_tokens.weight"),
        layers,
        final_ln_w: tensor_f32(st, "model.final_layernorm.weight"),
        final_ln_b: tensor_f32(st, "model.final_layernorm.bias"),
        lm_head_w: tensor_f32(st, "lm_head.weight"),
        lm_head_b: tensor_f32(st, "lm_head.bias"),
    }
}

// ─── Sampling (CPU — only thing that stays on CPU) ─────────────────────

fn sample(logits: &[f32], history: &[u32], rng: &mut impl Rng) -> u32 {
    let mut l: Box<[f32]> = logits.into();
    for &t in history {
        let i = t as usize;
        if i < l.len() {
            if l[i] > 0.0 { l[i] /= REPETITION_PENALTY; } else { l[i] *= REPETITION_PENALTY; }
        }
    }
    if TEMPERATURE <= 0.0 {
        return l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
    }
    let max = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = l.iter().enumerate()
        .map(|(i, &v)| (i, ((v - max) / TEMPERATURE).exp())).collect();
    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let total: f32 = probs.iter().map(|p| p.1).sum();
    let cutoff = TOP_P * total;
    let mut cum = 0f32;
    let mut cands = Vec::new();
    for &(idx, p) in &probs { cum += p; cands.push((idx, p)); if cum >= cutoff { break; } }
    let ct: f32 = cands.iter().map(|p| p.1).sum();
    let thresh = rng.random::<f32>() * ct;
    let mut acc = 0f32;
    for &(idx, p) in &cands { acc += p; if acc >= thresh { return idx as u32; } }
    cands.last().unwrap().0 as u32
}

fn eprint_status(msg: &str) { eprint!("\r\x1b[2K\x1b[1;36m⠋\x1b[0m {msg}"); let _ = io::stderr().flush(); }
fn eprint_done(msg: &str) { eprintln!("\r\x1b[2K\x1b[1;32m✓\x1b[0m {msg}"); }

// ─── Metal Shader Source ────────────────────────────────────────────────

const METAL_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ─── Tiled MatMul: C[M,N] = A[M,K] * B^T[N,K] + bias[N] ──────────────
// A is input row-major [M, K], B is weight [N, K] (out_features, in_features),
// so we compute A * B^T. For single-token inference M=1.
// TILE_SIZE=16 for shared memory tiling.

constant int TILE = 16;

kernel void matmul_tiled(
    device const float* A       [[buffer(0)]],   // [M, K]
    device const float* B       [[buffer(1)]],   // [N, K] row-major (weight)
    device const float* bias    [[buffer(2)]],   // [N]
    device float* C             [[buffer(3)]],   // [M, N]
    constant uint& M            [[buffer(4)]],
    constant uint& N            [[buffer(5)]],
    constant uint& K            [[buffer(6)]],
    uint2 gid                   [[threadgroup_position_in_grid]],
    uint2 tid                   [[thread_position_in_threadgroup]]
) {
    // Each threadgroup computes a TILE x TILE block of C
    int row = gid.y * TILE + tid.y;
    int col = gid.x * TILE + tid.x;

    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];

    float sum = 0.0f;

    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE + tid.x;
        int bCol = t * TILE + tid.y;

        // Load A tile
        if (row < (int)M && aCol < (int)K) {
            As[tid.y][tid.x] = A[row * K + aCol];
        } else {
            As[tid.y][tid.x] = 0.0f;
        }

        // Load B tile (B is [N,K], we want B[col, bCol])
        if (col < (int)N && bCol < (int)K) {
            Bs[tid.y][tid.x] = B[col * K + bCol];
        } else {
            Bs[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int k = 0; k < TILE; k++) {
            sum += As[tid.y][k] * Bs[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < (int)M && col < (int)N) {
        C[row * N + col] = sum + bias[col];
    }
}

// ─── LayerNorm: out[i] = (x[i] - mean) / sqrt(var + eps) * w[i] + b[i] ──
// Single-thread per normalization (dim is small enough: 2048)
kernel void layer_norm(
    device const float* input   [[buffer(0)]],   // [dim]
    device const float* weight  [[buffer(1)]],   // [dim]
    device const float* bias_ln [[buffer(2)]],   // [dim]
    device float* output        [[buffer(3)]],   // [dim]
    constant uint& dim          [[buffer(4)]],
    constant float& eps         [[buffer(5)]],
    uint tid                    [[thread_position_in_grid]]
) {
    // Single thread computes entire LayerNorm (launched with 1 thread)
    if (tid != 0) return;

    float mean = 0.0f;
    for (uint i = 0; i < dim; i++) {
        mean += input[i];
    }
    mean /= float(dim);

    float var = 0.0f;
    for (uint i = 0; i < dim; i++) {
        float d = input[i] - mean;
        var += d * d;
    }
    var /= float(dim);

    float inv_std = 1.0f / sqrt(var + eps);

    for (uint i = 0; i < dim; i++) {
        output[i] = (input[i] - mean) * inv_std * weight[i] + bias_ln[i];
    }
}

// ─── GELU activation (tanh approximation) ──────────────────────────────
kernel void gelu_activation(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint& count        [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    float x = input[tid];
    float x3 = x * x * x;
    float inner = 0.7978846f * (x + 0.044715f * x3);
    float t = tanh(inner);
    output[tid] = 0.5f * x * (1.0f + t);
}

// ─── Partial RoPE (half-split, first rotary_dim dims) ───────────────────
// data layout: [num_heads * head_dim] flat vector
// rotary_dim = partial_rotary_factor * head_dim (e.g., 32 out of 64)
// half = rotary_dim / 2 = 16
// For each head h, dims [0..half) and [half..rotary_dim) get rotated
kernel void rope_partial(
    device float* data          [[buffer(0)]],   // [num_heads * head_dim]
    constant uint& num_heads    [[buffer(1)]],
    constant uint& head_dim     [[buffer(2)]],
    constant uint& rotary_dim   [[buffer(3)]],
    constant uint& pos          [[buffer(4)]],
    constant float& theta       [[buffer(5)]],
    uint tid                    [[thread_position_in_grid]]
) {
    uint half_rot = rotary_dim / 2;
    // tid indexes (head, pair_index)
    uint h = tid / half_rot;
    uint i = tid % half_rot;
    if (h >= num_heads) return;

    float freq = 1.0f / pow(theta, 2.0f * float(i) / float(rotary_dim));
    float angle = float(pos) * freq;
    float c = cos(angle);
    float s = sin(angle);

    uint base = h * head_dim;
    float x0 = data[base + i];
    float x1 = data[base + i + half_rot];
    data[base + i]            = x0 * c - x1 * s;
    data[base + i + half_rot] = x0 * s + x1 * c;
}

// ─── KV Cache Write: write K[dim] and V[dim] at position ──────────────
// Cache layout: [dim * MAX_SEQ], where cache[c * MAX_SEQ + pos] = data[c]
kernel void kv_cache_write(
    device const float* k_data  [[buffer(0)]],   // [dim]
    device const float* v_data  [[buffer(1)]],   // [dim]
    device float* k_cache       [[buffer(2)]],   // [dim * MAX_SEQ]
    device float* v_cache       [[buffer(3)]],   // [dim * MAX_SEQ]
    constant uint& dim          [[buffer(4)]],
    constant uint& max_seq      [[buffer(5)]],
    constant uint& pos          [[buffer(6)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    k_cache[tid * max_seq + pos] = k_data[tid];
    v_cache[tid * max_seq + pos] = v_data[tid];
}

// ─── Attention: per-head dot-product attention with causal mask ────────
// For single-token decode: Q is [num_heads, head_dim], one position.
// K cache: [dim * MAX_SEQ] with dim = num_heads * head_dim
// Output: attn_out[num_heads * head_dim]
// Each thread handles one head.
kernel void attention_single_token(
    device const float* Q       [[buffer(0)]],   // [num_heads * head_dim]
    device const float* K_cache [[buffer(1)]],   // [dim * MAX_SEQ]
    device const float* V_cache [[buffer(2)]],   // [dim * MAX_SEQ]
    device float* output        [[buffer(3)]],   // [num_heads * head_dim]
    constant uint& num_heads    [[buffer(4)]],
    constant uint& head_dim     [[buffer(5)]],
    constant uint& max_seq      [[buffer(6)]],
    constant uint& seq_len      [[buffer(7)]],   // number of valid positions (pos+1)
    uint h                      [[thread_position_in_grid]]
) {
    if (h >= num_heads) return;

    float scale = 1.0f / sqrt(float(head_dim));

    // Compute attention scores for this head over all valid positions
    // scores[p] = dot(Q[h], K[h, p]) * scale
    // K_cache layout: cache[(h*head_dim + d) * MAX_SEQ + p]

    // Find max score for numerical stability
    float max_score = -1e30f;
    for (uint p = 0; p < seq_len; p++) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            float q_val = Q[h * head_dim + d];
            float k_val = K_cache[(h * head_dim + d) * max_seq + p];
            dot += q_val * k_val;
        }
        dot *= scale;
        if (dot > max_score) max_score = dot;
    }

    // Softmax: exp and sum
    float sum_exp = 0.0f;
    // We'll recompute scores — for small seq_len this is fine
    // For larger models we'd store scores in threadgroup memory
    for (uint p = 0; p < seq_len; p++) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            float q_val = Q[h * head_dim + d];
            float k_val = K_cache[(h * head_dim + d) * max_seq + p];
            dot += q_val * k_val;
        }
        dot *= scale;
        sum_exp += exp(dot - max_score);
    }

    // Weighted sum of values
    for (uint d = 0; d < head_dim; d++) {
        float val = 0.0f;
        for (uint p = 0; p < seq_len; p++) {
            // Recompute score
            float dot = 0.0f;
            for (uint dd = 0; dd < head_dim; dd++) {
                float q_val = Q[h * head_dim + dd];
                float k_val = K_cache[(h * head_dim + dd) * max_seq + p];
                dot += q_val * k_val;
            }
            dot *= scale;
            float prob = exp(dot - max_score) / sum_exp;
            float v_val = V_cache[(h * head_dim + d) * max_seq + p];
            val += prob * v_val;
        }
        output[h * head_dim + d] = val;
    }
}

// ─── Residual add: out = a + b + c ─────────────────────────────────────
kernel void residual_add3(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device const float* c       [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& count        [[buffer(4)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    output[tid] = a[tid] + b[tid] + c[tid];
}

// ─── Element-wise add: out = a + b ─────────────────────────────────────
kernel void vec_add(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant uint& count        [[buffer(3)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    output[tid] = a[tid] + b[tid];
}

// ─── Copy kernel (for embedding lookup) ────────────────────────────────
kernel void embed_lookup(
    device const float* embed_table [[buffer(0)]],  // [vocab_size, dim]
    device float* output            [[buffer(1)]],  // [dim]
    constant uint& token_id         [[buffer(2)]],
    constant uint& dim              [[buffer(3)]],
    uint tid                        [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    output[tid] = embed_table[token_id * dim + tid];
}
"#;

// ─── Metal GPU Context ─────────────────────────────────────────────────

struct GpuContext {
    device: Device,
    queue: CommandQueue,
    // Pipeline states for each kernel
    matmul_pso: ComputePipelineState,
    layer_norm_pso: ComputePipelineState,
    gelu_pso: ComputePipelineState,
    rope_pso: ComputePipelineState,
    kv_write_pso: ComputePipelineState,
    attention_pso: ComputePipelineState,
    residual3_pso: ComputePipelineState,
    vec_add_pso: ComputePipelineState,
    embed_pso: ComputePipelineState,
}

impl GpuContext {
    fn new() -> Self {
        let device = Device::system_default().expect("No Metal device found");
        let queue = device.new_command_queue();
        let library = device.new_library_with_source(METAL_SHADERS, &CompileOptions::new())
            .expect("Failed to compile Metal shaders");

        let make_pso = |name: &str| -> ComputePipelineState {
            let func = library.get_function(name, None).unwrap_or_else(|e| panic!("missing kernel {name}: {e}"));
            device.new_compute_pipeline_state_with_function(&func).unwrap_or_else(|e| panic!("PSO {name}: {e}"))
        };

        GpuContext {
            matmul_pso: make_pso("matmul_tiled"),
            layer_norm_pso: make_pso("layer_norm"),
            gelu_pso: make_pso("gelu_activation"),
            rope_pso: make_pso("rope_partial"),
            kv_write_pso: make_pso("kv_cache_write"),
            attention_pso: make_pso("attention_single_token"),
            residual3_pso: make_pso("residual_add3"),
            vec_add_pso: make_pso("vec_add"),
            embed_pso: make_pso("embed_lookup"),
            device,
            queue,
        }
    }

    fn buf_from_slice(&self, data: &[f32]) -> Buffer {
        let len = (data.len() * std::mem::size_of::<f32>()) as u64;
        self.device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            len,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn buf_zeros(&self, count: usize) -> Buffer {
        let len = (count * std::mem::size_of::<f32>()) as u64;
        let buf = self.device.new_buffer(len, MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::write_bytes(buf.contents() as *mut u8, 0, len as usize);
        }
        buf
    }

    fn read_buf(buf: &Buffer, count: usize) -> Vec<f32> {
        let mut out = vec![0f32; count];
        unsafe {
            std::ptr::copy_nonoverlapping(buf.contents() as *const f32, out.as_mut_ptr(), count);
        }
        out
    }

    fn write_u32_to_buf(buf: &Buffer, val: u32) {
        unsafe {
            *(buf.contents() as *mut u32) = val;
        }
    }
}

// ─── GPU Layer Buffers ──────────────────────────────────────────────────

struct GpuLayerBuffers {
    ln_weight: Buffer,
    ln_bias: Buffer,
    q_proj_w: Buffer, q_proj_b: Buffer,
    k_proj_w: Buffer, k_proj_b: Buffer,
    v_proj_w: Buffer, v_proj_b: Buffer,
    dense_w: Buffer, dense_b: Buffer,
    fc1_w: Buffer, fc1_b: Buffer,
    fc2_w: Buffer, fc2_b: Buffer,
    // KV cache buffers (persistent on GPU)
    k_cache: Buffer,  // [dim * MAX_SEQ]
    v_cache: Buffer,  // [dim * MAX_SEQ]
}

struct GpuModelBuffers {
    embed_table: Buffer,
    layers: Vec<GpuLayerBuffers>,
    final_ln_w: Buffer, final_ln_b: Buffer,
    lm_head_w: Buffer, lm_head_b: Buffer,
}

fn upload_weights(ctx: &GpuContext, weights: &ModelWeights, cfg: &Config) -> GpuModelBuffers {
    let d = cfg.hidden_size;
    let layers: Vec<GpuLayerBuffers> = weights.layers.iter().map(|lw| {
        GpuLayerBuffers {
            ln_weight: ctx.buf_from_slice(&lw.ln_weight),
            ln_bias: ctx.buf_from_slice(&lw.ln_bias),
            q_proj_w: ctx.buf_from_slice(&lw.q_proj_w),
            q_proj_b: ctx.buf_from_slice(&lw.q_proj_b),
            k_proj_w: ctx.buf_from_slice(&lw.k_proj_w),
            k_proj_b: ctx.buf_from_slice(&lw.k_proj_b),
            v_proj_w: ctx.buf_from_slice(&lw.v_proj_w),
            v_proj_b: ctx.buf_from_slice(&lw.v_proj_b),
            dense_w: ctx.buf_from_slice(&lw.dense_w),
            dense_b: ctx.buf_from_slice(&lw.dense_b),
            fc1_w: ctx.buf_from_slice(&lw.fc1_w),
            fc1_b: ctx.buf_from_slice(&lw.fc1_b),
            fc2_w: ctx.buf_from_slice(&lw.fc2_w),
            fc2_b: ctx.buf_from_slice(&lw.fc2_b),
            k_cache: ctx.buf_zeros(d * MAX_SEQ),
            v_cache: ctx.buf_zeros(d * MAX_SEQ),
        }
    }).collect();

    GpuModelBuffers {
        embed_table: ctx.buf_from_slice(&weights.embed_tokens),
        layers,
        final_ln_w: ctx.buf_from_slice(&weights.final_ln_w),
        final_ln_b: ctx.buf_from_slice(&weights.final_ln_b),
        lm_head_w: ctx.buf_from_slice(&weights.lm_head_w),
        lm_head_b: ctx.buf_from_slice(&weights.lm_head_b),
    }
}

// ─── GPU Forward Pass ───────────────────────────────────────────────────

fn gpu_forward(
    ctx: &GpuContext,
    bufs: &GpuModelBuffers,
    cfg: &Config,
    // Scratch buffers (reused across calls)
    hidden: &Buffer,      // [dim]
    normed: &Buffer,      // [dim]
    q_buf: &Buffer,       // [dim]
    k_buf: &Buffer,       // [dim]
    v_buf: &Buffer,       // [dim]
    fc1_out: &Buffer,     // [intermediate_size]
    fc1_gelu: &Buffer,    // [intermediate_size]
    attn_out: &Buffer,    // [dim]
    dense_out: &Buffer,   // [dim]
    mlp_out: &Buffer,     // [dim]
    residual: &Buffer,    // [dim]
    logits: &Buffer,      // [vocab_size]
    // Parameters
    param_m: &Buffer,     // u32: 1
    param_dim: &Buffer,   // u32: dim
    param_ff: &Buffer,    // u32: intermediate_size
    param_vocab: &Buffer, // u32: vocab_size
    param_nh: &Buffer,    // u32: num_heads
    param_hd: &Buffer,    // u32: head_dim
    param_rotdim: &Buffer,// u32: rotary_dim
    param_maxseq: &Buffer,// u32: MAX_SEQ
    param_pos: &Buffer,   // u32: current position
    param_seqlen: &Buffer,// u32: seq_len (pos+1)
    param_eps: &Buffer,   // f32: layer_norm_eps
    param_theta: &Buffer, // f32: rope_theta
    param_token: &Buffer, // u32: token_id
    token_id: u32,
    pos: usize,
) {
    let d = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let hd = cfg.head_dim();
    let ff = cfg.intermediate_size;
    let rot_dim = cfg.rotary_dim();
    let half_rot = rot_dim / 2;

    // Update position parameters
    GpuContext::write_u32_to_buf(param_pos, pos as u32);
    GpuContext::write_u32_to_buf(param_seqlen, (pos + 1) as u32);
    GpuContext::write_u32_to_buf(param_token, token_id);

    // Embedding lookup on GPU
    {
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.embed_pso);
        enc.set_buffer(0, Some(&bufs.embed_table), 0);
        enc.set_buffer(1, Some(hidden), 0);
        enc.set_buffer(2, Some(param_token), 0);
        enc.set_buffer(3, Some(param_dim), 0);
        let tg = MTLSize::new(((d + 63) / 64) as u64, 1, 1);
        let tpg = MTLSize::new(64, 1, 1);
        enc.dispatch_thread_groups(tg, tpg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Process each layer
    for i in 0..cfg.num_hidden_layers {
        let lb = &bufs.layers[i];

        // Copy hidden to residual
        unsafe {
            std::ptr::copy_nonoverlapping(
                hidden.contents() as *const f32,
                residual.contents() as *mut f32,
                d,
            );
        }

        let cb = ctx.queue.new_command_buffer();

        // 1. LayerNorm
        {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.layer_norm_pso);
            enc.set_buffer(0, Some(hidden), 0);
            enc.set_buffer(1, Some(&lb.ln_weight), 0);
            enc.set_buffer(2, Some(&lb.ln_bias), 0);
            enc.set_buffer(3, Some(normed), 0);
            enc.set_buffer(4, Some(param_dim), 0);
            enc.set_buffer(5, Some(param_eps), 0);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            enc.end_encoding();
        }

        cb.commit();
        cb.wait_until_completed();

        // 2. QKV projections (tiled matmul) — can run in parallel via same command buffer
        {
            let cb = ctx.queue.new_command_buffer();

            // Q projection: [1, dim] x [dim, dim]^T = [1, dim]
            {
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&ctx.matmul_pso);
                enc.set_buffer(0, Some(normed), 0);
                enc.set_buffer(1, Some(&lb.q_proj_w), 0);
                enc.set_buffer(2, Some(&lb.q_proj_b), 0);
                enc.set_buffer(3, Some(q_buf), 0);
                enc.set_buffer(4, Some(param_m), 0);
                enc.set_buffer(5, Some(param_dim), 0);
                enc.set_buffer(6, Some(param_dim), 0);
                let tg_x = (d + 15) / 16;
                let tg_y = (1 + 15) / 16;
                enc.dispatch_thread_groups(MTLSize::new(tg_x as u64, tg_y as u64, 1), MTLSize::new(16, 16, 1));
                enc.end_encoding();
            }

            // K projection
            {
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&ctx.matmul_pso);
                enc.set_buffer(0, Some(normed), 0);
                enc.set_buffer(1, Some(&lb.k_proj_w), 0);
                enc.set_buffer(2, Some(&lb.k_proj_b), 0);
                enc.set_buffer(3, Some(k_buf), 0);
                enc.set_buffer(4, Some(param_m), 0);
                enc.set_buffer(5, Some(param_dim), 0);
                enc.set_buffer(6, Some(param_dim), 0);
                let tg_x = (d + 15) / 16;
                enc.dispatch_thread_groups(MTLSize::new(tg_x as u64, 1, 1), MTLSize::new(16, 16, 1));
                enc.end_encoding();
            }

            // V projection
            {
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&ctx.matmul_pso);
                enc.set_buffer(0, Some(normed), 0);
                enc.set_buffer(1, Some(&lb.v_proj_w), 0);
                enc.set_buffer(2, Some(&lb.v_proj_b), 0);
                enc.set_buffer(3, Some(v_buf), 0);
                enc.set_buffer(4, Some(param_m), 0);
                enc.set_buffer(5, Some(param_dim), 0);
                enc.set_buffer(6, Some(param_dim), 0);
                let tg_x = (d + 15) / 16;
                enc.dispatch_thread_groups(MTLSize::new(tg_x as u64, 1, 1), MTLSize::new(16, 16, 1));
                enc.end_encoding();
            }

            // FC1 projection (MLP, parallel with attention)
            {
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&ctx.matmul_pso);
                enc.set_buffer(0, Some(normed), 0);
                enc.set_buffer(1, Some(&lb.fc1_w), 0);
                enc.set_buffer(2, Some(&lb.fc1_b), 0);
                enc.set_buffer(3, Some(fc1_out), 0);
                enc.set_buffer(4, Some(param_m), 0);
                enc.set_buffer(5, Some(param_ff), 0);
                enc.set_buffer(6, Some(param_dim), 0);
                let tg_x = (ff + 15) / 16;
                enc.dispatch_thread_groups(MTLSize::new(tg_x as u64, 1, 1), MTLSize::new(16, 16, 1));
                enc.end_encoding();
            }

            cb.commit();
            cb.wait_until_completed();
        }

        // 3. GELU on fc1_out -> fc1_gelu
        {
            let cb = ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.gelu_pso);
            enc.set_buffer(0, Some(fc1_out), 0);
            enc.set_buffer(1, Some(fc1_gelu), 0);
            enc.set_buffer(2, Some(param_ff), 0);
            let tg = (ff + 255) / 256;
            enc.dispatch_thread_groups(MTLSize::new(tg as u64, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // 4. RoPE on Q and K (on GPU)
        {
            let cb = ctx.queue.new_command_buffer();
            // RoPE Q
            {
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&ctx.rope_pso);
                enc.set_buffer(0, Some(q_buf), 0);
                enc.set_buffer(1, Some(param_nh), 0);
                enc.set_buffer(2, Some(param_hd), 0);
                enc.set_buffer(3, Some(param_rotdim), 0);
                enc.set_buffer(4, Some(param_pos), 0);
                enc.set_buffer(5, Some(param_theta), 0);
                let total_pairs = nh * half_rot;
                let tg = (total_pairs + 63) / 64;
                enc.dispatch_thread_groups(MTLSize::new(tg as u64, 1, 1), MTLSize::new(64, 1, 1));
                enc.end_encoding();
            }
            // RoPE K
            {
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&ctx.rope_pso);
                enc.set_buffer(0, Some(k_buf), 0);
                enc.set_buffer(1, Some(param_nh), 0);
                enc.set_buffer(2, Some(param_hd), 0);
                enc.set_buffer(3, Some(param_rotdim), 0);
                enc.set_buffer(4, Some(param_pos), 0);
                enc.set_buffer(5, Some(param_theta), 0);
                let total_pairs = nh * half_rot;
                let tg = (total_pairs + 63) / 64;
                enc.dispatch_thread_groups(MTLSize::new(tg as u64, 1, 1), MTLSize::new(64, 1, 1));
                enc.end_encoding();
            }
            cb.commit();
            cb.wait_until_completed();
        }

        // 5. Write K, V to KV cache
        {
            let cb = ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.kv_write_pso);
            enc.set_buffer(0, Some(k_buf), 0);
            enc.set_buffer(1, Some(v_buf), 0);
            enc.set_buffer(2, Some(&lb.k_cache), 0);
            enc.set_buffer(3, Some(&lb.v_cache), 0);
            enc.set_buffer(4, Some(param_dim), 0);
            enc.set_buffer(5, Some(param_maxseq), 0);
            enc.set_buffer(6, Some(param_pos), 0);
            let tg = (d + 63) / 64;
            enc.dispatch_thread_groups(MTLSize::new(tg as u64, 1, 1), MTLSize::new(64, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // 6. Attention (single-token, per-head)
        {
            let cb = ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.attention_pso);
            enc.set_buffer(0, Some(q_buf), 0);
            enc.set_buffer(1, Some(&lb.k_cache), 0);
            enc.set_buffer(2, Some(&lb.v_cache), 0);
            enc.set_buffer(3, Some(attn_out), 0);
            enc.set_buffer(4, Some(param_nh), 0);
            enc.set_buffer(5, Some(param_hd), 0);
            enc.set_buffer(6, Some(param_maxseq), 0);
            enc.set_buffer(7, Some(param_seqlen), 0);
            // One thread per head
            enc.dispatch_thread_groups(MTLSize::new(nh as u64, 1, 1), MTLSize::new(1, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // 7. Output projection (dense): attn_out -> dense_out
        {
            let cb = ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.matmul_pso);
            enc.set_buffer(0, Some(attn_out), 0);
            enc.set_buffer(1, Some(&lb.dense_w), 0);
            enc.set_buffer(2, Some(&lb.dense_b), 0);
            enc.set_buffer(3, Some(dense_out), 0);
            enc.set_buffer(4, Some(param_m), 0);
            enc.set_buffer(5, Some(param_dim), 0);
            enc.set_buffer(6, Some(param_dim), 0);
            let tg_x = (d + 15) / 16;
            enc.dispatch_thread_groups(MTLSize::new(tg_x as u64, 1, 1), MTLSize::new(16, 16, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // 8. FC2 projection: fc1_gelu -> mlp_out
        {
            let cb = ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.matmul_pso);
            enc.set_buffer(0, Some(fc1_gelu), 0);
            enc.set_buffer(1, Some(&lb.fc2_w), 0);
            enc.set_buffer(2, Some(&lb.fc2_b), 0);
            enc.set_buffer(3, Some(mlp_out), 0);
            enc.set_buffer(4, Some(param_m), 0);
            enc.set_buffer(5, Some(param_dim), 0);
            enc.set_buffer(6, Some(param_ff), 0);
            let tg_x = (d + 15) / 16;
            enc.dispatch_thread_groups(MTLSize::new(tg_x as u64, 1, 1), MTLSize::new(16, 16, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // 9. Residual: hidden = residual + dense_out + mlp_out
        {
            let cb = ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.residual3_pso);
            enc.set_buffer(0, Some(residual), 0);
            enc.set_buffer(1, Some(dense_out), 0);
            enc.set_buffer(2, Some(mlp_out), 0);
            enc.set_buffer(3, Some(hidden), 0);
            enc.set_buffer(4, Some(param_dim), 0);
            let tg = (d + 255) / 256;
            enc.dispatch_thread_groups(MTLSize::new(tg as u64, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
    }

    // Final LayerNorm
    {
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.layer_norm_pso);
        enc.set_buffer(0, Some(hidden), 0);
        enc.set_buffer(1, Some(&bufs.final_ln_w), 0);
        enc.set_buffer(2, Some(&bufs.final_ln_b), 0);
        enc.set_buffer(3, Some(normed), 0);
        enc.set_buffer(4, Some(param_dim), 0);
        enc.set_buffer(5, Some(param_eps), 0);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // LM Head: [1, dim] x [vocab, dim]^T = [1, vocab]
    {
        let v = cfg.vocab_size;
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.matmul_pso);
        enc.set_buffer(0, Some(normed), 0);
        enc.set_buffer(1, Some(&bufs.lm_head_w), 0);
        enc.set_buffer(2, Some(&bufs.lm_head_b), 0);
        enc.set_buffer(3, Some(logits), 0);
        enc.set_buffer(4, Some(param_m), 0);
        enc.set_buffer(5, Some(param_vocab), 0);
        enc.set_buffer(6, Some(param_dim), 0);
        let tg_x = (v + 15) / 16;
        enc.dispatch_thread_groups(MTLSize::new(tg_x as u64, 1, 1), MTLSize::new(16, 16, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
}

// ─── Main ──────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let (cfg, tok_path, sb) = download(REPO_ID)?;
    let d = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let hd = cfg.head_dim();
    let ff = cfg.intermediate_size;
    let rot_dim = cfg.rotary_dim();
    eprintln!("  Model: {} ({} layers, dim={}, {} heads, rot_dim={})",
        REPO_ID, cfg.num_hidden_layers, d, nh, rot_dim);

    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| format!("tokenizer: {e}"))?;
    eprint_status("Loading weights");
    let st = SafeTensors::deserialize(&sb)?;
    let weights = load_weights(&st, &cfg);

    eprint_status("Initializing Metal GPU");
    let ctx = GpuContext::new();
    eprintln!("  GPU: {}", ctx.device.name());

    eprint_status("Uploading weights to GPU");
    let bufs = upload_weights(&ctx, &weights, &cfg);
    eprint_done(&format!("GPU ready ({:.1}s)", start.elapsed().as_secs_f64()));

    // Allocate scratch buffers (persistent on GPU)
    let hidden = ctx.buf_zeros(d);
    let normed = ctx.buf_zeros(d);
    let q_buf = ctx.buf_zeros(d);
    let k_buf = ctx.buf_zeros(d);
    let v_buf = ctx.buf_zeros(d);
    let fc1_out = ctx.buf_zeros(ff);
    let fc1_gelu = ctx.buf_zeros(ff);
    let attn_out = ctx.buf_zeros(d);
    let dense_out = ctx.buf_zeros(d);
    let mlp_out = ctx.buf_zeros(d);
    let residual = ctx.buf_zeros(d);
    let logits_buf = ctx.buf_zeros(cfg.vocab_size);

    // Parameter buffers (constant across forward passes, updated for position)
    let param_m = ctx.buf_from_slice(&[f32::from_bits(1u32)]);
    let param_dim = ctx.buf_from_slice(&[f32::from_bits(d as u32)]);
    let param_ff = ctx.buf_from_slice(&[f32::from_bits(ff as u32)]);
    let param_vocab = ctx.buf_from_slice(&[f32::from_bits(cfg.vocab_size as u32)]);
    let param_nh = ctx.buf_from_slice(&[f32::from_bits(nh as u32)]);
    let param_hd = ctx.buf_from_slice(&[f32::from_bits(hd as u32)]);
    let param_rotdim = ctx.buf_from_slice(&[f32::from_bits(rot_dim as u32)]);
    let param_maxseq = ctx.buf_from_slice(&[f32::from_bits(MAX_SEQ as u32)]);
    let param_pos = ctx.buf_zeros(1);
    let param_seqlen = ctx.buf_zeros(1);
    let param_eps = ctx.buf_from_slice(&[cfg.layer_norm_eps as f32]);
    let param_theta = ctx.buf_from_slice(&[cfg.rope_theta as f32]);
    let param_token = ctx.buf_zeros(1);

    let mut logits_cpu = vec![0f32; cfg.vocab_size];
    let mut rng = rand::rng();

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  Phi-1.5 Instruct (1.4B) — Apple GPU (Metal)");
    eprintln!("  Type a message, press Enter. Ctrl-C to quit.");
    eprintln!("═══════════════════════════════════════════════════════");

    loop {
        // Reset KV caches
        for lb in &bufs.layers {
            unsafe {
                std::ptr::write_bytes(lb.k_cache.contents() as *mut f32, 0, d * MAX_SEQ);
                std::ptr::write_bytes(lb.v_cache.contents() as *mut f32, 0, d * MAX_SEQ);
            }
        }

        eprintln!();
        eprint!("\x1b[1;32m> \x1b[0m");
        io::stderr().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 { break; }
        let prompt = input.trim();
        if prompt.is_empty() { continue; }

        let formatted = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
        let enc = tokenizer.encode(formatted.as_str(), false).map_err(|e| format!("encode: {e}"))?;
        let prompt_ids = enc.get_ids();

        if prompt_ids.len() + MAX_NEW_TOKENS >= MAX_SEQ {
            eprintln!("  (prompt too long)");
            continue;
        }

        let prefill_start = Instant::now();
        let mut position = 0usize;
        let mut all_tokens: Vec<u32> = Vec::new();

        // Prefill
        for &tok in prompt_ids {
            all_tokens.push(tok);
            gpu_forward(
                &ctx, &bufs, &cfg,
                &hidden, &normed, &q_buf, &k_buf, &v_buf,
                &fc1_out, &fc1_gelu, &attn_out, &dense_out, &mlp_out,
                &residual, &logits_buf,
                &param_m, &param_dim, &param_ff, &param_vocab,
                &param_nh, &param_hd, &param_rotdim, &param_maxseq,
                &param_pos, &param_seqlen, &param_eps, &param_theta,
                &param_token, tok, position,
            );
            position += 1;
        }

        // Read logits from GPU
        logits_cpu = GpuContext::read_buf(&logits_buf, cfg.vocab_size);
        let first_tok = sample(&logits_cpu, &all_tokens, &mut rng);
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        eprint!("\r\x1b[2K");

        all_tokens.push(first_tok);
        let base_text = tokenizer.decode(prompt_ids, true).map_err(|e| format!("{e}"))?;
        let mut prev_text = tokenizer.decode(&all_tokens, true).map_err(|e| format!("{e}"))?;
        if let Some(delta) = prev_text.strip_prefix(&base_text) {
            let clean = delta.replace("<|im_end|>", "").replace("<|im_start|>", "");
            print!("{clean}");
        }
        io::stdout().flush()?;

        let eos_token = tokenizer.token_to_id("<|im_end|>");
        let gen_start = Instant::now();
        let mut gen_count = 1usize;

        for _ in 0..MAX_NEW_TOKENS - 1 {
            if position >= MAX_SEQ - 1 { break; }
            let tok = *all_tokens.last().unwrap();
            if eos_token == Some(tok) { break; }

            gpu_forward(
                &ctx, &bufs, &cfg,
                &hidden, &normed, &q_buf, &k_buf, &v_buf,
                &fc1_out, &fc1_gelu, &attn_out, &dense_out, &mlp_out,
                &residual, &logits_buf,
                &param_m, &param_dim, &param_ff, &param_vocab,
                &param_nh, &param_hd, &param_rotdim, &param_maxseq,
                &param_pos, &param_seqlen, &param_eps, &param_theta,
                &param_token, tok, position,
            );
            position += 1;

            logits_cpu = GpuContext::read_buf(&logits_buf, cfg.vocab_size);
            let next = sample(&logits_cpu, &all_tokens, &mut rng);
            all_tokens.push(next);
            gen_count += 1;

            let cur = tokenizer.decode(&all_tokens, true).map_err(|e| format!("{e}"))?;
            if let Some(delta) = cur.strip_prefix(&prev_text) {
                let clean = delta.replace("<|im_end|>", "").replace("<|im_start|>", "");
                print!("{clean}");
            }
            io::stdout().flush()?;
            prev_text = cur;
        }

        let gen_elapsed = gen_start.elapsed().as_secs_f64();
        let tok_s = gen_count as f64 / gen_elapsed;
        println!();
        eprintln!("  \x1b[2m[{gen_count} tokens, {tok_s:.1} tok/s, prefill {prefill_ms:.0}ms]\x1b[0m");
    }

    Ok(())
}
