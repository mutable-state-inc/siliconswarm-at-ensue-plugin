/// GPU + ANE Pipeline: Two chips, one inference.
///
/// Proves that running GPU and ANE concurrently on different parts of a
/// transformer layer gives higher throughput than either chip alone.
///
/// - GPU: matrix multiply via Metal compute shader (simulates attention)
/// - ANE: FFN via private API inner_product (gate+SiLU+up+down)
/// - Pipeline: GPU and ANE process different layers simultaneously
///
/// Run: cargo run --release --example gpu_ane_pipeline

use ane::{Graph, Shape, TensorData};
use metal::*;
use objc2_foundation::NSQualityOfService;
use std::time::Instant;
use std::thread;

const DIM: usize = 2048;
const HIDDEN: usize = 11008;
const SEQ: usize = 64;
const NUM_LAYERS: usize = 36;

/// GPU compute work using a Metal compute shader that does real arithmetic.
/// Burns approximately `ms_target` milliseconds of GPU compute.
fn gpu_work(device: &Device, queue: &CommandQueue, ms_target: f64) {
    let shader_src = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void matmul_sim(device float *a [[buffer(0)]],
                               device float *b [[buffer(1)]],
                               device float *c [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
            float sum = 0.0;
            for (int i = 0; i < 512; i++) {
                sum += a[id * 512 + i] * b[i];
            }
            c[id] = sum;
        }
    "#;

    let lib = device.new_library_with_source(shader_src, &CompileOptions::new())
        .expect("shader compile");
    let func = lib.get_function("matmul_sim", None).expect("get function");
    let pso = device.new_compute_pipeline_state_with_function(&func).expect("PSO");

    let n = 512 * 512;
    let buf_a = device.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);
    let buf_b = device.new_buffer((512 * 4) as u64, MTLResourceOptions::StorageModeShared);
    let buf_c = device.new_buffer((512 * 4) as u64, MTLResourceOptions::StorageModeShared);

    unsafe {
        let ptr = buf_a.contents() as *mut f32;
        for i in 0..n { *ptr.add(i) = 0.01; }
        let ptr_b = buf_b.contents() as *mut f32;
        for i in 0..512 { *ptr_b.add(i) = 0.01; }
    }

    // Scale iterations to match target time. Each dispatch does 512×512 MACs.
    // Real attention at dim=2048, seq=64 is ~134M MACs. Scale accordingly.
    let iters = ((ms_target * 50.0) as usize).max(1);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pso);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_c), 0);
    for _ in 0..iters {
        enc.dispatch_threads(MTLSize::new(512, 1, 1), MTLSize::new(64, 1, 1));
    }
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn main() {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  GPU + ANE Pipeline: Two Chips, One Inference");
    eprintln!("  dim={DIM}, hidden={HIDDEN}, seq={SEQ}, layers={NUM_LAYERS}");
    eprintln!("═══════════════════════════════════════════════════════════\n");

    // Init Metal GPU
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    eprintln!("GPU: {}", device.name());

    // Compile ANE FFN kernel
    eprint!("Compiling ANE FFN kernel... ");
    let ffn_start = Instant::now();
    let mut g = Graph::new();
    let h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
    let gate = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
    let up = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
    let gs = g.sigmoid(gate);
    let gl = g.multiplication(gate, gs);
    let mix = g.multiplication(gl, up);
    let down = g.inner_product(mix, &vec![0.01; DIM * HIDDEN], HIDDEN, DIM);
    let _ = g.addition(h, down);
    let ane_exec = g.compile(NSQualityOfService::Default).expect("ANE compile");
    eprintln!("{:.0}ms", ffn_start.elapsed().as_secs_f64() * 1000.0);

    let ane_input = TensorData::with_f32(&vec![0.01; DIM * SEQ],
        Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
    let ane_output = TensorData::new(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });

    // Calibrate GPU work to match ANE FFN time
    eprint!("Calibrating GPU work... ");
    // Measure ANE FFN time
    for _ in 0..10 { let _ = ane_exec.run_cached_direct(&[&ane_input], &[&ane_output]); }
    let ane_bench_start = Instant::now();
    let ane_iters = 100;
    for _ in 0..ane_iters { let _ = ane_exec.run_cached_direct(&[&ane_input], &[&ane_output]); }
    let ane_ms = ane_bench_start.elapsed().as_secs_f64() * 1000.0 / ane_iters as f64;
    eprintln!("ANE FFN = {ane_ms:.2}ms/layer");

    // Measure GPU work time
    eprint!("Calibrating GPU attention sim... ");
    gpu_work(&device, &queue, 1.0); // warmup
    let gpu_start = Instant::now();
    let gpu_iters = 50;
    for _ in 0..gpu_iters { gpu_work(&device, &queue, ane_ms); }
    let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0 / gpu_iters as f64;
    eprintln!("{gpu_ms:.2}ms/call\n");

    let tokens = 10;

    // === TEST 1: ANE only (sequential attn-sim + FFN) ===
    eprintln!("--- ANE only: sequential (simulated attn + FFN) ---");
    let start = Instant::now();
    for _ in 0..tokens {
        for _ in 0..NUM_LAYERS {
            // "Attention" on ANE (reuse FFN kernel as proxy)
            let _ = ane_exec.run_cached_direct(&[&ane_input], &[&ane_output]);
            // FFN on ANE
            let _ = ane_exec.run_cached_direct(&[&ane_input], &[&ane_output]);
        }
    }
    let ane_only_ms = start.elapsed().as_secs_f64() * 1000.0;
    let ane_only_tok_s = tokens as f64 / start.elapsed().as_secs_f64();
    eprintln!("  {ane_only_tok_s:.1} tok/s ({ane_only_ms:.0}ms)\n");

    // === TEST 2: GPU only (sequential attn + FFN on GPU) ===
    eprintln!("--- GPU only: sequential (attn + FFN-sim on GPU) ---");
    let start = Instant::now();
    for _ in 0..tokens {
        for _ in 0..NUM_LAYERS {
            gpu_work(&device, &queue, ane_ms); // attention
            gpu_work(&device, &queue, ane_ms); // FFN
        }
    }
    let gpu_only_ms = start.elapsed().as_secs_f64() * 1000.0;
    let gpu_only_tok_s = tokens as f64 / start.elapsed().as_secs_f64();
    eprintln!("  {gpu_only_tok_s:.1} tok/s ({gpu_only_ms:.0}ms)\n");

    // === TEST 3: GPU + ANE pipeline ===
    // GPU does "attention" while ANE does FFN for the previous layer
    eprintln!("--- GPU + ANE pipeline (GPU attn || ANE FFN) ---");
    let start = Instant::now();
    for _ in 0..tokens {
        // Layer 0: GPU attention only (no previous FFN to overlap)
        gpu_work(&device, &queue, ane_ms);

        // Layers 1..N: GPU attn[N] concurrent with ANE FFN[N-1]
        for _ in 1..NUM_LAYERS {
            thread::scope(|s| {
                // ANE: FFN for previous layer
                s.spawn(|| {
                    let _ = ane_exec.run_cached_direct(&[&ane_input], &[&ane_output]);
                });
                // GPU: attention for current layer
                gpu_work(&device, &queue, ane_ms);
            });
        }

        // Last FFN on ANE
        let _ = ane_exec.run_cached_direct(&[&ane_input], &[&ane_output]);
    }
    let pipe_ms = start.elapsed().as_secs_f64() * 1000.0;
    let pipe_tok_s = tokens as f64 / start.elapsed().as_secs_f64();
    eprintln!("  {pipe_tok_s:.1} tok/s ({pipe_ms:.0}ms)\n");

    // === SUMMARY ===
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  RESULTS: {NUM_LAYERS} layers × {tokens} tokens");
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  ANE only (sequential):     {ane_only_tok_s:5.1} tok/s");
    eprintln!("  GPU only (sequential):     {gpu_only_tok_s:5.1} tok/s");
    eprintln!("  GPU + ANE pipeline:        {pipe_tok_s:5.1} tok/s  ({:+.0}% vs best single-chip)",
        (pipe_tok_s / ane_only_tok_s.max(gpu_only_tok_s) - 1.0) * 100.0);
    eprintln!("═══════════════════════════════════════════════════════════");
}
