/// FFN on ANE with 3 separate matmul kernels (no fusion).
use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn build_matmul(ic: usize, oc: usize, seq: usize) -> Graph {
    let sp = seq + oc;
    let mut g = Graph::new();
    let input = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: sp });
    let acts = g.slice(input, [0, 0, 0, 0], [1, ic, 1, seq]);
    let wts = g.slice(input, [0, 0, 0, seq], [1, ic, 1, oc]);
    let acts_r = g.reshape(acts, Shape { batch: 1, channels: 1, height: ic, width: seq });
    let acts_t = g.transpose(acts_r, [0, 1, 3, 2]);
    let wts_r = g.reshape(wts, Shape { batch: 1, channels: 1, height: ic, width: oc });
    let mm = g.matrix_multiplication(acts_t, wts_r, false, false);
    let mm_t = g.transpose(mm, [0, 1, 3, 2]);
    let _out = g.reshape(mm_t, Shape { batch: 1, channels: oc, height: 1, width: seq });
    g
}

fn main() {
    let dim = 2560;
    let hidden = 9216;
    let seq = 64;

    println!("Building 3 separate matmul kernels for SwiGLU FFN...");

    // Kernel 1: gate projection (dim -> hidden)
    let start = Instant::now();
    let gate_exec = build_matmul(dim, hidden, seq).compile(NSQualityOfService::Default).unwrap();
    println!("  gate ({dim}x{hidden}): compiled in {:?}", start.elapsed());

    // Kernel 2: up projection (dim -> hidden)
    let start = Instant::now();
    let up_exec = build_matmul(dim, hidden, seq).compile(NSQualityOfService::Default).unwrap();
    println!("  up ({dim}x{hidden}): compiled in {:?}", start.elapsed());

    // Kernel 3: down projection (hidden -> dim)
    let start = Instant::now();
    let down_exec = build_matmul(hidden, dim, seq).compile(NSQualityOfService::Default).unwrap();
    println!("  down ({hidden}x{dim}): compiled in {:?}", start.elapsed());

    // Create I/O tensors
    let gate_sp = seq + hidden;
    let up_sp = seq + hidden;
    let down_sp = seq + dim;

    let gate_in = TensorData::with_f32(&vec![0.01f32; dim * gate_sp], Shape { batch: 1, channels: dim, height: 1, width: gate_sp });
    let gate_out = TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: seq });
    let up_in = TensorData::with_f32(&vec![0.01f32; dim * up_sp], Shape { batch: 1, channels: dim, height: 1, width: up_sp });
    let up_out = TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: seq });
    let down_in = TensorData::with_f32(&vec![0.01f32; hidden * down_sp], Shape { batch: 1, channels: hidden, height: 1, width: down_sp });
    let down_out = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });

    // Warmup
    for _ in 0..3 {
        gate_exec.run_cached(&[&gate_in], &[&gate_out]).unwrap();
        up_exec.run_cached(&[&up_in], &[&up_out]).unwrap();
        down_exec.run_cached(&[&down_in], &[&down_out]).unwrap();
    }

    // Benchmark individual kernels
    let n = 100;
    let start = Instant::now();
    for _ in 0..n { gate_exec.run_cached(&[&gate_in], &[&gate_out]).unwrap(); }
    let gate_ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;

    let start = Instant::now();
    for _ in 0..n { up_exec.run_cached(&[&up_in], &[&up_out]).unwrap(); }
    let up_ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;

    let start = Instant::now();
    for _ in 0..n { down_exec.run_cached(&[&down_in], &[&down_out]).unwrap(); }
    let down_ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;

    let ffn_ms = gate_ms + up_ms + down_ms;

    // Benchmark full FFN (all 3 sequentially)
    let start = Instant::now();
    for _ in 0..n {
        gate_exec.run_cached(&[&gate_in], &[&gate_out]).unwrap();
        up_exec.run_cached(&[&up_in], &[&up_out]).unwrap();
        // SiLU would happen on CPU here — not benchmarked
        down_exec.run_cached(&[&down_in], &[&down_out]).unwrap();
    }
    let total_ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;

    // HW execution time for one gate matmul
    let hw_ns = gate_exec.run_cached_with_stats(&[&gate_in], &[&gate_out]).unwrap();

    println!("\n=== FFN on ANE (Qwen3.5-4B: {dim}→{hidden}→{dim}) ===");
    println!("  gate:  {gate_ms:.3}ms  (HW: {:.3}ms)", hw_ns as f64 / 1_000_000.0);
    println!("  up:    {up_ms:.3}ms");
    println!("  down:  {down_ms:.3}ms");
    println!("  total: {ffn_ms:.3}ms (sum) / {total_ms:.3}ms (sequential)");
    println!("\n48 FFN layers: {:.1}ms", 48.0 * total_ms);
    println!("GPU baseline (ALL ops including attention): ~10.5ms");
    println!("ANE dispatch overhead: {:.3}ms per call", total_ms - (hw_ns as f64 / 1_000_000.0 * 3.0));
}
