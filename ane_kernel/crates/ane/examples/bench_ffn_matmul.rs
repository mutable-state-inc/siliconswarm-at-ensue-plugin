/// Full SwiGLU FFN on ANE using dynamic-weight matmul pattern.
/// This is the correct approach for Qwen3.5-4B inference.
use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let dim = 2560;
    let hidden = 9216;
    let seq = 64; // spatial width (ANE min)

    // Pack all 3 weight matrices + activation into one IOSurface:
    // [1, dim, 1, seq + hidden + hidden + hidden] for gate_w, up_w
    // Actually, for SwiGLU we need 3 projections:
    // gate: dim -> hidden, up: dim -> hidden (both from same input)
    // down: hidden -> dim (from mix)
    //
    // Approach: build 3 separate kernels and chain them
    // Kernel 1: x -> gate, up (fused: one matmul for [dim -> 2*hidden])
    // Kernel 2: SiLU(gate) * up -> down projection

    // === Kernel 1: gate+up projection (fused) ===
    let fused_oc = hidden * 2; // gate + up concatenated
    let sp1 = seq + fused_oc;
    let mut g1 = Graph::new();
    let input1 = g1.placeholder(Shape { batch: 1, channels: dim, height: 1, width: sp1 });
    let acts1 = g1.slice(input1, [0, 0, 0, 0], [1, dim, 1, seq]);
    let wts1 = g1.slice(input1, [0, 0, 0, seq], [1, dim, 1, fused_oc]);
    let acts1_r = g1.reshape(acts1, Shape { batch: 1, channels: 1, height: dim, width: seq });
    let acts1_t = g1.transpose(acts1_r, [0, 1, 3, 2]);
    let wts1_r = g1.reshape(wts1, Shape { batch: 1, channels: 1, height: dim, width: fused_oc });
    let mm1 = g1.matrix_multiplication(acts1_t, wts1_r, false, false);
    // mm1: [1, 1, seq, 2*hidden]
    // Split into gate [seq, hidden] and up [seq, hidden]
    let gate = g1.slice(mm1, [0, 0, 0, 0], [1, 1, seq, hidden]);
    let up = g1.slice(mm1, [0, 0, 0, hidden], [1, 1, seq, hidden]);
    // SiLU(gate) * up
    let gate_sig = g1.sigmoid(gate);
    let gate_silu = g1.multiplication(gate, gate_sig);
    let mix = g1.multiplication(gate_silu, up);
    // Reshape to output: [1, 1, seq, hidden] -> [1, hidden, 1, seq]
    let mix_t = g1.transpose(mix, [0, 1, 3, 2]);
    let _mix_out = g1.reshape(mix_t, Shape { batch: 1, channels: hidden, height: 1, width: seq });

    println!("Compiling gate+up kernel (dim={dim}, hidden={hidden})...");
    let start = Instant::now();
    let exec1 = g1.compile(NSQualityOfService::Default).unwrap();
    println!("  Compiled in {:?}", start.elapsed());

    // === Kernel 2: down projection ===
    let sp2 = seq + dim;
    let mut g2 = Graph::new();
    let input2 = g2.placeholder(Shape { batch: 1, channels: hidden, height: 1, width: sp2 });
    let acts2 = g2.slice(input2, [0, 0, 0, 0], [1, hidden, 1, seq]);
    let wts2 = g2.slice(input2, [0, 0, 0, seq], [1, hidden, 1, dim]);
    let acts2_r = g2.reshape(acts2, Shape { batch: 1, channels: 1, height: hidden, width: seq });
    let acts2_t = g2.transpose(acts2_r, [0, 1, 3, 2]);
    let wts2_r = g2.reshape(wts2, Shape { batch: 1, channels: 1, height: hidden, width: dim });
    let mm2 = g2.matrix_multiplication(acts2_t, wts2_r, false, false);
    let mm2_t = g2.transpose(mm2, [0, 1, 3, 2]);
    let _out2 = g2.reshape(mm2_t, Shape { batch: 1, channels: dim, height: 1, width: seq });

    println!("Compiling down kernel...");
    let start = Instant::now();
    let exec2 = g2.compile(NSQualityOfService::Default).unwrap();
    println!("  Compiled in {:?}", start.elapsed());

    // Create I/O tensors
    let in1 = TensorData::with_f32(&vec![0.01f32; dim * sp1],
        Shape { batch: 1, channels: dim, height: 1, width: sp1 });
    let out1 = TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: seq });
    let in2 = TensorData::with_f32(&vec![0.01f32; hidden * sp2],
        Shape { batch: 1, channels: hidden, height: 1, width: sp2 });
    let out2 = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });

    // Warmup
    println!("Warming up...");
    for _ in 0..3 {
        exec1.run_cached(&[&in1], &[&out1]).unwrap();
        exec2.run_cached(&[&in2], &[&out2]).unwrap();
    }

    // Benchmark: full FFN = gate+up + down
    let n = 100;
    println!("Benchmarking {n} full FFN (gate+up+SiLU + down)...");
    let start = Instant::now();
    for _ in 0..n {
        exec1.run_cached(&[&in1], &[&out1]).unwrap();
        exec2.run_cached(&[&in2], &[&out2]).unwrap();
    }
    let dur = start.elapsed();
    let per_ms = dur.as_secs_f64() * 1000.0 / n as f64;

    println!("\n=== SwiGLU FFN on ANE (Qwen3.5-4B) ===");
    println!("{n} FFNs: {:.0}ms ({:.2}ms per FFN)", dur.as_secs_f64() * 1000.0, per_ms);
    println!("48 FFN layers: {:.1}ms", 48.0 * per_ms);
    println!("GPU baseline (ALL ops): ~10.5ms");
}
