/// Benchmark a single SwiGLU FFN layer on ANE at Qwen3.5-4B dimensions.
use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let dim = 2560;
    let hidden = 9216;
    let width = 64; // ANE minimum spatial width

    println!("Building FFN graph: dim={dim}, hidden={hidden}, width={width}");

    let mut g = Graph::new();
    let x = g.placeholder(Shape { channels: dim, height: 1, width, batch: 1 });

    // Gate: dim -> hidden
    let gate_w = g.constant(&vec![0.001f32; hidden * dim],
        Shape { channels: hidden, height: dim, width: 1, batch: 1 });
    let gate = g.convolution_2d_1x1(x, gate_w, None);

    // Up: dim -> hidden
    let up_w = g.constant(&vec![0.001f32; hidden * dim],
        Shape { channels: hidden, height: dim, width: 1, batch: 1 });
    let up = g.convolution_2d_1x1(x, up_w, None);

    // SiLU(gate) * up
    let gate_sig = g.sigmoid(gate);
    let gate_silu = g.multiplication(gate, gate_sig);
    let mix = g.multiplication(gate_silu, up);

    // Down: hidden -> dim
    let down_w = g.constant(&vec![0.001f32; dim * hidden],
        Shape { channels: dim, height: hidden, width: 1, batch: 1 });
    let _output = g.convolution_2d_1x1(mix, down_w, None);

    println!("Compiling on ANE...");
    let start = Instant::now();
    let exec = match g.compile(NSQualityOfService::Default) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("COMPILE FAILED: {e}");
            std::process::exit(1);
        }
    };
    println!("Compiled in {:?}", start.elapsed());

    // Create I/O tensors
    let input_shape = Shape { channels: dim, height: 1, width, batch: 1 };
    let output_shape = input_shape; // same shape after down projection
    let input = TensorData::with_f32(&vec![0.1f32; dim * width], input_shape);
    let output = TensorData::new(output_shape);

    // Warmup
    println!("Warming up...");
    for _ in 0..5 {
        exec.run_cached(&[&input], &[&output]).unwrap();
    }

    // Benchmark
    let n = 200;
    println!("Benchmarking {n} FFN evals...");
    let start = Instant::now();
    for _ in 0..n {
        exec.run_cached(&[&input], &[&output]).unwrap();
    }
    let dur = start.elapsed();
    let per_ms = dur.as_secs_f64() * 1000.0 / n as f64;

    // Also get hw execution time
    let hw_ns = exec.run_cached_with_stats(&[&input], &[&output]).unwrap();

    // Read output
    let out = output.read_f32();
    println!("Output first 5: {:?}", &out[..5.min(out.len())]);

    println!("\n=== FFN on ANE (Qwen3.5-4B dims) ===");
    println!("{n} evals: {:.0}ms ({:.3}ms/eval)", dur.as_secs_f64() * 1000.0, per_ms);
    println!("HW execution time: {}ns ({:.3}ms)", hw_ns, hw_ns as f64 / 1_000_000.0);
    println!("48 FFN layers: {:.1}ms", 48.0 * per_ms);
    println!("GPU baseline (all ops): ~10.5ms");
}
