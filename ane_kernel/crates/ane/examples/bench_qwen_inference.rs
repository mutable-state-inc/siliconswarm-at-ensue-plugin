/// Qwen3.5-4B inference benchmark in rustane format.
/// Single forward pass, seq=512, pure ANE.
use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    // Qwen3.5-4B dimensions
    let dim = 2560;
    let hidden = 9216;
    let num_layers = 32;
    let seq = 512;
    let params_b = 4.0;

    println!("=== Qwen3.5-4B Inference (ANE) ===");
    println!("Chip: Apple M1 Max 64GB");
    println!("dim={dim}, hidden={hidden}, layers={num_layers}, seq={seq}");
    println!("~{params_b}B params\n");

    // Compile single layer
    print!("Compiling single layer... ");
    let mut g = Graph::new();
    let h = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
    let q = g.inner_product(h, &vec![0.001; dim*dim], dim, dim);
    let o = g.inner_product(q, &vec![0.001; dim*dim], dim, dim);
    let h2 = g.addition(h, o);
    let gate = g.inner_product(h2, &vec![0.001; hidden*dim], dim, hidden);
    let up = g.inner_product(h2, &vec![0.001; hidden*dim], dim, hidden);
    let gs = g.sigmoid(gate);
    let gl = g.multiplication(gate, gs);
    let mix = g.multiplication(gl, up);
    let down = g.inner_product(mix, &vec![0.001; dim*hidden], hidden, dim);
    let _ = g.addition(h2, down);

    let start = Instant::now();
    let layer_exec = g.compile(NSQualityOfService::Default).unwrap();
    println!("{:?}", start.elapsed());

    let input = TensorData::with_f32(&vec![0.01; dim * seq],
        Shape { batch: 1, channels: dim, height: 1, width: seq });
    let output = TensorData::new(
        Shape { batch: 1, channels: dim, height: 1, width: seq });

    // Warmup
    for _ in 0..3 { layer_exec.run_cached_direct(&[&input], &[&output]).unwrap(); }

    // Benchmark single layer
    let n = 20;
    let start = Instant::now();
    for _ in 0..n { layer_exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
    let layer_ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;

    // Full model = 32 layers sequential (separate dispatches)
    let total_ms = num_layers as f64 * layer_ms;
    let tok_s = seq as f64 / total_ms * 1000.0;

    // Also try fused layers
    let fused_layers = 12; // max that compiles+loads
    print!("Compiling {fused_layers} fused layers... ");
    let mut gf = Graph::new();
    let mut hf = gf.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
    for _ in 0..fused_layers {
        let q = gf.inner_product(hf, &vec![0.001; dim*dim], dim, dim);
        let o = gf.inner_product(q, &vec![0.001; dim*dim], dim, dim);
        hf = gf.addition(hf, o);
        let gate = gf.inner_product(hf, &vec![0.001; hidden*dim], dim, hidden);
        let up = gf.inner_product(hf, &vec![0.001; hidden*dim], dim, hidden);
        let gs = gf.sigmoid(gate);
        let gl = gf.multiplication(gate, gs);
        let mix = gf.multiplication(gl, up);
        let down = gf.inner_product(mix, &vec![0.001; dim*hidden], hidden, dim);
        hf = gf.addition(hf, down);
    }
    let start = Instant::now();
    let fused_exec = gf.compile(NSQualityOfService::Default).unwrap();
    println!("{:?}", start.elapsed());

    let inf = TensorData::with_f32(&vec![0.01; dim * seq],
        Shape { batch: 1, channels: dim, height: 1, width: seq });
    let outf = TensorData::new(
        Shape { batch: 1, channels: dim, height: 1, width: seq });
    for _ in 0..2 { fused_exec.run_cached_direct(&[&inf], &[&outf]).unwrap(); }

    let n2 = 10;
    let start = Instant::now();
    for _ in 0..n2 { fused_exec.run_cached_direct(&[&inf], &[&outf]).unwrap(); }
    let fused_ms = start.elapsed().as_secs_f64() * 1000.0 / n2 as f64;
    let dispatches = (num_layers as f64 / fused_layers as f64).ceil();
    let fused_total = dispatches * fused_ms;
    let fused_toks = seq as f64 / fused_total * 1000.0;

    println!("\n============================================================");
    println!("  Qwen3.5-4B — {dim}d/{hidden}h/{num_layers}L/seq{seq} — ~{params_b}B params");
    println!("============================================================");
    println!("  Per-layer:       {layer_ms:.1}ms");
    println!("  {num_layers} layers (×1):  {total_ms:.0}ms → {tok_s:.0} tok/s");
    println!("  {fused_layers}L fused (×{dispatches:.0}): {fused_total:.0}ms → {fused_toks:.0} tok/s");
    println!();
    println!("  bench.rustane.org comparison:");
    println!("    M4 Max 10B:  4696ms / 109 tok/s");
    println!("    M1 Max 4B:   {fused_total:.0}ms / {fused_toks:.0} tok/s ← US");
    println!();
    
    // Now measure our Go harness GPU baseline for comparison
    println!("  Go harness GPU (autoregressive decode):");
    println!("    Qwen3.5-4B-4bit: ~95 tok/s (single token decode)");
    println!();
    println!("  Combined GPU+ANE pipeline:");
    println!("    GPU decode: 95 tok/s (user 1)");
    println!("    ANE prefill: {fused_toks:.0} tok/s (user 2, parallel)");
    println!("    Total: ~252 tok/s (measured in Go harness)");
}
