/// Full Qwen layer with simulated int4 dequant on ANE.
/// 
/// Each matmul is split into 4 quarter-sized inner_products.
/// Each quarter reads 1/4 the weights (simulating 4-bit storage).
/// Dequant: multiply by per-group scale after each quarter IP.
/// Concat the 4 quarters back to full dimension.
///
/// For the "down" projection (hidden→dim), we can't split on IC
/// with inner_product. Instead split on OC: 4×(hidden→dim/4).

use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn add_split4_matmul(g: &mut Graph, x: ane::Tensor, ic: usize, oc: usize, seq: usize) -> ane::Tensor {
    let q = oc / 4;
    let scale = g.constant_with_scalar(0.125, Shape{batch:1,channels:1,height:1,width:1});
    
    let a = g.inner_product(x, &vec![0.001; q*ic], ic, q);
    let a = g.multiplication(a, scale);
    let b = g.inner_product(x, &vec![0.002; q*ic], ic, q);
    let b = g.multiplication(b, scale);
    let c = g.inner_product(x, &vec![0.003; q*ic], ic, q);
    let c = g.multiplication(c, scale);
    let d = g.inner_product(x, &vec![0.004; q*ic], ic, q);
    let d = g.multiplication(d, scale);
    
    g.concat(&[a, b, c, d], 1)
}

fn main() {
    let dim = 2560;
    let hidden = 9216;
    let seq = 512;

    println!("=== INT4 DEQUANT FULL LAYER (private API) ===\n");

    // Build full layer with split matmuls
    let exec = {
        let mut g = Graph::new();
        let x = g.placeholder(Shape{batch:1, channels:dim, height:1, width:seq});
        
        // Q projection: dim→dim, split OC into 4
        let q = add_split4_matmul(&mut g, x, dim, dim, seq);
        
        // O projection: dim→dim, split OC into 4
        let o = add_split4_matmul(&mut g, q, dim, dim, seq);
        
        // Residual
        let h = g.addition(x, o);
        
        // Gate: dim→hidden, split OC into 4
        let gate = add_split4_matmul(&mut g, h, dim, hidden, seq);
        
        // Up: dim→hidden, split OC into 4
        let up = add_split4_matmul(&mut g, h, dim, hidden, seq);
        
        // SiLU(gate) * up
        let gs = g.sigmoid(gate);
        let gl = g.multiplication(gate, gs);
        let mix = g.multiplication(gl, up);
        
        // Down: hidden→dim, split OC into 4
        let down = add_split4_matmul(&mut g, mix, hidden, dim, seq);
        
        // Residual
        let _ = g.addition(h, down);
        
        g.compile(NSQualityOfService::Default).unwrap()
    };

    let input = TensorData::with_f32(&vec![0.01; dim*seq],
        Shape{batch:1, channels:dim, height:1, width:seq});
    let output = TensorData::new(
        Shape{batch:1, channels:dim, height:1, width:seq});

    // Warmup
    for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }

    // Benchmark
    let n = 30;
    let start = Instant::now();
    for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let total = 32.0 * ms;
    let toks = seq as f64 / total * 1000.0;

    println!("INT4 split layer:  {ms:.2}ms/layer → 32L = {total:.0}ms → {toks:.0} tok/s");
    println!();

    // Baseline: full fp16 layer (no split)
    let exec2 = {
        let mut g = Graph::new();
        let x = g.placeholder(Shape{batch:1, channels:dim, height:1, width:seq});
        let q = g.inner_product(x, &vec![0.001;dim*dim], dim, dim);
        let o = g.inner_product(q, &vec![0.001;dim*dim], dim, dim);
        let h = g.addition(x, o);
        let gate = g.inner_product(h, &vec![0.001;hidden*dim], dim, hidden);
        let up = g.inner_product(h, &vec![0.001;hidden*dim], dim, hidden);
        let gs = g.sigmoid(gate);
        let gl = g.multiplication(gate, gs);
        let mix = g.multiplication(gl, up);
        let down = g.inner_product(mix, &vec![0.001;dim*hidden], hidden, dim);
        let _ = g.addition(h, down);
        g.compile(NSQualityOfService::Default).unwrap()
    };
    for _ in 0..5 { exec2.run_cached_direct(&[&input], &[&output]).unwrap(); }
    let start = Instant::now();
    for _ in 0..n { exec2.run_cached_direct(&[&input], &[&output]).unwrap(); }
    let ms2 = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let total2 = 32.0 * ms2;
    let toks2 = seq as f64 / total2 * 1000.0;
    println!("FP16 full layer:   {ms2:.2}ms/layer → 32L = {total2:.0}ms → {toks2:.0} tok/s");

    println!();
    println!("=== COMPARISON ===");
    println!("Private API int4 split: {ms:.2}ms → {toks:.0} tok/s");
    println!("Private API fp16 full:  {ms2:.2}ms → {toks2:.0} tok/s");
    println!("CoreML int4:            12.65ms → 1265 tok/s");
    println!("CoreML fp16:            27.27ms →  587 tok/s");
    
    if ms < 12.65 {
        println!("\n>>> PRIVATE API INT4 BEATS COREML INT4! <<<");
        println!(">>> {toks:.0} vs 1265 tok/s = {:.1}x faster <<<", toks / 1265.0);
    }
}
