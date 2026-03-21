use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let dim = 2560;
    let hidden = 9216;
    let seq = 512;

    // Full layer: Q + O + gate + up + silu + down + residuals
    let exec = {
        let mut g = Graph::new();
        let x = g.placeholder(Shape{batch:1,channels:dim,height:1,width:seq});
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
    let input = TensorData::with_f32(&vec![0.01;dim*seq], Shape{batch:1,channels:dim,height:1,width:seq});
    let output = TensorData::new(Shape{batch:1,channels:dim,height:1,width:seq});
    for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
    let n = 30;
    let start = Instant::now();
    for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let total = 32.0 * ms;
    let toks = seq as f64 / total * 1000.0;

    println!("=== FAIR COMPARISON: FULL LAYER, SEQ={seq} ===\n");
    println!("Private API (inner_product fp16):");
    println!("  {ms:.2}ms/layer → 32L = {total:.0}ms → {toks:.0} tok/s\n");
    println!("CoreML (from earlier benchmark):");
    println!("  fp16 ALL:     27.27ms/layer → 32L = 873ms →  587 tok/s");
    println!("  int8 ALL:     15.00ms/layer → 32L = 480ms → 1067 tok/s");
    println!("  int4 ALL:     12.65ms/layer → 32L = 405ms → 1265 tok/s\n");
    
    if ms < 12.65 {
        println!(">>> PRIVATE API BEATS COREML INT4! {ms:.2}ms < 12.65ms <<<");
        println!(">>> {toks:.0} tok/s vs 1265 tok/s <<<");
    } else if ms < 15.00 {
        println!(">>> PRIVATE API BEATS COREML INT8! {ms:.2}ms < 15.00ms <<<");
    } else if ms < 27.27 {
        println!(">>> PRIVATE API BEATS COREML FP16! {ms:.2}ms < 27.27ms <<<");
    } else {
        println!("CoreML wins at all quantization levels.");
    }
}
