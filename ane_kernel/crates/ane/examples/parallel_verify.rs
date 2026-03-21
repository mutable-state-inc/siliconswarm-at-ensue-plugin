use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let dim = 2560;
    let hidden = 9216;
    let candidates = 64; // test 64 candidate continuations simultaneously

    // Build a full layer that processes 64 candidates at once
    let mut g = Graph::new();
    let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: candidates });
    let q = g.inner_product(x, &vec![0.001; dim*dim], dim, dim);
    let o = g.inner_product(q, &vec![0.001; dim*dim], dim, dim);
    let h = g.addition(x, o);
    let gate = g.inner_product(h, &vec![0.001; hidden*dim], dim, hidden);
    let up = g.inner_product(h, &vec![0.001; hidden*dim], dim, hidden);
    let gs = g.sigmoid(gate);
    let gl = g.multiplication(gate, gs);
    let mix = g.multiplication(gl, up);
    let down = g.inner_product(mix, &vec![0.001; dim*hidden], hidden, dim);
    let _out = g.addition(h, down);

    let exec = g.compile(NSQualityOfService::Default).unwrap();
    let input = TensorData::with_f32(&vec![0.01; dim*candidates],
        Shape { batch: 1, channels: dim, height: 1, width: candidates });
    let output = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: candidates });
    for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }

    let n = 50;
    let start = Instant::now();
    for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
    let layer_ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let full_model_ms = 32.0 * layer_ms;

    // Probability analysis:
    // GPU generates token N, producing logits over 152K vocab
    // Take top-64 most likely tokens
    // ANE runs full model for all 64 candidates simultaneously
    // If ANY candidate matches GPU's token N+1, we skip that GPU step
    //
    // The probability that the correct next token is in top-64 of 152K:
    // For greedy decode (temp=0), top-1 accuracy is 100% by definition
    // For the model's own distribution, top-64 captures a large fraction
    // Typical top-64 coverage: 80-95% of probability mass
    //
    // If match probability = p:
    //   Without parallel verify: 1 token per GPU step (10.5ms)  
    //   With parallel verify: (1+p) tokens per max(GPU step, ANE step)
    //   ANE step = full_model_ms for 64 candidates
    //   If ANE_time <= GPU_time: effective tok/s = (1+p) * GPU_tok/s

    println!("=== PARALLEL VERIFICATION: 64 CANDIDATES ON ANE ===");
    println!("ANE per-layer ({candidates} candidates): {layer_ms:.2}ms");
    println!("ANE full model (32 layers):             {full_model_ms:.1}ms");
    println!("GPU single token:                       10.5ms");
    println!();
    
    if full_model_ms <= 10.5 {
        println!("ANE fits within GPU time! Zero overhead.");
        for p in [0.5, 0.7, 0.8, 0.9, 0.95] {
            let effective = (1.0 + p) * 95.0;
            println!("  p={p:.2} (top-64 hit rate): {effective:.0} tok/s ({:.1}x)", effective/95.0);
        }
    } else {
        println!("ANE takes {full_model_ms:.1}ms > GPU 10.5ms");
        println!("Need to reduce layers or use fewer candidates.");
        
        // How many layers fit in GPU time?
        let max_layers = (10.5 / layer_ms).floor() as usize;
        println!("Max ANE layers in 10.5ms: {max_layers}");
        
        // What about fewer candidates?
        // batch=64 is the ANE minimum, can't go lower
        println!("Minimum batch is 64 (ANE hardware limit)");
    }
}
