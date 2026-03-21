use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let dim = 2560;
    let hidden = 9216;
    let seq = 64;
    
    // Build single Qwen layer: Q + O + gate + up + down using inner_product
    // All constant weights — baked into compiled model
    let mut g = Graph::new();
    let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
    
    // Q projection
    let q = g.inner_product(x, &vec![0.001; dim * dim], dim, dim);
    // O projection (simplified attention: just Q -> O)
    let o = g.inner_product(q, &vec![0.001; dim * dim], dim, dim);
    // Residual
    let h = g.addition(x, o);
    
    // FFN gate
    let gate = g.inner_product(h, &vec![0.001; hidden * dim], dim, hidden);
    // FFN up
    let up = g.inner_product(h, &vec![0.001; hidden * dim], dim, hidden);
    // SiLU(gate) * up
    let gate_sig = g.sigmoid(gate);
    let gate_silu = g.multiplication(gate, gate_sig);
    let mix = g.multiplication(gate_silu, up);
    // FFN down
    let down = g.inner_product(mix, &vec![0.001; dim * hidden], hidden, dim);
    // Residual
    let _out = g.addition(h, down);
    
    println!("Compiling single Qwen layer (inner_product, dim={dim}, hidden={hidden})...");
    let start = Instant::now();
    match g.compile(NSQualityOfService::Default) {
        Ok(exec) => {
            println!("  COMPILE OK ({:?})", start.elapsed());
            
            let input = TensorData::with_f32(&vec![0.01; dim * seq], Shape { batch: 1, channels: dim, height: 1, width: seq });
            let output = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });
            for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
            
            let n = 100;
            let start = Instant::now();
            for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
            let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
            
            println!("\n=== SINGLE QWEN LAYER ON ANE (inner_product) ===");
            println!("{ms:.3}ms per layer");
            println!("32 layers: {:.1}ms ({:.0} tok/s)", 32.0 * ms, 1000.0 / (32.0 * ms));
            println!("GPU baseline: ~10.5ms (95 tok/s)");
        }
        Err(e) => println!("  COMPILE FAILED: {e}"),
    }
}
