/// Inference benchmark matching bench.rustane.org format.
/// Single forward pass through full model, seq=512.
use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn build_model(dim: usize, hidden: usize, num_layers: usize, seq: usize) -> Option<ane::Executable> {
    let mut g = Graph::new();
    let mut h = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
    
    for _ in 0..num_layers {
        // Attention: Q + O projections
        let q = g.inner_product(h, &vec![0.001; dim*dim], dim, dim);
        let o = g.inner_product(q, &vec![0.001; dim*dim], dim, dim);
        h = g.addition(h, o);
        
        // FFN: gate + up + silu + down
        let gate = g.inner_product(h, &vec![0.001; hidden*dim], dim, hidden);
        let up = g.inner_product(h, &vec![0.001; hidden*dim], dim, hidden);
        let gs = g.sigmoid(gate);
        let gl = g.multiplication(gate, gs);
        let mix = g.multiplication(gl, up);
        let down = g.inner_product(mix, &vec![0.001; dim*hidden], hidden, dim);
        h = g.addition(h, down);
    }
    
    g.compile(NSQualityOfService::Default).ok()
}

fn main() {
    let seq = 512;
    
    // Model configs matching rustane's scales
    // (dim, hidden, layers, approx_params_name)
    let configs = [
        (1536, 4096, 14, "600M"),
        (2048, 5504, 20, "1.5B"),
        (2560, 6912, 24, "3B"),  
        (3072, 8192, 32, "5B"),
        (4096, 11008, 32, "10B"),
    ];
    
    println!("=== INFERENCE BENCHMARK (bench.rustane.org format) ===");
    println!("Chip: Apple M1 Max 64GB");
    println!("Sequence length: {seq}\n");
    println!("{:>6} {:>6} {:>4} {:>4} {:>10} {:>8}", "Scale", "Dim", "Hid", "Lay", "ms", "tok/s");
    println!("{}", "-".repeat(50));
    
    for (dim, hidden, layers, name) in configs {
        // Try to compile — may fail if model too large
        // Build per-layer since full model may not compile at once
        let mut layer_exec = None;
        {
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
            layer_exec = g.compile(NSQualityOfService::Default).ok();
        }
        
        match layer_exec {
            Some(exec) => {
                let input = TensorData::with_f32(&vec![0.01; dim * seq],
                    Shape { batch: 1, channels: dim, height: 1, width: seq });
                let output = TensorData::new(
                    Shape { batch: 1, channels: dim, height: 1, width: seq });
                
                // Warmup
                for _ in 0..3 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                
                // Benchmark single layer
                let n = 20;
                let start = Instant::now();
                for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let layer_ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
                
                let total_ms = layers as f64 * layer_ms;
                let tok_s = seq as f64 / total_ms * 1000.0;
                let params_m = (layers as f64 * (2.0 * dim as f64 * dim as f64 + 3.0 * dim as f64 * hidden as f64)) / 1e6;
                
                println!("{name:>6} {dim:>6} {hidden:>4} {layers:>4} {total_ms:>9.0}ms {tok_s:>7.0}");
                
                // Also try fused multi-layer
                if let Some(fused) = build_model(dim, hidden, layers.min(12), seq) {
                    let input2 = TensorData::with_f32(&vec![0.01; dim * seq],
                        Shape { batch: 1, channels: dim, height: 1, width: seq });
                    let output2 = TensorData::new(
                        Shape { batch: 1, channels: dim, height: 1, width: seq });
                    for _ in 0..2 { fused.run_cached_direct(&[&input2], &[&output2]).unwrap(); }
                    let n2 = 10;
                    let start = Instant::now();
                    for _ in 0..n2 { fused.run_cached_direct(&[&input2], &[&output2]).unwrap(); }
                    let fused_ms = start.elapsed().as_secs_f64() * 1000.0 / n2 as f64;
                    let fused_layers = layers.min(12);
                    let remaining = layers - fused_layers;
                    let fused_total = fused_ms + remaining as f64 * layer_ms;
                    let fused_dispatches = (layers as f64 / fused_layers as f64).ceil();
                    let fused_total2 = fused_dispatches * fused_ms;
                    let fused_toks = seq as f64 / fused_total2 * 1000.0;
                    println!("{:>6} (fused {fused_layers}L × {fused_dispatches:.0} dispatches) {fused_total2:>9.0}ms {fused_toks:>7.0}", "");
                }
            }
            None => println!("{name:>6} {dim:>6} {hidden:>4} {layers:>4}      FAIL"),
        }
    }
    
    println!("\nComparison: M4 Max 128GB results from bench.rustane.org:");
    println!("  10B: 4696ms, 109 tok/s");
    println!("  20B: 40933ms, 13 tok/s");
}
