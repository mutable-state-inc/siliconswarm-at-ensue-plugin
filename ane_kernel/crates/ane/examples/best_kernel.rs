use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let dim = 1536;
    let hidden = 8960;
    let layers = 28;

    println!("=== BEST KERNEL: Qwen3.5-1.7B ===\n");

    // Combine: 2L fused + seq=1024
    for (fused, seq) in [(1,512), (2,512), (1,1024), (2,1024), (2,768), (4,512)] {
        let e = {
            let mut g = Graph::new();
            let mut h = g.placeholder(Shape{batch:1,channels:dim,height:1,width:seq});
            for _ in 0..fused {
                let q = g.inner_product(h, &vec![0.001;dim*dim], dim, dim);
                let o = g.inner_product(q, &vec![0.001;dim*dim], dim, dim);
                h = g.addition(h, o);
                let gate = g.inner_product(h, &vec![0.001;hidden*dim], dim, hidden);
                let up = g.inner_product(h, &vec![0.001;hidden*dim], dim, hidden);
                let gs = g.sigmoid(gate);
                let gl = g.multiplication(gate, gs);
                let mix = g.multiplication(gl, up);
                let down = g.inner_product(mix, &vec![0.001;dim*hidden], hidden, dim);
                h = g.addition(h, down);
            }
            match g.compile(NSQualityOfService::Default) {
                Ok(e) => e,
                Err(_) => { println!("{fused}L×seq{seq:4}:  COMPILE FAIL"); continue; }
            }
        };
        let input = TensorData::with_f32(&vec![0.01;dim*seq], Shape{batch:1,channels:dim,height:1,width:seq});
        let output = TensorData::new(Shape{batch:1,channels:dim,height:1,width:seq});
        for _ in 0..5 { e.run_cached_direct(&[&input], &[&output]).unwrap(); }
        let n = 30;
        let start = Instant::now();
        for _ in 0..n { e.run_cached_direct(&[&input], &[&output]).unwrap(); }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let dispatches = (layers as f64 / fused as f64).ceil();
        let total = dispatches * ms;
        let toks = seq as f64 / total * 1000.0;
        let per_layer = ms / fused as f64;
        println!("{fused}L×seq{seq:4}: {ms:.1}ms/{fused}L ({per_layer:.1}ms/L) ×{dispatches:.0} = {total:.0}ms → {toks:.0} tok/s");
    }
    
    println!("\nBaseline: 1L×seq512 = 13.7ms/L × 28 = 383ms → 1337 tok/s");
    println!("CoreML fp16:                   14.0ms/L × 28 = 391ms → 1309 tok/s");
    println!("CoreML int8:                    7.9ms/L × 28 = 220ms → 2323 tok/s");
}
