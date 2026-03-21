/// INT4 dequantization on ANE using constant weights.
/// 
/// Pack 4-bit weights into fp16 constants at 1/4 the output channels.
/// On ANE: 
///   1. inner_product(x, packed_weights) → compressed output [hidden/4]
///   2. Unpack via multiply + add (simulating scale * int4 + zero_point)
///   3. Repeat 4x for each quarter, concat results
///
/// This reads 1/4 the weight data while doing 4x the inner_products
/// + some elementwise dequant ops. Net effect depends on whether
/// the bandwidth savings exceed the extra compute.

use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let dim = 2560;
    let hidden = 9216;
    let seq = 512;
    let quarter = hidden / 4; // 2304

    println!("=== INT4 DEQUANT ON ANE ===\n");

    // Approach: 4 quarter-sized inner_products + scale/offset + concat
    // Each inner_product reads dim*quarter fp16 weights = 11.8MB
    // Total weight reads: 4 * 11.8MB = 47.2MB (same as fp16!)
    // But each inner_product is faster because ANE caches smaller blocks better
    
    // Actually — for true int4, we'd pack 4 weights per fp16 value
    // and have 1 inner_product at 1/4 channels, then unpack.
    // Let me test: 1 inner_product at 1/4 + elementwise unpack
    
    let mut g = Graph::new();
    let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
    
    // "Packed" inner_product: reads 1/4 the weights
    let packed = g.inner_product(x, &vec![0.001; quarter * dim], dim, quarter);
    
    // Simulate dequant: expand quarter→hidden via elementwise
    // In real int4: unpack bits. Here: just scale + tile
    let scale = g.constant_with_scalar(0.1, Shape{batch:1,channels:1,height:1,width:1});
    let offset = g.constant_with_scalar(0.0, Shape{batch:1,channels:1,height:1,width:1});
    let scaled = g.multiplication(packed, scale);
    let _dequant = g.addition(scaled, offset);
    
    // This gives us [1, quarter, 1, seq] — not full hidden dim.
    // For a full simulation we need 4 of these concatenated:
    
    let mut g2 = Graph::new();
    let x2 = g2.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
    let s = g2.constant_with_scalar(0.1, Shape{batch:1,channels:1,height:1,width:1});
    
    // 4 quarter inner_products with different weights (simulating 4 int4 groups)
    let p0 = g2.inner_product(x2, &vec![0.001; quarter * dim], dim, quarter);
    let p1 = g2.inner_product(x2, &vec![0.002; quarter * dim], dim, quarter);
    let p2 = g2.inner_product(x2, &vec![0.003; quarter * dim], dim, quarter);
    let p3 = g2.inner_product(x2, &vec![0.004; quarter * dim], dim, quarter);
    
    // Scale each (simulating per-group dequant)
    let d0 = g2.multiplication(p0, s);
    let d1 = g2.multiplication(p1, s);
    let d2 = g2.multiplication(p2, s);
    let d3 = g2.multiplication(p3, s);
    
    // Concat to full hidden dim
    let _full = g2.concat(&[d0, d1, d2, d3], 1);
    
    match g2.compile(NSQualityOfService::Default) {
        Ok(exec) => {
            let input = TensorData::with_f32(&vec![0.01; dim*seq], Shape{batch:1,channels:dim,height:1,width:seq});
            let output = TensorData::new(Shape{batch:1,channels:hidden,height:1,width:seq});
            for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
            let n = 50;
            let start = Instant::now();
            for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
            let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
            let weight_mb = (4 * quarter * dim * 2) as f64 / 1e6;
            println!("4×quarter IP + dequant + concat: {ms:.2}ms ({weight_mb:.1}MB total weights)");
        }
        Err(e) => println!("4×quarter: FAIL: {e}"),
    }
    
    // Compare with single full inner_product
    {
        let mut g = Graph::new();
        let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
        let _y = g.inner_product(x, &vec![0.001; hidden * dim], dim, hidden);
        let exec = g.compile(NSQualityOfService::Default).unwrap();
        let input = TensorData::with_f32(&vec![0.01; dim*seq], Shape{batch:1,channels:dim,height:1,width:seq});
        let output = TensorData::new(Shape{batch:1,channels:hidden,height:1,width:seq});
        for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
        let n = 50;
        let start = Instant::now();
        for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("1×full IP (baseline):             {ms:.2}ms (47.2MB weights)");
    }
    
    // Single quarter (theoretical 4x bandwidth win)
    {
        let mut g = Graph::new();
        let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
        let _y = g.inner_product(x, &vec![0.001; quarter * dim], dim, quarter);
        let exec = g.compile(NSQualityOfService::Default).unwrap();
        let input = TensorData::with_f32(&vec![0.01; dim*seq], Shape{batch:1,channels:dim,height:1,width:seq});
        let output = TensorData::new(Shape{batch:1,channels:quarter,height:1,width:seq});
        for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
        let n = 50;
        let start = Instant::now();
        for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("1×quarter IP (1/4 bandwidth):     {ms:.2}ms (11.8MB weights)");
    }
    
    // Full layer with 4x-split matmuls (all 5 projections)
    println!("\n--- Full layer with split matmuls ---");
    {
        let mut g = Graph::new();
        let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
        let s = g.constant_with_scalar(1.0, Shape{batch:1,channels:1,height:1,width:1});
        
        // Q: dim->dim, split into 4×(dim->dim/4)
        let qdim = dim / 4;
        let q0 = g.multiplication(g.inner_product(x, &vec![0.001; qdim*dim], dim, qdim), s);
        let q1 = g.multiplication(g.inner_product(x, &vec![0.001; qdim*dim], dim, qdim), s);
        let q2 = g.multiplication(g.inner_product(x, &vec![0.001; qdim*dim], dim, qdim), s);
        let q3 = g.multiplication(g.inner_product(x, &vec![0.001; qdim*dim], dim, qdim), s);
        let q = g.concat(&[q0,q1,q2,q3], 1);
        
        // O: dim->dim
        let o0 = g.multiplication(g.inner_product(q, &vec![0.001; qdim*dim], dim, qdim), s);
        let o1 = g.multiplication(g.inner_product(q, &vec![0.001; qdim*dim], dim, qdim), s);
        let o2 = g.multiplication(g.inner_product(q, &vec![0.001; qdim*dim], dim, qdim), s);
        let o3 = g.multiplication(g.inner_product(q, &vec![0.001; qdim*dim], dim, qdim), s);
        let o = g.concat(&[o0,o1,o2,o3], 1);
        let h = g.addition(x, o);
        
        // Gate: dim->hidden, split into 4×(dim->hidden/4)
        let hq = hidden / 4;
        let g0 = g.multiplication(g.inner_product(h, &vec![0.001; hq*dim], dim, hq), s);
        let g1 = g.multiplication(g.inner_product(h, &vec![0.001; hq*dim], dim, hq), s);
        let g2 = g.multiplication(g.inner_product(h, &vec![0.001; hq*dim], dim, hq), s);
        let g3 = g.multiplication(g.inner_product(h, &vec![0.001; hq*dim], dim, hq), s);
        let gate = g.concat(&[g0,g1,g2,g3], 1);
        
        // Up: same
        let u0 = g.multiplication(g.inner_product(h, &vec![0.001; hq*dim], dim, hq), s);
        let u1 = g.multiplication(g.inner_product(h, &vec![0.001; hq*dim], dim, hq), s);
        let u2 = g.multiplication(g.inner_product(h, &vec![0.001; hq*dim], dim, hq), s);
        let u3 = g.multiplication(g.inner_product(h, &vec![0.001; hq*dim], dim, hq), s);
        let up = g.concat(&[u0,u1,u2,u3], 1);
        
        let gs2 = g.sigmoid(gate);
        let gl = g.multiplication(gate, gs2);
        let mix = g.multiplication(gl, up);
        
        // Down: hidden->dim, split on IC side (4×(hidden/4->dim))
        // Can't split IC with inner_product — IC must match input channels
        // Use full down projection instead
        let down = g.inner_product(mix, &vec![0.001; dim*hidden], hidden, dim);
        let _out = g.addition(h, down);
        
        match g.compile(NSQualityOfService::Default) {
            Ok(exec) => {
                let input = TensorData::with_f32(&vec![0.01; dim*seq], Shape{batch:1,channels:dim,height:1,width:seq});
                let output = TensorData::new(Shape{batch:1,channels:dim,height:1,width:seq});
                for _ in 0..3 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let n = 20;
                let start = Instant::now();
                for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
                let total_32 = 32.0 * ms;
                let toks = seq as f64 / total_32 * 1000.0;
                println!("Split layer (4x Q,O,gate,up):  {ms:.2}ms/layer → 32L={total_32:.0}ms → {toks:.0} tok/s");
            }
            Err(e) => println!("Split layer: FAIL: {e}"),
        }
    }
    
    // Original unsplit layer for comparison
    {
        let mut g = Graph::new();
        let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
        let q = g.inner_product(x, &vec![0.001; dim*dim], dim, dim);
        let o = g.inner_product(q, &vec![0.001; dim*dim], dim, dim);
        let h = g.addition(x, o);
        let gate = g.inner_product(h, &vec![0.001; hidden*dim], dim, hidden);
        let up = g.inner_product(h, &vec![0.001; hidden*dim], dim, hidden);
        let gs2 = g.sigmoid(gate);
        let gl = g.multiplication(gate, gs2);
        let mix = g.multiplication(gl, up);
        let down = g.inner_product(mix, &vec![0.001; dim*hidden], hidden, dim);
        let _out = g.addition(h, down);
        let exec = g.compile(NSQualityOfService::Default).unwrap();
        let input = TensorData::with_f32(&vec![0.01; dim*seq], Shape{batch:1,channels:dim,height:1,width:seq});
        let output = TensorData::new(Shape{batch:1,channels:dim,height:1,width:seq});
        for _ in 0..3 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
        let n = 20;
        let start = Instant::now();
        for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let total_32 = 32.0 * ms;
        let toks = seq as f64 / total_32 * 1000.0;
        println!("Unsplit layer (baseline):       {ms:.2}ms/layer → 32L={total_32:.0}ms → {toks:.0} tok/s");
    }
    
    println!("\nCoreML int4:                    12.65ms/layer → 32L=405ms → 1265 tok/s");
}
