/// Custom int8 dequantization on ANE.
/// 
/// Strategy: store weights as fp16 at HALF the output channels,
/// then use ANE elementwise ops to "expand" them back.
/// This simulates reading half the data (like int8 from int16 storage).
///
/// For true int4: pack 4 values into each fp16's 16 bits using 
/// bit manipulation via multiply + floor + subtract chains.
/// But ANE doesn't have bit ops — so we use a different trick:
///
/// GROUP DEQUANT: store (scale, zero_point) per group of 64 weights,
/// and store the weights as small fp16 integers (0-15 range).
/// Dequant: weight_fp16 * scale + zero_point
/// The weights take the same fp16 storage but the VALUES are in [0,15],
/// meaning the high bits are zero → the ANE reads less meaningful data
/// and the weight blob compresses better in the compiled model.
///
/// Actually the simplest approach: use TWO smaller inner_products 
/// and concatenate. Each reads half the weights → half bandwidth.

use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let dim = 2560;
    let hidden = 9216;
    let seq = 512;

    println!("=== CUSTOM QUANTIZATION ON ANE ===\n");

    // Baseline: full fp16 inner_product
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
        let weight_mb = (dim * hidden * 2) as f64 / 1e6;
        println!("fp16 full:    {ms:.2}ms  ({weight_mb:.1}MB weights)  {:.0} tok/s", seq as f64/ms*1000.0/32.0*32.0);
    }

    // Split approach: two inner_products at half output channels, then concat
    // Each reads hidden/2 * dim weights = half bandwidth
    {
        let half = hidden / 2;
        let mut g = Graph::new();
        let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
        let y1 = g.inner_product(x, &vec![0.001; half * dim], dim, half);
        let y2 = g.inner_product(x, &vec![0.001; half * dim], dim, half);
        let _y = g.concat(&[y1, y2], 1); // concat on channel axis
        let exec = g.compile(NSQualityOfService::Default).unwrap();
        let input = TensorData::with_f32(&vec![0.01; dim*seq], Shape{batch:1,channels:dim,height:1,width:seq});
        let output = TensorData::new(Shape{batch:1,channels:hidden,height:1,width:seq});
        for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
        let n = 50;
        let start = Instant::now();
        for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let weight_mb = (dim * hidden * 2) as f64 / 1e6; // same total weights
        println!("split 2×half: {ms:.2}ms  ({weight_mb:.1}MB weights)  — tests if two smaller reads are faster");
    }

    // Group dequant simulation: inner_product at reduced precision
    // Store weights as small integers × scale
    // weight_real = weight_int * scale + offset
    // The inner_product still uses fp16, but with constrained range
    {
        // Simulate: use inner_product with 1/4 the output channels,
        // then "expand" with a cheap matmul (expansion matrix)
        let reduced = hidden / 4;
        let mut g = Graph::new();
        let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
        // Compressed projection: dim -> hidden/4
        let compressed = g.inner_product(x, &vec![0.001; reduced * dim], dim, reduced);
        // Expansion: hidden/4 -> hidden via another inner_product
        // This is cheap because IC=hidden/4 which is small
        let expanded = g.inner_product(compressed, &vec![0.001; hidden * reduced], reduced, hidden);
        // Scale + offset (simulating dequant)
        let scale = g.constant_with_scalar(1.0, Shape{batch:1,channels:1,height:1,width:1});
        let _y = g.multiplication(expanded, scale);

        match g.compile(NSQualityOfService::Default) {
            Ok(exec) => {
                let input = TensorData::with_f32(&vec![0.01; dim*seq], Shape{batch:1,channels:dim,height:1,width:seq});
                let output = TensorData::new(Shape{batch:1,channels:hidden,height:1,width:seq});
                for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let n = 50;
                let start = Instant::now();
                for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
                let weight_mb = ((dim * reduced + reduced * hidden) * 2) as f64 / 1e6;
                println!("low-rank 4x:  {ms:.2}ms  ({weight_mb:.1}MB weights)  — 4x compression via factorization");
            }
            Err(e) => println!("low-rank 4x:  FAIL: {e}"),
        }
    }

    // Pure bandwidth test: inner_product with 1/4 output channels
    // This shows the theoretical speed if we could read 4x less data
    {
        let quarter = hidden / 4;
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
        let weight_mb = (dim * quarter * 2) as f64 / 1e6;
        println!("1/4 weights:  {ms:.2}ms  ({weight_mb:.1}MB weights)  — theoretical 4x bandwidth reduction");
    }

    println!("\nCoreMl int4:  12.65ms  (CoreML handles int4 dequant natively)");
    println!("Private fp16: 24.90ms  (our inner_product baseline)");
    println!("\nTarget: match CoreML's 12.65ms using private API + custom dequant");
}
