use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let dim = 2560;
    let seq = 64;
    
    // RMSNorm + SiLU + residual using working ops only
    // rsqrt(x) = power(x, -0.5) [rsqrt is broken on macOS 26]
    
    // Single layer elementwise
    let mut g = Graph::new();
    let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
    let neg_half = g.constant_with_scalar(-0.5, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let eps = g.constant_with_scalar(1e-6, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    
    // RMSNorm: x * (mean(x^2) + eps)^(-0.5)
    let sq = g.multiplication(x, x);
    let m = g.reduce_mean(sq, 1);
    let me = g.addition(m, eps);
    let rms = g.power(me, neg_half); // rsqrt via power
    let normed = g.multiplication(x, rms);
    // SiLU
    let sig = g.sigmoid(normed);
    let silu = g.multiplication(normed, sig);
    // Residual
    let _out = g.addition(x, silu);
    
    let exec = g.compile(NSQualityOfService::Default).unwrap();
    let input = TensorData::with_f32(&vec![0.5; dim * seq], Shape { batch: 1, channels: dim, height: 1, width: seq });
    let output = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });
    for _ in 0..10 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
    
    let n = 1000;
    let start = Instant::now();
    for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
    let ms1 = start.elapsed().as_secs_f64() * 1000.0 / n as f64;

    // 8 layers of elementwise fused
    let mut g2 = Graph::new();
    let mut h = g2.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
    let nh = g2.constant_with_scalar(-0.5, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let ep = g2.constant_with_scalar(1e-6, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    for _ in 0..8 {
        // Input norm
        let sq = g2.multiplication(h, h);
        let m = g2.reduce_mean(sq, 1);
        let me = g2.addition(m, ep);
        let rms = g2.power(me, nh);
        let n = g2.multiplication(h, rms);
        h = g2.addition(h, n); // residual 1
        // Post norm + SiLU
        let sq2 = g2.multiplication(h, h);
        let m2 = g2.reduce_mean(sq2, 1);
        let me2 = g2.addition(m2, ep);
        let rms2 = g2.power(me2, nh);
        let n2 = g2.multiplication(h, rms2);
        let sg = g2.sigmoid(n2);
        let sl = g2.multiplication(n2, sg);
        h = g2.addition(h, sl); // residual 2
    }
    let exec2 = g2.compile(NSQualityOfService::Default).unwrap();
    let input2 = TensorData::with_f32(&vec![0.5; dim * seq], Shape { batch: 1, channels: dim, height: 1, width: seq });
    let output2 = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });
    for _ in 0..10 { exec2.run_cached_direct(&[&input2], &[&output2]).unwrap(); }
    let start = Instant::now();
    for _ in 0..n { exec2.run_cached_direct(&[&input2], &[&output2]).unwrap(); }
    let ms8 = start.elapsed().as_secs_f64() * 1000.0 / n as f64;

    // 48 layers fused (6 dispatches of 8)
    let total_48 = 6.0 * ms8;

    println!("=== ANE ELEMENTWISE AT DIM=2560 (REAL OPS) ===");
    println!("1 layer (norm+silu+res): {ms1:.3}ms");
    println!("8 layers fused:         {ms8:.3}ms ({:.3}ms/layer)", ms8/8.0);
    println!("48 layers (6×8):        {total_48:.2}ms");
    println!("");
    println!("GPU total per token:    ~10.5ms (matmuls + elementwise)");
    println!("GPU elementwise only:   ~2-3ms (estimated 20% of total)");
    println!("ANE elementwise:        {total_48:.2}ms");
    println!("");
    if total_48 < 10.5 {
        println!("ANE can do ALL elementwise during GPU matmul time!");
        println!("Pipeline: GPU matmuls ({:.1}ms) || ANE elementwise ({:.1}ms) = {:.1}ms",
            10.5 - 2.5, total_48, (10.5 - 2.5f64).max(total_48));
        println!("Potential: {:.1} tok/s", 1000.0 / (10.5 - 2.5f64).max(total_48));
    } else {
        println!("ANE elementwise slower than GPU total — pipeline won't help");
    }
}
