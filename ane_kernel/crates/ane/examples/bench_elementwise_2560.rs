use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let dim = 2560;
    let seq = 64;
    
    // Build: RMSNorm + SiLU + residual add (all elementwise, dim=2560)
    let mut g = Graph::new();
    let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
    
    // RMSNorm: x * rsqrt(mean(x^2) + eps)
    let x_sq = g.multiplication(x, x);
    let mean = g.reduce_mean(x_sq, 1); // reduce over channels
    let eps = g.constant_with_scalar(1e-6, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let mean_eps = g.addition(mean, eps);
    let rsqrt = g.reciprocal_square_root(mean_eps);
    let normed = g.multiplication(x, rsqrt);
    
    // SiLU: x * sigmoid(x)
    let sig = g.sigmoid(normed);
    let silu = g.multiplication(normed, sig);
    
    // Residual add
    let _out = g.addition(x, silu);
    
    let exec = g.compile(NSQualityOfService::Default).unwrap();
    let input = TensorData::with_f32(&vec![0.5; dim * seq], Shape { batch: 1, channels: dim, height: 1, width: seq });
    let output = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });
    
    for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
    
    let n = 1000;
    let start = Instant::now();
    for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
    
    // Build a bigger graph: 2x norm + 1x silu + 2x residual (one full layer's elementwise)
    let mut g2 = Graph::new();
    let x2 = g2.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
    
    // Input norm
    let sq1 = g2.multiplication(x2, x2);
    let m1 = g2.reduce_mean(sq1, 1);
    let e1 = g2.constant_with_scalar(1e-6, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let me1 = g2.addition(m1, e1);
    let rs1 = g2.reciprocal_square_root(me1);
    let n1 = g2.multiplication(x2, rs1);
    // Residual 1
    let h1 = g2.addition(x2, n1);
    // Post norm
    let sq2 = g2.multiplication(h1, h1);
    let m2 = g2.reduce_mean(sq2, 1);
    let me2 = g2.addition(m2, e1);
    let rs2 = g2.reciprocal_square_root(me2);
    let n2 = g2.multiplication(h1, rs2);
    // SiLU
    let sg = g2.sigmoid(n2);
    let sl = g2.multiplication(n2, sg);
    // Residual 2
    let _o2 = g2.addition(h1, sl);
    
    let exec2 = g2.compile(NSQualityOfService::Default).unwrap();
    let input2 = TensorData::with_f32(&vec![0.5; dim * seq], Shape { batch: 1, channels: dim, height: 1, width: seq });
    let output2 = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });
    for _ in 0..5 { exec2.run_cached_direct(&[&input2], &[&output2]).unwrap(); }
    
    let start = Instant::now();
    for _ in 0..n { exec2.run_cached_direct(&[&input2], &[&output2]).unwrap(); }
    let ms2 = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
    
    // Chain 8 layers of elementwise in one graph
    let mut g3 = Graph::new();
    let mut h = g3.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
    let eps3 = g3.constant_with_scalar(1e-6, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    for _ in 0..8 {
        let sq = g3.multiplication(h, h);
        let m = g3.reduce_mean(sq, 1);
        let me = g3.addition(m, eps3);
        let rs = g3.reciprocal_square_root(me);
        let n = g3.multiplication(h, rs);
        h = g3.addition(h, n);
        let sq2 = g3.multiplication(h, h);
        let m2 = g3.reduce_mean(sq2, 1);
        let me2 = g3.addition(m2, eps3);
        let rs2 = g3.reciprocal_square_root(me2);
        let n2 = g3.multiplication(h, rs2);
        let sg = g3.sigmoid(n2);
        let sl = g3.multiplication(n2, sg);
        h = g3.addition(h, sl);
    }
    let exec3 = g3.compile(NSQualityOfService::Default).unwrap();
    let input3 = TensorData::with_f32(&vec![0.5; dim * seq], Shape { batch: 1, channels: dim, height: 1, width: seq });
    let output3 = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });
    for _ in 0..5 { exec3.run_cached_direct(&[&input3], &[&output3]).unwrap(); }
    let start = Instant::now();
    for _ in 0..n { exec3.run_cached_direct(&[&input3], &[&output3]).unwrap(); }
    let ms3 = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
    
    println!("=== ANE ELEMENTWISE AT DIM=2560 ===");
    println!("Simple (norm+silu+res):     {ms:.3}ms");
    println!("Full layer elementwise:     {ms2:.3}ms");
    println!("8 layers fused elementwise: {ms3:.3}ms ({:.3}ms/layer)", ms3/8.0);
    println!("");
    println!("48 layers (6 dispatches of 8): {:.2}ms", 6.0 * ms3);
    println!("GPU elementwise time:          ~15.6ms (from profiling)");
    println!("Can ANE do it faster?          {}", if 6.0*ms3 < 15.6 {"YES"} else {"NO"});
}
