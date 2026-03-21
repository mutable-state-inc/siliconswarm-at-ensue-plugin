use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn bench(label: &str, exec: &ane::Executable, input: &TensorData, output: &TensorData, seq: usize) {
    for _ in 0..5 { exec.run_cached_direct(&[input], &[output]).unwrap(); }
    let n = 50;
    let start = Instant::now();
    for _ in 0..n { exec.run_cached_direct(&[input], &[output]).unwrap(); }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let total = 32.0 * ms;
    let toks = seq as f64 / total * 1000.0;
    println!("{label:30} {ms:.2}ms/layer  32L={total:.0}ms  {toks:.0} tok/s");
}

fn main() {
    let dim = 2560;
    let hidden = 9216;
    let seq = 512;
    
    println!("=== SPLIT vs FULL INNER_PRODUCT (seq={seq}) ===\n");
    
    // 1. Full single matmul
    let e1 = {
        let mut g = Graph::new();
        let x = g.placeholder(Shape{batch:1,channels:dim,height:1,width:seq});
        let _ = g.inner_product(x, &vec![0.001;hidden*dim], dim, hidden);
        g.compile(NSQualityOfService::Default).unwrap()
    };
    let i1 = TensorData::with_f32(&vec![0.01;dim*seq], Shape{batch:1,channels:dim,height:1,width:seq});
    let o1 = TensorData::new(Shape{batch:1,channels:hidden,height:1,width:seq});
    bench("1× full (dim→hidden)", &e1, &i1, &o1, seq);

    // 2. 4× quarter matmuls concatenated
    let e2 = {
        let q = hidden/4;
        let mut g = Graph::new();
        let x = g.placeholder(Shape{batch:1,channels:dim,height:1,width:seq});
        let a = g.inner_product(x, &vec![0.001;q*dim], dim, q);
        let b = g.inner_product(x, &vec![0.002;q*dim], dim, q);
        let c = g.inner_product(x, &vec![0.003;q*dim], dim, q);
        let d = g.inner_product(x, &vec![0.004;q*dim], dim, q);
        let _ = g.concat(&[a,b,c,d], 1);
        g.compile(NSQualityOfService::Default).unwrap()
    };
    let o2 = TensorData::new(Shape{batch:1,channels:hidden,height:1,width:seq});
    bench("4× quarter concat", &e2, &i1, &o2, seq);

    // 3. 2× half matmuls concatenated
    let e3 = {
        let h = hidden/2;
        let mut g = Graph::new();
        let x = g.placeholder(Shape{batch:1,channels:dim,height:1,width:seq});
        let a = g.inner_product(x, &vec![0.001;h*dim], dim, h);
        let b = g.inner_product(x, &vec![0.002;h*dim], dim, h);
        let _ = g.concat(&[a,b], 1);
        g.compile(NSQualityOfService::Default).unwrap()
    };
    let o3 = TensorData::new(Shape{batch:1,channels:hidden,height:1,width:seq});
    bench("2× half concat", &e3, &i1, &o3, seq);

    // 4. Single quarter (pure 4x bandwidth reduction)
    let e4 = {
        let q = hidden/4;
        let mut g = Graph::new();
        let x = g.placeholder(Shape{batch:1,channels:dim,height:1,width:seq});
        let _ = g.inner_product(x, &vec![0.001;q*dim], dim, q);
        g.compile(NSQualityOfService::Default).unwrap()
    };
    let o4 = TensorData::new(Shape{batch:1,channels:hidden/4,height:1,width:seq});
    bench("1× quarter (4x less BW)", &e4, &i1, &o4, seq);
    
    println!("\nCoreML int4 single layer: 12.65ms");
    println!("If 4×quarter is < 12.65ms, private API + split > CoreML!");
}
