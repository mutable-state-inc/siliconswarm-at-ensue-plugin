use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn bench(label: &str, exec: &ane::Executable, input: &TensorData, output: &TensorData, seq: usize, layers: usize) {
    for _ in 0..5 { exec.run_cached_direct(&[input], &[output]).unwrap(); }
    let n = 40;
    let start = Instant::now();
    for _ in 0..n { exec.run_cached_direct(&[input], &[output]).unwrap(); }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let total = layers as f64 * ms;
    let toks = seq as f64 / total * 1000.0;
    println!("{label:45} {ms:.2}ms/layer  {total:.0}ms total  {toks:.0} tok/s");
}

fn main() {
    let dim = 1536;
    let hidden = 8960;
    let layers = 28;
    let seq = 512;

    println!("=== KERNEL OPTIMIZATION: Qwen3.5-1.7B ===\n");

    // 1. BASELINE: current kernel (5 separate inner_products)
    let e1 = {
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
    let i1 = TensorData::with_f32(&vec![0.01;dim*seq], Shape{batch:1,channels:dim,height:1,width:seq});
    let o1 = TensorData::new(Shape{batch:1,channels:dim,height:1,width:seq});
    bench("1. Baseline (5 IPs)", &e1, &i1, &o1, seq, layers);

    // 2. FUSED GATE+UP: one wide inner_product dim→2*hidden, then slice
    let e2 = {
        let fused_out = hidden * 2;
        let mut g = Graph::new();
        let x = g.placeholder(Shape{batch:1,channels:dim,height:1,width:seq});
        let q = g.inner_product(x, &vec![0.001;dim*dim], dim, dim);
        let o = g.inner_product(q, &vec![0.001;dim*dim], dim, dim);
        let h = g.addition(x, o);
        // Fused gate+up: one matmul instead of two
        let fused = g.inner_product(h, &vec![0.001;fused_out*dim], dim, fused_out);
        let gate = g.slice(fused, [0,0,0,0], [1,hidden,1,seq]);
        let up = g.slice(fused, [0,hidden,0,0], [1,hidden,1,seq]);
        let gs = g.sigmoid(gate);
        let gl = g.multiplication(gate, gs);
        let mix = g.multiplication(gl, up);
        let down = g.inner_product(mix, &vec![0.001;dim*hidden], hidden, dim);
        let _ = g.addition(h, down);
        g.compile(NSQualityOfService::Default).unwrap()
    };
    let o2 = TensorData::new(Shape{batch:1,channels:dim,height:1,width:seq});
    bench("2. Fused gate+up (4 IPs)", &e2, &i1, &o2, seq, layers);

    // 3. SKIP O PROJECTION: Q directly to residual (simplified attention)
    let e3 = {
        let mut g = Graph::new();
        let x = g.placeholder(Shape{batch:1,channels:dim,height:1,width:seq});
        let q = g.inner_product(x, &vec![0.001;dim*dim], dim, dim);
        // Skip O — straight to residual
        let h = g.addition(x, q);
        let gate = g.inner_product(h, &vec![0.001;hidden*dim], dim, hidden);
        let up = g.inner_product(h, &vec![0.001;hidden*dim], dim, hidden);
        let gs = g.sigmoid(gate);
        let gl = g.multiplication(gate, gs);
        let mix = g.multiplication(gl, up);
        let down = g.inner_product(mix, &vec![0.001;dim*hidden], hidden, dim);
        let _ = g.addition(h, down);
        g.compile(NSQualityOfService::Default).unwrap()
    };
    bench("3. Skip O proj (4 IPs)", &e3, &i1, &o1, seq, layers);

    // 4. FUSED GATE+UP + SKIP O (3 IPs total)
    let e4 = {
        let fused_out = hidden * 2;
        let mut g = Graph::new();
        let x = g.placeholder(Shape{batch:1,channels:dim,height:1,width:seq});
        let q = g.inner_product(x, &vec![0.001;dim*dim], dim, dim);
        let h = g.addition(x, q);
        let fused = g.inner_product(h, &vec![0.001;fused_out*dim], dim, fused_out);
        let gate = g.slice(fused, [0,0,0,0], [1,hidden,1,seq]);
        let up = g.slice(fused, [0,hidden,0,0], [1,hidden,1,seq]);
        let gs = g.sigmoid(gate);
        let gl = g.multiplication(gate, gs);
        let mix = g.multiplication(gl, up);
        let down = g.inner_product(mix, &vec![0.001;dim*hidden], hidden, dim);
        let _ = g.addition(h, down);
        g.compile(NSQualityOfService::Default).unwrap()
    };
    bench("4. Fused gate+up + skip O (3 IPs)", &e4, &i1, &o1, seq, layers);

    // 5. DIFFERENT SEQ LENGTHS
    for test_seq in [256, 384, 512, 768, 1024] {
        let e = {
            let mut g = Graph::new();
            let x = g.placeholder(Shape{batch:1,channels:dim,height:1,width:test_seq});
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
            match g.compile(NSQualityOfService::Default) {
                Ok(e) => e,
                Err(_) => { println!("5. seq={test_seq:4}                                        COMPILE FAIL"); continue; }
            }
        };
        let ti = TensorData::with_f32(&vec![0.01;dim*test_seq], Shape{batch:1,channels:dim,height:1,width:test_seq});
        let to2 = TensorData::new(Shape{batch:1,channels:dim,height:1,width:test_seq});
        // For seq comparison, report tok/s based on test_seq tokens
        for _ in 0..5 { e.run_cached_direct(&[&ti], &[&to2]).unwrap(); }
        let n = 30;
        let start = Instant::now();
        for _ in 0..n { e.run_cached_direct(&[&ti], &[&to2]).unwrap(); }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let total = layers as f64 * ms;
        let toks = test_seq as f64 / total * 1000.0;
        let throughput = (dim * test_seq * 2) as f64 / ms / 1e6; // activation GB/s
        println!("5. seq={test_seq:4}                                       {ms:.2}ms/layer  {total:.0}ms total  {toks:.0} tok/s  ({throughput:.1} GB/s act)");
    }

    // 6. TWO LAYERS FUSED
    let e6 = {
        let mut g = Graph::new();
        let mut h = g.placeholder(Shape{batch:1,channels:dim,height:1,width:seq});
        for _ in 0..2 {
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
        g.compile(NSQualityOfService::Default).unwrap()
    };
    let o6 = TensorData::new(Shape{batch:1,channels:dim,height:1,width:seq});
    for _ in 0..5 { e6.run_cached_direct(&[&i1], &[&o6]).unwrap(); }
    let n = 30;
    let start = Instant::now();
    for _ in 0..n { e6.run_cached_direct(&[&i1], &[&o6]).unwrap(); }
    let ms6 = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let dispatches6 = (layers as f64 / 2.0).ceil();
    let total6 = dispatches6 * ms6;
    let toks6 = seq as f64 / total6 * 1000.0;
    println!("6. 2L fused (×{dispatches6:.0} dispatches)                      {ms6:.2}ms/2L    {total6:.0}ms total  {toks6:.0} tok/s");
}
