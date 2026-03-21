use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let dim = 2560; let hidden = 9216; let seq = 64; let nl = 12;
    let mut g = Graph::new();
    let mut h = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
    for _ in 0..nl {
        let q = g.inner_product(h, &vec![0.001; dim*dim], dim, dim);
        let o = g.inner_product(q, &vec![0.001; dim*dim], dim, dim);
        h = g.addition(h, o);
        let gate = g.inner_product(h, &vec![0.001; hidden*dim], dim, hidden);
        let up = g.inner_product(h, &vec![0.001; hidden*dim], dim, hidden);
        let gs = g.sigmoid(gate); let gl = g.multiplication(gate, gs); let mix = g.multiplication(gl, up);
        let down = g.inner_product(mix, &vec![0.001; dim*hidden], hidden, dim);
        h = g.addition(h, down);
    }
    let exec = g.compile(NSQualityOfService::Default).unwrap();
    let input = TensorData::with_f32(&vec![0.01; dim*seq], Shape { batch: 1, channels: dim, height: 1, width: seq });
    let output = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });
    for _ in 0..3 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
    let n = 30;
    let start = Instant::now();
    for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let dispatches = (32.0 / nl as f64).ceil();
    let total = dispatches * ms;
    println!("=== {nl} LAYERS FUSED ON ANE (inner_product, dim={dim}) ===");
    println!("{nl}L per dispatch: {ms:.2}ms ({:.2}ms/layer)", ms / nl as f64);
    println!("32 layers ({dispatches:.0} dispatches): {total:.1}ms");
    println!("{:.1} tok/s", 1000.0 / total);
    println!("GPU: 95 tok/s");
}
