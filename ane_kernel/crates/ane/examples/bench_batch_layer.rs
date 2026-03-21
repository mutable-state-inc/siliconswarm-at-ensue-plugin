use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let dim = 2560;
    let hidden = 9216;
    let batch = 64; // 64 tokens simultaneously

    // Full layer at batch=64
    let mut g = Graph::new();
    let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: batch });
    let q = g.inner_product(x, &vec![0.001; dim*dim], dim, dim);
    let o = g.inner_product(q, &vec![0.001; dim*dim], dim, dim);
    let h = g.addition(x, o);
    let gate = g.inner_product(h, &vec![0.001; hidden*dim], dim, hidden);
    let up = g.inner_product(h, &vec![0.001; hidden*dim], dim, hidden);
    let gs = g.sigmoid(gate);
    let gl = g.multiplication(gate, gs);
    let mix = g.multiplication(gl, up);
    let down = g.inner_product(mix, &vec![0.001; dim*hidden], hidden, dim);
    let _out = g.addition(h, down);

    println!("Compiling full layer at batch={batch}...");
    let exec = g.compile(NSQualityOfService::Default).unwrap();

    let input = TensorData::with_f32(&vec![0.01; dim*batch],
        Shape { batch: 1, channels: dim, height: 1, width: batch });
    let output = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: batch });
    for _ in 0..3 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }

    let n = 50;
    let start = Instant::now();
    for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let toks_per_sec = batch as f64 / ms * 1000.0;

    println!("\n=== FULL QWEN LAYER, BATCH={batch} ===");
    println!("Per dispatch:    {ms:.2}ms ({batch} tokens)");
    println!("Throughput:      {toks_per_sec:.0} tok/s per layer");
    println!("32 layers:       {:.0} tok/s", toks_per_sec / 32.0 * batch as f64 / (32.0 * ms / 1000.0) / (batch as f64));

    // Actually: 32 layers sequential, each processes batch=64
    let total_32 = 32.0 * ms;
    let total_toks = batch as f64 / (total_32 / 1000.0);
    println!("32 layers seq:   {total_32:.1}ms for {batch} tokens = {total_toks:.0} tok/s");
    println!("");
    println!("GPU batch=1:     95 tok/s (single user)");
    println!("ANE batch=64:    {total_toks:.0} tok/s (64 users simultaneously)");
    println!("ANE per-user:    {:.1} tok/s", total_toks / batch as f64);
}
