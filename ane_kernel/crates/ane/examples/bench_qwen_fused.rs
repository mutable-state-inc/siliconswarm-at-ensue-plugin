use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let dim = 2560;
    let hidden = 9216;
    let seq = 64;
    
    for num_layers in [1, 2, 3, 4] {
        let mut g = Graph::new();
        let mut h = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
        
        for _ in 0..num_layers {
            let q = g.inner_product(h, &vec![0.001; dim * dim], dim, dim);
            let o = g.inner_product(q, &vec![0.001; dim * dim], dim, dim);
            h = g.addition(h, o);
            let gate = g.inner_product(h, &vec![0.001; hidden * dim], dim, hidden);
            let up = g.inner_product(h, &vec![0.001; hidden * dim], dim, hidden);
            let gs = g.sigmoid(gate);
            let gl = g.multiplication(gate, gs);
            let mix = g.multiplication(gl, up);
            let down = g.inner_product(mix, &vec![0.001; dim * hidden], hidden, dim);
            h = g.addition(h, down);
        }
        
        let start = Instant::now();
        match g.compile(NSQualityOfService::Default) {
            Ok(exec) => {
                let cdur = start.elapsed();
                let input = TensorData::with_f32(&vec![0.01; dim * seq], Shape { batch: 1, channels: dim, height: 1, width: seq });
                let output = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });
                for _ in 0..3 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let n = 50;
                let start = Instant::now();
                for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
                let dispatches = (32.0 / num_layers as f64).ceil();
                let total = dispatches * ms;
                println!("{num_layers}L fused: {ms:.2}ms ({:.2}ms/L), compile {cdur:.0?}, 32L={total:.1}ms ({:.0} tok/s)", ms/num_layers as f64, 1000.0/total);
            }
            Err(e) => println!("{num_layers}L fused: FAIL: {e}"),
        }
    }
}
