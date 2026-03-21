use ane::{Graph, Shape};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let dim = 2560;
    let hidden = 9216;
    let seq = 64;
    
    for nl in [4, 6, 8, 10, 12, 16, 20, 24, 28, 32] {
        let mut g = Graph::new();
        let mut h = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
        for _ in 0..nl {
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
            Ok(_) => println!("{nl:>2} layers: COMPILE OK ({:?})", start.elapsed()),
            Err(e) => { println!("{nl:>2} layers: FAIL ({e})"); break; }
        }
    }
}
