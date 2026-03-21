use ane::{Graph, Shape};
use objc2_foundation::NSQualityOfService;

fn main() {
    let dim = 256;
    // Binary search for max spatial width
    let mut lo = 64usize;
    let mut hi = 100000usize;
    
    while lo < hi - 1 {
        let mid = (lo + hi) / 2;
        let mut g = Graph::new();
        let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: mid });
        let _y = g.addition(x, x);
        match g.compile(NSQualityOfService::Default) {
            Ok(_) => { lo = mid; }
            Err(_) => { hi = mid; }
        }
    }
    println!("Max spatial width for dim={dim}: {lo}");
    
    // Also test with matmul
    let mut lo2 = 64usize;
    let mut hi2 = lo + 1;
    while lo2 < hi2 - 1 {
        let mid = (lo2 + hi2) / 2;
        let oc = 256;
        let seq = mid - oc;
        if seq < 64 { lo2 = mid; continue; }
        let mut g = Graph::new();
        let input = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: mid });
        let acts = g.slice(input, [0, 0, 0, 0], [1, dim, 1, seq]);
        let wts = g.slice(input, [0, 0, 0, seq], [1, dim, 1, oc]);
        let a_r = g.reshape(acts, Shape { batch: 1, channels: 1, height: dim, width: seq });
        let a_t = g.transpose(a_r, [0, 1, 3, 2]);
        let w_r = g.reshape(wts, Shape { batch: 1, channels: 1, height: dim, width: oc });
        let _mm = g.matrix_multiplication(a_t, w_r, false, false);
        match g.compile(NSQualityOfService::Default) {
            Ok(_) => { lo2 = mid; }
            Err(_) => { hi2 = mid; }
        }
    }
    println!("Max spatial width for matmul dim={dim}: {lo2}");
    
    let weights_per_layer = 4 * dim + 3 * dim; // 7 * dim for same-dim model
    let usable = lo2 - 64; // subtract seq
    let max_layers = usable / weights_per_layer;
    println!("Max layers in one dispatch (dim={dim}): {max_layers}");
}
