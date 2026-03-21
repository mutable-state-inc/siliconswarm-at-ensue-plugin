use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn build_matmul(ic: usize, oc: usize, seq: usize) -> Graph {
    let sp = seq + oc;
    let mut g = Graph::new();
    let input = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: sp });
    let acts = g.slice(input, [0, 0, 0, 0], [1, ic, 1, seq]);
    let wts = g.slice(input, [0, 0, 0, seq], [1, ic, 1, oc]);
    let acts_r = g.reshape(acts, Shape { batch: 1, channels: 1, height: ic, width: seq });
    let acts_t = g.transpose(acts_r, [0, 1, 3, 2]);
    let wts_r = g.reshape(wts, Shape { batch: 1, channels: 1, height: ic, width: oc });
    let mm = g.matrix_multiplication(acts_t, wts_r, false, false);
    let mm_t = g.transpose(mm, [0, 1, 3, 2]);
    let _out = g.reshape(mm_t, Shape { batch: 1, channels: oc, height: 1, width: seq });
    g
}

fn main() {
    let dim = 2560;
    let hidden = 9216;
    // seq must be >= 64 (ANE min spatial width)
    // But for decode we only need 1 token — pad to 64
    let seq = 64;

    let exec = build_matmul(dim, hidden, seq).compile(NSQualityOfService::Default).unwrap();

    let sp = seq + hidden;
    let input = TensorData::with_f32(&vec![0.01f32; dim * sp],
        Shape { batch: 1, channels: dim, height: 1, width: sp });
    let output = TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: seq });

    for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }

    let n = 200;
    let start = Instant::now();
    for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;

    // Full FFN = 3 matmuls
    println!("Single matmul {dim}x{hidden}: {ms:.3}ms");
    println!("Full FFN (3 matmuls): {:.1}ms", 3.0 * ms);
    println!("48 layers: {:.0}ms", 48.0 * 3.0 * ms);
    println!("GPU: ~10.5ms total");
    println!("");
    println!("Key insight: each matmul reads {:.1}MB of weights", 
        (dim * hidden * 2) as f64 / 1_000_000.0);
    println!("At 40GB/s ANE BW: {:.1}ms minimum", 
        (dim * hidden * 2) as f64 / 40_000_000_000.0 * 1000.0);
}
