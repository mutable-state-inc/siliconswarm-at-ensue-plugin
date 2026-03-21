use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    for (ic, oc) in [(256, 512), (512, 1024), (2560, 9216)] {
        let seq = 64;
        let mut g = Graph::new();
        let x = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: seq });
        let _y = g.inner_product(x, &vec![0.001f32; oc * ic], ic, oc);
        
        let start = Instant::now();
        match g.compile(NSQualityOfService::Default) {
            Ok(exec) => {
                let dur = start.elapsed();
                let input = TensorData::with_f32(&vec![0.01; ic * seq], Shape { batch: 1, channels: ic, height: 1, width: seq });
                let output = TensorData::new(Shape { batch: 1, channels: oc, height: 1, width: seq });
                for _ in 0..3 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let n = 100;
                let start = Instant::now();
                for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
                println!("inner_product {ic}x{oc}: COMPILE OK ({dur:?}), {ms:.3}ms/eval");
            }
            Err(e) => println!("inner_product {ic}x{oc}: FAILED: {e}"),
        }
    }
}
