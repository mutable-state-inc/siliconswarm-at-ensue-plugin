use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let seq = 64;
    // Test inner_product at various weight sizes to see bandwidth curve
    println!("{:>6} {:>6} {:>8} {:>8} {:>8}", "IC", "OC", "WeightMB", "ms", "GB/s");
    for (ic, oc) in [(256,256),(256,1024),(256,2048),(256,4096),(256,9216),
                      (512,9216),(1024,9216),(2560,2560),(2560,9216)] {
        let mut g = Graph::new();
        let x = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: seq });
        let _y = g.inner_product(x, &vec![0.001; oc * ic], ic, oc);
        match g.compile(NSQualityOfService::Default) {
            Ok(exec) => {
                let input = TensorData::with_f32(&vec![0.01; ic * seq], Shape { batch: 1, channels: ic, height: 1, width: seq });
                let output = TensorData::new(Shape { batch: 1, channels: oc, height: 1, width: seq });
                for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let n = 200;
                let start = Instant::now();
                for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
                let weight_mb = (ic * oc * 2) as f64 / 1e6;
                let bw = weight_mb / ms * 1000.0 / 1000.0;
                println!("{ic:>6} {oc:>6} {weight_mb:>7.1}MB {ms:>7.3}ms {bw:>7.1}");
            }
            Err(_) => println!("{ic:>6} {oc:>6}    FAIL"),
        }
    }
}
