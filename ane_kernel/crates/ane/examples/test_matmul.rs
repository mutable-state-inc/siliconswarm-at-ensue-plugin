use ane::{Graph, Shape};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    // Test matmul at various sizes
    for (ic, oc, seq) in [(64, 128, 64), (256, 512, 64), (768, 2048, 64), (2560, 9216, 64)] {
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

        let start = Instant::now();
        match g.compile(NSQualityOfService::Default) {
            Ok(exec) => {
                let compile_dur = start.elapsed();
                
                // Benchmark eval
                let input_data = ane::TensorData::with_f32(&vec![0.01f32; ic * sp], 
                    Shape { batch: 1, channels: ic, height: 1, width: sp });
                let output_data = ane::TensorData::new(
                    Shape { batch: 1, channels: oc, height: 1, width: seq });
                
                // Warmup
                for _ in 0..3 { exec.run_cached(&[&input_data], &[&output_data]).unwrap(); }
                
                let n = 100;
                let start = Instant::now();
                for _ in 0..n { exec.run_cached(&[&input_data], &[&output_data]).unwrap(); }
                let dur = start.elapsed();
                let per_ms = dur.as_secs_f64() * 1000.0 / n as f64;
                
                println!("matmul {ic}x{oc} seq={seq}: OK (compile {:.0}ms, {:.3}ms/eval)", 
                    compile_dur.as_secs_f64()*1000.0, per_ms);
                drop(exec);
            }
            Err(e) => println!("matmul {ic}x{oc} seq={seq}: FAILED: {e}"),
        }
    }
}
