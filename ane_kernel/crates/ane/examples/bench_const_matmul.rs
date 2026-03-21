/// Benchmark matmul with CONSTANT weights (baked into compiled model).
/// The ANE may cache constant weights in SRAM, avoiding DRAM reads entirely.
use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    // Small matmul that might fit in ANE SRAM
    for (ic, oc) in [(256, 512), (512, 1024), (768, 2048), (2560, 9216)] {
        let seq = 64;
        let mut g = Graph::new();
        let x = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: seq });
        
        // Constant weight — baked into compiled model, potentially cached in SRAM
        let w = g.constant(&vec![0.001f32; oc * ic],
            Shape { channels: oc, height: ic, width: 1, batch: 1 });
        // Use convolution_2d_1x1 for constant-weight matmul
        // (This FAILED before, but let's verify)
        let _y = g.convolution_2d_1x1(x, w, None);
        
        match g.compile(NSQualityOfService::Default) {
            Ok(exec) => {
                let input = TensorData::with_f32(&vec![0.01f32; ic * seq],
                    Shape { batch: 1, channels: ic, height: 1, width: seq });
                let output = TensorData::new(Shape { batch: 1, channels: oc, height: 1, width: seq });
                for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let n = 200;
                let start = Instant::now();
                for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
                println!("const {ic}x{oc}: {ms:.3}ms");
            }
            Err(e) => {
                // Fallback: try matrix_multiplication with constant weights
                let mut g2 = Graph::new();
                let x2 = g2.placeholder(Shape { batch: 1, channels: 1, height: 1, width: ic * seq });
                let w2 = g2.constant(&vec![0.001f32; oc * ic],
                    Shape { batch: 1, channels: 1, height: ic, width: oc });
                let x2_r = g2.reshape(x2, Shape { batch: 1, channels: 1, height: seq, width: ic });
                let _mm = g2.matrix_multiplication(x2_r, w2, false, false);
                
                match g2.compile(NSQualityOfService::Default) {
                    Ok(exec) => {
                        let input = TensorData::with_f32(&vec![0.01f32; ic * seq],
                            Shape { batch: 1, channels: 1, height: 1, width: ic * seq });
                        let output = TensorData::new(Shape { batch: 1, channels: 1, height: seq, width: oc });
                        for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                        let n = 200;
                        let start = Instant::now();
                        for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                        let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
                        println!("const_mm {ic}x{oc}: {ms:.3}ms");
                    }
                    Err(e2) => println!("const {ic}x{oc}: conv FAILED ({e}), mm FAILED ({e2})"),
                }
            }
        }
    }
}
