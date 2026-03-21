/// Find where ANE beats GPU for token generation.
/// GPU scales at ~0.22ms/layer for Qwen3.5-4B.
/// ANE's per-tile time is ~0.53ms for 256xN matmul.
use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let seq = 64;
    let tile_ic = 256;
    
    println!("ANE matmul latency vs output dimension:\n");
    println!("{:>6} {:>8} {:>8} {:>10}", "OC", "ms/tile", "weight", "GB/s");
    
    for oc in [64, 128, 256, 512, 1024, 2048, 4096] {
        let mut g = Graph::new();
        let x = g.placeholder(Shape { batch: 1, channels: 1, height: 1, width: tile_ic * seq });
        let w = g.constant(&vec![0.001f32; oc * tile_ic],
            Shape { batch: 1, channels: 1, height: tile_ic, width: oc });
        let x_r = g.reshape(x, Shape { batch: 1, channels: 1, height: seq, width: tile_ic });
        let _mm = g.matrix_multiplication(x_r, w, false, false);
        
        match g.compile(NSQualityOfService::Default) {
            Ok(exec) => {
                let input = TensorData::with_f32(&vec![0.01f32; tile_ic * seq],
                    Shape { batch: 1, channels: 1, height: 1, width: tile_ic * seq });
                let output = TensorData::new(Shape { batch: 1, channels: 1, height: seq, width: oc });
                for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let n = 200;
                let start = Instant::now();
                for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
                let weight_kb = tile_ic * oc * 2 / 1024;
                let bw = (tile_ic * oc * 2) as f64 / ms / 1_000_000.0;
                println!("{oc:6} {ms:8.3} {weight_kb:6}KB {bw:8.1}");
            }
            Err(_) => println!("{oc:6}    FAIL"),
        }
    }
    
    println!("\nANE dispatch overhead: ~0.1ms (XPC round-trip)");
    println!("Minimum practical latency: ~0.2ms per kernel");
    println!("GPU per-layer: ~0.22ms (Qwen3.5-4B with fused 4-bit)");
}
