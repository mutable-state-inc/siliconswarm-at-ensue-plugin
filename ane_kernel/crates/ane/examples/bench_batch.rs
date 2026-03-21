use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let dim = 2560;
    let hidden = 9216;
    
    println!("=== ANE THROUGHPUT vs BATCH SIZE ===\n");
    println!("{:>5} {:>8} {:>10} {:>10} {:>10}", "batch", "ms/eval", "tok/s", "GB/s", "TOPS");
    
    // seq dimension = batch size (number of tokens processed simultaneously)
    for batch in [1, 2, 4, 8, 16, 32, 64] {
        let mut g = Graph::new();
        let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: batch });
        let _y = g.inner_product(x, &vec![0.001; hidden * dim], dim, hidden);
        
        match g.compile(NSQualityOfService::Default) {
            Ok(exec) => {
                let input = TensorData::with_f32(&vec![0.01; dim * batch],
                    Shape { batch: 1, channels: dim, height: 1, width: batch });
                let output = TensorData::new(
                    Shape { batch: 1, channels: hidden, height: 1, width: batch });
                for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let n = 100;
                let start = Instant::now();
                for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
                let toks = batch as f64 / ms * 1000.0;
                let weight_bytes = dim * hidden * 2; // fp16
                let bw = weight_bytes as f64 / ms / 1e6; // GB/s
                let flops = 2.0 * dim as f64 * hidden as f64 * batch as f64; // multiply-adds
                let tops = flops / ms / 1e9; // TOPS
                println!("{batch:>5} {ms:>7.3}ms {toks:>9.0} {bw:>9.1} {tops:>9.2}");
            }
            Err(e) => println!("{batch:>5} FAIL: {e}"),
        }
    }
    
    println!("\nANE peak: 11 TOPS (int8) ≈ 5.5 TFLOPS (fp16)");
    println!("GPU single token: 95 tok/s");
    println!("\nAt batch=64: ANE processes 64 tokens per dispatch.");
    println!("For serving multiple users simultaneously,");
    println!("ANE throughput in tok/s is what matters, not latency.");
}
