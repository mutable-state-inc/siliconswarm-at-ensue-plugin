use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let dim = 2560;
    let hidden = 9216;
    
    println!("=== ANE PREFILL BENCHMARK ===\n");
    println!("{:>5} {:>8} {:>10} {:>10}", "seqlen", "ms", "tok/s", "vs GPU");

    // GPU prefill baseline (from bench_ane_test.go):
    // BenchmarkInference GPU Prefill: ~94ms for 23 tokens = ~245 prompt_tok/s
    // For longer prompts, GPU is memory-bandwidth bound
    let gpu_prefill_toks = 245.0; // approximate from benchmarks

    for seq_len in [64, 128, 256, 512] {
        let mut g = Graph::new();
        let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq_len });
        // Single layer
        let q = g.inner_product(x, &vec![0.001; dim*dim], dim, dim);
        let o = g.inner_product(q, &vec![0.001; dim*dim], dim, dim);
        let h = g.addition(x, o);
        let gate = g.inner_product(h, &vec![0.001; hidden*dim], dim, hidden);
        let up = g.inner_product(h, &vec![0.001; hidden*dim], dim, hidden);
        let gs = g.sigmoid(gate);
        let gl = g.multiplication(gate, gs);
        let mix = g.multiplication(gl, up);
        let down = g.inner_product(mix, &vec![0.001; dim*hidden], hidden, dim);
        let _out = g.addition(h, down);

        match g.compile(NSQualityOfService::Default) {
            Ok(exec) => {
                let input = TensorData::with_f32(&vec![0.01; dim*seq_len],
                    Shape { batch: 1, channels: dim, height: 1, width: seq_len });
                let output = TensorData::new(
                    Shape { batch: 1, channels: dim, height: 1, width: seq_len });
                for _ in 0..3 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let n = 30;
                let start = Instant::now();
                for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let layer_ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
                let total_ms = 32.0 * layer_ms; // 32 layers
                let toks = seq_len as f64 / total_ms * 1000.0;
                println!("{seq_len:>5} {total_ms:>7.1}ms {toks:>9.0} {:.1}x", toks/gpu_prefill_toks);
            }
            Err(e) => println!("{seq_len:>5} FAIL: {e}"),
        }
    }
    
    println!("\nGPU prefill: ~{gpu_prefill_toks:.0} prompt_tok/s");
    println!("If ANE prefill > GPU prefill at some sequence length,");
    println!("we can offload prefill to ANE and free GPU for decode.");
}
