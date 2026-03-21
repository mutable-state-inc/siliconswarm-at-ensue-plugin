/// Single-dispatch full transformer layer on ANE.
/// All SDPA attention replaced with simple linear attention (no KV cache).
/// Goal: one ANE dispatch per token for maximum throughput.
///
/// For Qwen3.5-4B decode, each token needs:
///   - Input norm (RMSNorm approximated as mul+rsqrt)
///   - QKV projection (matmul)
///   - Attention (matmul)  
///   - Output projection (matmul)
///   - Residual add
///   - Post norm
///   - FFN gate+up (matmul)
///   - SiLU
///   - FFN down (matmul)
///   - Residual add
///
/// All using dynamic-weight matmul pattern.

use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn add_matmul(g: &mut Graph, input: ane::Tensor, ic: usize, oc: usize, seq: usize, w_offset: usize, packed_input: ane::Tensor) -> ane::Tensor {
    // Slice weights from packed input at w_offset
    let wts = g.slice(packed_input, [0, 0, 0, w_offset], [1, ic, 1, oc]);
    let acts_r = g.reshape(input, Shape { batch: 1, channels: 1, height: ic, width: seq });
    let acts_t = g.transpose(acts_r, [0, 1, 3, 2]);
    let wts_r = g.reshape(wts, Shape { batch: 1, channels: 1, height: ic, width: oc });
    let mm = g.matrix_multiplication(acts_t, wts_r, false, false);
    let mm_t = g.transpose(mm, [0, 1, 3, 2]);
    g.reshape(mm_t, Shape { batch: 1, channels: oc, height: 1, width: seq })
}

fn main() {
    // Use smaller dims first to test the pattern
    let dim = 256;
    let hidden = 1024;
    let seq = 64;

    // Pack ALL weights into one IOSurface spatial dimension:
    // gate_w: dim*hidden, up_w: dim*hidden, down_w: hidden*dim
    // Total weight slots: 2*dim*hidden + hidden*dim = 3*dim*hidden
    let gate_offset = seq;
    let up_offset = gate_offset + hidden;
    let down_offset = up_offset + hidden;
    let total_w_slots = hidden + hidden + dim; // for the 3 projections from dim side
    let total_sp = seq + total_w_slots;

    // But wait - matmul needs weights in [1, IC, 1, OC] format
    // For gate/up: IC=dim, OC=hidden → weights occupy `hidden` spatial slots per IC channel
    // For down: IC=hidden, OC=dim → needs different IC
    // Can't share IC dimension across matmuls with different IC...
    
    // Simpler approach: chain 2 matmul kernels (gate+up fused, then down)
    // Each as a separate slice from the packed input
    
    // Actually, let's just benchmark chaining elementwise ops to see
    // how many we can stack before hitting a limit
    
    let mut g = Graph::new();
    let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
    
    // Simulate FFN with elementwise ops only (no weights needed):
    // This tests how fast ANE runs a deep chain of ops
    let mut h = x;
    let n_ops = 100; // 100 elementwise ops
    for _ in 0..n_ops {
        let sig = g.sigmoid(h);
        h = g.multiplication(h, sig); // SiLU
    }
    
    println!("Compiling {n_ops}-op chain (dim={dim}, seq={seq})...");
    let start = Instant::now();
    let exec = g.compile(NSQualityOfService::Default).unwrap();
    println!("Compiled in {:?}", start.elapsed());
    
    let input = TensorData::with_f32(&vec![0.5f32; dim * seq],
        Shape { batch: 1, channels: dim, height: 1, width: seq });
    let output = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });
    
    for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
    
    let n = 500;
    let start = Instant::now();
    for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
    
    println!("{n_ops} SiLU ops in one dispatch: {ms:.3}ms");
    println!("Per op: {:.4}ms", ms / n_ops as f64);
    println!("Dispatch overhead: {:.3}ms (estimated from subtraction)", ms.max(0.1) - (n_ops as f64 * 0.001));
    
    // Now test: how many matmuls can we chain?
    // Using dynamic-weight pattern, each matmul needs its own slice of the input
    println!("\n--- Testing chained matmuls ---");
    
    let ic = 256;
    let oc = 256; // keep small to fit constant limit
    for n_matmuls in [1, 2, 3, 5, 8] {
        // Each matmul: slice acts + slice weights from different offsets
        let sp = seq + oc * n_matmuls; // space for N weight matrices
        let mut g = Graph::new();
        let packed = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: sp });
        
        let mut h = g.slice(packed, [0, 0, 0, 0], [1, ic, 1, seq]);
        for i in 0..n_matmuls {
            let w_off = seq + i * oc;
            let wts = g.slice(packed, [0, 0, 0, w_off], [1, ic, 1, oc]);
            let h_r = g.reshape(h, Shape { batch: 1, channels: 1, height: ic, width: seq });
            let h_t = g.transpose(h_r, [0, 1, 3, 2]);
            let w_r = g.reshape(wts, Shape { batch: 1, channels: 1, height: ic, width: oc });
            let mm = g.matrix_multiplication(h_t, w_r, false, false);
            let mm_t = g.transpose(mm, [0, 1, 3, 2]);
            h = g.reshape(mm_t, Shape { batch: 1, channels: oc, height: 1, width: seq });
        }
        
        match g.compile(NSQualityOfService::Default) {
            Ok(exec) => {
                let input = TensorData::with_f32(&vec![0.01f32; ic * sp],
                    Shape { batch: 1, channels: ic, height: 1, width: sp });
                let output = TensorData::new(Shape { batch: 1, channels: oc, height: 1, width: seq });
                for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let n = 200;
                let start = Instant::now();
                for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
                let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
                println!("{n_matmuls} chained {ic}x{oc} matmuls: {ms:.3}ms ({:.3}ms/matmul)", ms / n_matmuls as f64);
            }
            Err(e) => println!("{n_matmuls} chained matmuls: FAIL: {e}"),
        }
    }
}
