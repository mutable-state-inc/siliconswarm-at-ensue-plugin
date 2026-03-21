use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let dim = 256;
    let hidden = 256;
    let seq = 64;
    let num_layers = 9;

    let weights_per_layer = 4 * dim + 3 * hidden;
    let sp = seq + num_layers * weights_per_layer;

    let mut g = Graph::new();
    let packed = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: sp });
    let mut h = g.slice(packed, [0, 0, 0, 0], [1, dim, 1, seq]);
    let mut w_offset = seq;

    for _ in 0..num_layers {
        // Q,K,V,O projections + attention
        let h_r = g.reshape(h, Shape { batch: 1, channels: 1, height: dim, width: seq });
        let h_t = g.transpose(h_r, [0, 1, 3, 2]);
        
        for _ in 0..3 { // Q, K, V
            let wts = g.slice(packed, [0, 0, 0, w_offset], [1, dim, 1, dim]);
            w_offset += dim;
            let _ = g.reshape(wts, Shape { batch: 1, channels: 1, height: dim, width: dim });
        }
        // Simplified: just do Q projection and skip attention for speed test
        let q_wts = g.slice(packed, [0, 0, 0, w_offset - 3*dim], [1, dim, 1, dim]);
        let q_r = g.reshape(q_wts, Shape { batch: 1, channels: 1, height: dim, width: dim });
        let q = g.matrix_multiplication(h_t, q_r, false, false);
        
        // O projection
        let o_wts = g.slice(packed, [0, 0, 0, w_offset], [1, dim, 1, dim]);
        w_offset += dim;
        let o_r = g.reshape(o_wts, Shape { batch: 1, channels: 1, height: dim, width: dim });
        let attn = g.matrix_multiplication(q, o_r, false, false);
        let attn_t = g.transpose(attn, [0, 1, 3, 2]);
        let attn_out = g.reshape(attn_t, Shape { batch: 1, channels: dim, height: 1, width: seq });
        h = g.addition(h, attn_out);

        // FFN
        let h_r2 = g.reshape(h, Shape { batch: 1, channels: 1, height: dim, width: seq });
        let h_t2 = g.transpose(h_r2, [0, 1, 3, 2]);
        
        let gate_wts = g.slice(packed, [0, 0, 0, w_offset], [1, dim, 1, hidden]);
        w_offset += hidden;
        let gate_r = g.reshape(gate_wts, Shape { batch: 1, channels: 1, height: dim, width: hidden });
        let gate = g.matrix_multiplication(h_t2, gate_r, false, false);
        
        let up_wts = g.slice(packed, [0, 0, 0, w_offset], [1, dim, 1, hidden]);
        w_offset += hidden;
        let up_r = g.reshape(up_wts, Shape { batch: 1, channels: 1, height: dim, width: hidden });
        let up = g.matrix_multiplication(h_t2, up_r, false, false);
        
        let gate_sig = g.sigmoid(gate);
        let gate_silu = g.multiplication(gate, gate_sig);
        let mix = g.multiplication(gate_silu, up);
        
        let down_wts = g.slice(packed, [0, 0, 0, w_offset], [1, dim, 1, dim]);
        w_offset += dim;
        let down_r = g.reshape(down_wts, Shape { batch: 1, channels: 1, height: hidden, width: dim });
        let ffn = g.matrix_multiplication(mix, down_r, false, false);
        let ffn_t = g.transpose(ffn, [0, 1, 3, 2]);
        let ffn_out = g.reshape(ffn_t, Shape { batch: 1, channels: dim, height: 1, width: seq });
        h = g.addition(h, ffn_out);
    }

    println!("Compiling {num_layers}-layer model (dim={dim}, sp={sp})...");
    let start = Instant::now();
    let exec = g.compile(NSQualityOfService::Default).unwrap();
    println!("Compiled in {:?}", start.elapsed());

    let input = TensorData::with_f32(&vec![0.01f32; dim * sp],
        Shape { batch: 1, channels: dim, height: 1, width: sp });
    let output = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });

    for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }

    let n = 500;
    let start = Instant::now();
    for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;

    let hw_ns = exec.run_cached_with_stats(&[&input], &[&output]).unwrap();

    println!("\n=== {num_layers} LAYERS, ONE ANE DISPATCH ===");
    println!("{ms:.3}ms per dispatch ({:.3}ms per layer)", ms / num_layers as f64);
    println!("HW time: {:.3}ms", hw_ns as f64 / 1e6);
    println!("48 layers = {} dispatches: {:.2}ms total", 48/num_layers + 1, (48.0/num_layers as f64).ceil() * ms);
    println!("{:.0} tok/s (projected 48 layers)", 1000.0 / ((48.0/num_layers as f64).ceil() * ms));
    println!("GPU: ~95 tok/s");
}
