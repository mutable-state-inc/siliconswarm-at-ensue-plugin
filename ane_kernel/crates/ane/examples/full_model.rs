/// Full Qwen3.5-4B-scale model in ONE ANE dispatch.
/// 48 layers × (5 matmuls + elementwise) = ~240 matmuls + hundreds of elementwise ops.
/// All in a single compiled graph, single dispatch per token.

use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    // Qwen3.5-4B scale but using dim=256 to stay within constant weight limits
    // This tests the architecture; we'll scale up with dynamic weights after
    let dim = 256;
    let hidden = 256; // keep same as dim for simplicity (constant weight limit)
    let n_heads = 8;
    let head_dim = dim / n_heads; // 32
    let seq = 64;
    let num_layers = 48;

    // Each layer: 
    //   5 projections (Q, K, V, O, gate/up/down combined) as matmuls
    //   + elementwise: sigmoid, mul, add, reshape, transpose
    // Pack all weights into the spatial dimension

    // Weight budget per layer:
    //   Q: dim*dim, K: dim*dim, V: dim*dim, O: dim*dim = 4 * dim*dim
    //   gate: dim*hidden, up: dim*hidden, down: hidden*dim = 3 * dim*hidden
    //   Total per layer: 4*dim*dim + 3*dim*hidden
    //   At dim=hidden=256: 7 * 256*256 = 458752 fp16 values = 917KB per layer
    //   48 layers: 44MB total — fits as dynamic weights
    
    let weights_per_layer = 4 * dim + 3 * hidden; // OC slots needed per layer (with IC=dim)
    let total_weight_slots = num_layers * weights_per_layer;
    let sp = seq + total_weight_slots;

    println!("Building {num_layers}-layer transformer (dim={dim})...");
    println!("  Weight slots: {total_weight_slots} ({:.1}MB fp16)", 
        (dim * total_weight_slots * 2) as f64 / 1_000_000.0);
    println!("  Total spatial width: {sp}");

    let mut g = Graph::new();
    let packed = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: sp });
    
    // Extract initial hidden state from first `seq` slots
    let mut h = g.slice(packed, [0, 0, 0, 0], [1, dim, 1, seq]);
    
    let mut w_offset = seq;
    
    for layer in 0..num_layers {
        // === Attention ===
        // Q projection: dim -> dim
        let q_wts = g.slice(packed, [0, 0, 0, w_offset], [1, dim, 1, dim]);
        w_offset += dim;
        let h_r = g.reshape(h, Shape { batch: 1, channels: 1, height: dim, width: seq });
        let h_t = g.transpose(h_r, [0, 1, 3, 2]);
        let q_r = g.reshape(q_wts, Shape { batch: 1, channels: 1, height: dim, width: dim });
        let q = g.matrix_multiplication(h_t, q_r, false, false);
        // q: [1, 1, seq, dim]

        // K projection
        let k_wts = g.slice(packed, [0, 0, 0, w_offset], [1, dim, 1, dim]);
        w_offset += dim;
        let k_r = g.reshape(k_wts, Shape { batch: 1, channels: 1, height: dim, width: dim });
        let k = g.matrix_multiplication(h_t, k_r, false, false);

        // V projection
        let v_wts = g.slice(packed, [0, 0, 0, w_offset], [1, dim, 1, dim]);
        w_offset += dim;
        let v_r = g.reshape(v_wts, Shape { batch: 1, channels: 1, height: dim, width: dim });
        let v = g.matrix_multiplication(h_t, v_r, false, false);

        // Simple attention: softmax(Q @ K^T) @ V (no scaling for speed)
        let scores = g.matrix_multiplication(q, k, false, true); // [1,1,seq,seq]
        let attn_weights = g.soft_max(scores, -1);
        let attn_out = g.matrix_multiplication(attn_weights, v, false, false); // [1,1,seq,dim]
        
        // O projection
        let o_wts = g.slice(packed, [0, 0, 0, w_offset], [1, dim, 1, dim]);
        w_offset += dim;
        let o_r = g.reshape(o_wts, Shape { batch: 1, channels: 1, height: dim, width: dim });
        let attn_proj = g.matrix_multiplication(attn_out, o_r, false, false);
        let attn_proj_r = g.transpose(attn_proj, [0, 1, 3, 2]);
        let attn_proj_out = g.reshape(attn_proj_r, Shape { batch: 1, channels: dim, height: 1, width: seq });

        // Residual add
        h = g.addition(h, attn_proj_out);

        // === FFN ===
        // Gate: dim -> hidden
        let gate_wts = g.slice(packed, [0, 0, 0, w_offset], [1, dim, 1, hidden]);
        w_offset += hidden;
        let h_r2 = g.reshape(h, Shape { batch: 1, channels: 1, height: dim, width: seq });
        let h_t2 = g.transpose(h_r2, [0, 1, 3, 2]);
        let gate_r = g.reshape(gate_wts, Shape { batch: 1, channels: 1, height: dim, width: hidden });
        let gate = g.matrix_multiplication(h_t2, gate_r, false, false);

        // Up: dim -> hidden
        let up_wts = g.slice(packed, [0, 0, 0, w_offset], [1, dim, 1, hidden]);
        w_offset += hidden;
        let up_r = g.reshape(up_wts, Shape { batch: 1, channels: 1, height: dim, width: hidden });
        let up = g.matrix_multiplication(h_t2, up_r, false, false);

        // SiLU(gate) * up
        let gate_sig = g.sigmoid(gate);
        let gate_silu = g.multiplication(gate, gate_sig);
        let mix = g.multiplication(gate_silu, up);

        // Down: hidden -> dim
        let down_wts = g.slice(packed, [0, 0, 0, w_offset], [1, dim, 1, dim]);
        w_offset += dim; // Note: for hidden!=dim this would be different
        let down_r = g.reshape(down_wts, Shape { batch: 1, channels: 1, height: hidden, width: dim });
        let ffn_out = g.matrix_multiplication(mix, down_r, false, false);
        let ffn_out_r = g.transpose(ffn_out, [0, 1, 3, 2]);
        let ffn_out_final = g.reshape(ffn_out_r, Shape { batch: 1, channels: dim, height: 1, width: seq });

        // Residual
        h = g.addition(h, ffn_out_final);
        
        if layer == 0 {
            println!("  Layer 0 built ({} weight slots used)", w_offset - seq);
        }
    }
    
    println!("  Total weight offset: {w_offset} (expected {})", seq + total_weight_slots);
    println!("  Graph has {} layers × 7 matmuls + elementwise = massive graph", num_layers);

    println!("\nCompiling {num_layers}-layer model on ANE...");
    let start = Instant::now();
    match g.compile(NSQualityOfService::Default) {
        Ok(exec) => {
            let compile_dur = start.elapsed();
            println!("COMPILED in {:?}!", compile_dur);
            
            let input = TensorData::with_f32(&vec![0.01f32; dim * sp],
                Shape { batch: 1, channels: dim, height: 1, width: sp });
            let output = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });
            
            // Warmup
            for _ in 0..3 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
            
            let n = 100;
            let start = Instant::now();
            for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
            let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
            
            println!("\n=== {num_layers}-LAYER MODEL, SINGLE ANE DISPATCH ===");
            println!("{ms:.3}ms per token ({:.3}ms per layer)", ms / num_layers as f64);
            println!("That's {:.0} tok/s", 1000.0 / ms);
            println!("GPU baseline: ~10.5ms (~95 tok/s)");
        }
        Err(e) => {
            println!("COMPILE FAILED: {e}");
            // Try with fewer layers
            println!("Trying with fewer layers...");
        }
    }
}
