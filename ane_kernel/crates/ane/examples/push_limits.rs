/// Push ANE limits: test seq=1, higher fusion, full layers with norms.
use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

const DIM: usize = 2048;
const HIDDEN: usize = 11008;
const LAYERS: usize = 36;

fn build_full_layer_ip(g: &mut Graph, h: ane::Tensor, dim: usize, hidden: usize, seq: usize) -> ane::Tensor {
    // RMSNorm: x * rsqrt(mean(x^2) + eps)
    let eps = g.constant_with_scalar(1e-6, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let neg_half = g.constant_with_scalar(-0.5, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    
    // Pre-attention norm
    let sq = g.multiplication(h, h);
    let m = g.reduce_mean(sq, 1); // mean over channels
    let me = g.addition(m, eps);
    let rms = g.power(me, neg_half);
    let normed = g.multiplication(h, rms);
    
    // Attention: Q + O projections
    let q = g.inner_product(normed, &vec![0.01; dim * dim], dim, dim);
    let o = g.inner_product(q, &vec![0.01; dim * dim], dim, dim);
    let h2 = g.addition(h, o);
    
    // Post-attention norm
    let sq2 = g.multiplication(h2, h2);
    let m2 = g.reduce_mean(sq2, 1);
    let me2 = g.addition(m2, eps);
    let rms2 = g.power(me2, neg_half);
    let normed2 = g.multiplication(h2, rms2);
    
    // FFN: gate + SiLU + up + down
    let gate = g.inner_product(normed2, &vec![0.01; hidden * dim], dim, hidden);
    let up = g.inner_product(normed2, &vec![0.01; hidden * dim], dim, hidden);
    let gs = g.sigmoid(gate);
    let gl = g.multiplication(gate, gs);
    let mix = g.multiplication(gl, up);
    let down = g.inner_product(mix, &vec![0.01; dim * hidden], hidden, dim);
    g.addition(h2, down)
}

fn bench_config(label: &str, nl: usize, seq: usize, full_layer: bool) {
    eprint!("  {label:<35} ");
    
    let mut g = Graph::new();
    let mut h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: seq });
    for _ in 0..nl {
        if full_layer {
            h = build_full_layer_ip(&mut g, h, DIM, HIDDEN, seq);
        } else {
            // FFN only
            let gate = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
            let up = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
            let gs = g.sigmoid(gate);
            let gl = g.multiplication(gate, gs);
            let mix = g.multiplication(gl, up);
            let down = g.inner_product(mix, &vec![0.01; DIM * HIDDEN], HIDDEN, DIM);
            h = g.addition(h, down);
        }
    }
    
    let start = Instant::now();
    let exec = match g.compile(NSQualityOfService::Default) {
        Ok(e) => e,
        Err(e) => { eprintln!("FAIL: {e}"); return; }
    };
    let compile_ms = start.elapsed().as_secs_f64() * 1000.0;
    
    let input = TensorData::with_f32(&vec![0.01; DIM * seq],
        Shape { batch: 1, channels: DIM, height: 1, width: seq });
    let output = TensorData::new(Shape { batch: 1, channels: DIM, height: 1, width: seq });
    
    // Warmup
    for _ in 0..10 {
        if exec.run_cached_direct(&[&input], &[&output]).is_err() {
            eprintln!("FAIL: run error"); return;
        }
    }
    
    let dispatches = (LAYERS + nl - 1) / nl;
    let tokens = 30;
    let start = Instant::now();
    for _ in 0..tokens {
        for _ in 0..dispatches {
            let _ = exec.run_cached_direct(&[&input], &[&output]);
        }
    }
    let dur = start.elapsed();
    let tok_s = tokens as f64 / dur.as_secs_f64();
    let ms_tok = dur.as_secs_f64() * 1000.0 / tokens as f64;
    let ms_layer = ms_tok / LAYERS as f64;
    
    eprintln!("{tok_s:.1} tok/s  ({ms_tok:.1}ms/tok, {ms_layer:.3}ms/layer, {dispatches}×dispatch, compile {compile_ms:.0}ms)");
}

fn main() {
    eprintln!("=== PUSH ANE LIMITS: Qwen2.5-3B (dim={DIM}, hidden={HIDDEN}, {LAYERS}L) ===\n");
    
    // Test 1: seq=1 (decode mode) vs seq=64 (prefill mode)
    eprintln!("--- Sequence length: decode (seq=1) vs prefill (seq=64) ---");
    for seq in [1, 64] {
        bench_config(&format!("ffn_6L_s{seq}"), 6, seq, false);
        bench_config(&format!("ffn_12L_s{seq}"), 12, seq, false);
    }
    
    eprintln!();
    
    // Test 2: Push fusion limits (12, 18, 24, 36)
    eprintln!("--- Max fusion (FFN only, seq=64) ---");
    for nl in [12, 18, 24, 36] {
        bench_config(&format!("ffn_{nl}L_s64"), nl, 64, false);
    }
    
    eprintln!();
    
    // Test 3: Full layer (norm + attn + norm + FFN) fusion
    eprintln!("--- Full layer fusion (norm+attn+norm+FFN, seq=64) ---");
    for nl in [1, 2, 3, 4, 6] {
        bench_config(&format!("full_{nl}L_s64"), nl, 64, true);
    }
    
    eprintln!();
    
    // Test 4: Full layer at seq=1
    eprintln!("--- Full layer fusion at seq=1 (decode) ---");
    for nl in [1, 2, 3, 4, 6] {
        bench_config(&format!("full_{nl}L_s1"), nl, 1, true);
    }
}
