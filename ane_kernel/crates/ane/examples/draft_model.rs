/// Speculative decoding draft model on ANE.
/// Qwen2.5-0.5B: dim=896, hidden=4864, 24 layers.
/// Small enough that weights might fit in ANE SRAM for fast inference.
use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

const DIM: usize = 896;
const HIDDEN: usize = 4864;
const LAYERS: usize = 24;
const SEQ: usize = 64;

fn bench(label: &str, nl: usize, build: impl FnOnce(&mut Graph, ane::Tensor) -> ane::Tensor) {
    eprint!("  {label:<50} ");
    let mut g = Graph::new();
    let h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
    let _ = build(&mut g, h);
    let compile_start = Instant::now();
    let exec = match g.compile(NSQualityOfService::Default) {
        Ok(e) => e,
        Err(e) => { eprintln!("FAIL: {}", &e.to_string()[..e.to_string().len().min(50)]); return; }
    };
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;
    
    let input = TensorData::with_f32(&vec![0.01; DIM * SEQ],
        Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
    let output = TensorData::new(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
    for _ in 0..15 { let _ = exec.run_cached_direct(&[&input], &[&output]); }
    
    let dispatches = (LAYERS + nl - 1) / nl;
    let tokens = 50;
    let start = Instant::now();
    for _ in 0..tokens {
        for _ in 0..dispatches { let _ = exec.run_cached_direct(&[&input], &[&output]); }
    }
    let dur = start.elapsed();
    let tok_s = tokens as f64 / dur.as_secs_f64();
    let ms_tok = dur.as_secs_f64() * 1000.0 / tokens as f64;
    let ms_layer = ms_tok / LAYERS as f64;
    
    // Weight data per layer: gate(hidden*dim) + up(hidden*dim) + down(dim*hidden) = 3*hidden*dim
    let weight_mb_per_layer = 3.0 * HIDDEN as f64 * DIM as f64 * 2.0 / 1024.0 / 1024.0;
    let total_weight_mb = weight_mb_per_layer * nl as f64;
    
    eprintln!("{tok_s:.1} tok/s  ({ms_tok:.1}ms/tok, {ms_layer:.3}ms/L, {dispatches}d, {total_weight_mb:.0}MB/disp, compile {compile_ms:.0}ms)");
}

fn main() {
    eprintln!("=== ANE DRAFT MODEL: Qwen2.5-0.5B (dim={DIM}, hidden={HIDDEN}, {LAYERS}L) ===");
    eprintln!("Weight per layer: {:.1}MB (fp16)", 3.0 * HIDDEN as f64 * DIM as f64 * 2.0 / 1024.0 / 1024.0);
    eprintln!("Total model weights: {:.0}MB (fp16)\n", 3.0 * HIDDEN as f64 * DIM as f64 * 2.0 * LAYERS as f64 / 1024.0 / 1024.0);
    
    // FFN-only sweep
    eprintln!("--- FFN-only fusion sweep ---");
    for nl in [1, 2, 3, 4, 6, 8, 12, 24] {
        let label = format!("{nl}L FFN (inner_product)");
        bench(&label, nl, |g, mut h| {
            for _ in 0..nl {
                let gate = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
                let up = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
                let gs = g.sigmoid(gate);
                let gl = g.multiplication(gate, gs);
                let mix = g.multiplication(gl, up);
                let down = g.inner_product(mix, &vec![0.01; DIM * HIDDEN], HIDDEN, DIM);
                h = g.addition(h, down);
            }
            h
        });
    }
    
    eprintln!();
    
    // Full layer (with attention Q/O projections + norms)
    eprintln!("--- Full layer (norm + attn Q/O + norm + FFN) ---");
    for nl in [1, 2, 3, 4, 6, 8, 12, 24] {
        let label = format!("{nl}L full layer");
        bench(&label, nl, |g, mut h| {
            let eps = g.constant_with_scalar(1e-6, Shape { batch: 1, channels: 1, height: 1, width: 1 });
            let neg_half = g.constant_with_scalar(-0.5, Shape { batch: 1, channels: 1, height: 1, width: 1 });
            for _ in 0..nl {
                // RMSNorm
                let sq = g.multiplication(h, h);
                let m = g.reduce_mean(sq, 1);
                let me = g.addition(m, eps);
                let rms = g.power(me, neg_half);
                let normed = g.multiplication(h, rms);
                // Attn Q+O
                let q = g.inner_product(normed, &vec![0.01; DIM * DIM], DIM, DIM);
                let o = g.inner_product(q, &vec![0.01; DIM * DIM], DIM, DIM);
                let h2 = g.addition(h, o);
                // RMSNorm
                let sq2 = g.multiplication(h2, h2);
                let m2 = g.reduce_mean(sq2, 1);
                let me2 = g.addition(m2, eps);
                let rms2 = g.power(me2, neg_half);
                let normed2 = g.multiplication(h2, rms2);
                // FFN
                let gate = g.inner_product(normed2, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
                let up = g.inner_product(normed2, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
                let gs = g.sigmoid(gate);
                let gl = g.multiplication(gate, gs);
                let mix = g.multiplication(gl, up);
                let down = g.inner_product(mix, &vec![0.01; DIM * HIDDEN], HIDDEN, DIM);
                h = g.addition(h2, down);
            }
            h
        });
    }
    
    eprintln!();
    eprintln!("--- For speculative decode comparison ---");
    eprintln!("GPU Qwen2.5-3B-4bit decode: ~130 tok/s");
    eprintln!("If ANE draft generates K tokens at D tok/s,");
    eprintln!("and GPU verifies K+1 tokens in one forward pass,");
    eprintln!("effective = (K * accept_rate + 1) / (K/D + verify_time)");
    eprintln!();
    
    // Calculate speculative decode projections
    let gpu_verify_ms = 1000.0 / 130.0; // ~7.7ms per verify pass
    eprintln!("--- Speculative decode projections (GPU verify=130 tok/s) ---");
    for draft_tok_s in [30.0, 40.0, 50.0, 60.0, 80.0, 100.0] {
        let draft_ms = 1000.0 / draft_tok_s;
        for k in [2, 4, 6, 8] {
            for accept in [0.5, 0.7, 0.9] {
                let accepted = k as f64 * accept + 1.0; // +1 for the verified token
                let total_ms = k as f64 * draft_ms + gpu_verify_ms;
                let effective = accepted / total_ms * 1000.0;
                if effective > 130.0 {
                    eprintln!("  draft={draft_tok_s:.0} tok/s, K={k}, accept={accept:.0}%: {effective:.0} effective tok/s (+{:.0}% vs GPU)",
                        (effective / 130.0 - 1.0) * 100.0);
                }
            }
        }
    }
}
