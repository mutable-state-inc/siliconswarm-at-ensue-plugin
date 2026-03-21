/// Hill-climb R3: find the seq × fusion sweet spot
use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

const DIM: usize = 2048;
const HIDDEN: usize = 11008;
const LAYERS: usize = 36;

fn bench(label: &str, seq: usize, nl: usize) {
    eprint!("  {label:<40} ");
    let mut g = Graph::new();
    let mut h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: seq });
    for _ in 0..nl {
        let gate = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
        let up = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
        let gs = g.sigmoid(gate);
        let gl = g.multiplication(gate, gs);
        let mix = g.multiplication(gl, up);
        let down = g.inner_product(mix, &vec![0.01; DIM * HIDDEN], HIDDEN, DIM);
        h = g.addition(h, down);
    }
    let exec = match g.compile(NSQualityOfService::Default) {
        Ok(e) => e,
        Err(e) => {
            let msg = e.to_string();
            if msg.len() > 40 { eprintln!("FAIL"); } else { eprintln!("FAIL: {msg}"); }
            return;
        }
    };
    let input = TensorData::with_f32(&vec![0.01; DIM * seq],
        Shape { batch: 1, channels: DIM, height: 1, width: seq });
    let output = TensorData::new(Shape { batch: 1, channels: DIM, height: 1, width: seq });
    for _ in 0..10 { let _ = exec.run_cached_direct(&[&input], &[&output]); }
    
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
    let prefill_tok_s = seq as f64 * tok_s;
    eprintln!("decode={tok_s:.1} tok/s  prefill={prefill_tok_s:.0} tok/s  ({ms_tok:.1}ms/tok, {ms_layer:.3}ms/L, {dispatches}×d)");
}

fn main() {
    eprintln!("=== HILL CLIMB R3: seq × fusion grid search ===\n");
    eprintln!("  {:<40} {:>10}  {:>12}  {:>20}", "Config", "decode", "prefill", "details");
    eprintln!("  {}", "-".repeat(95));
    
    // Grid: fusion × seq
    for nl in [1, 2, 3, 4, 6, 9, 12] {
        for seq in [64, 128, 256, 512] {
            let label = format!("{nl}L_s{seq}");
            bench(&label, seq, nl);
        }
        eprintln!();
    }
}
