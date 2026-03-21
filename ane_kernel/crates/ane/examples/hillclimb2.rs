/// Hill-climb round 2: bandwidth optimization
use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

const DIM: usize = 2048;
const HIDDEN: usize = 11008;
const LAYERS: usize = 36;

fn bench_tok(label: &str, seq: usize, nl: usize, build: impl FnOnce(&mut Graph, ane::Tensor) -> ane::Tensor) {
    eprint!("  {label:<50} ");
    let mut g = Graph::new();
    let h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: seq });
    let _ = build(&mut g, h);
    
    let exec = match g.compile(NSQualityOfService::Default) {
        Ok(e) => e,
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("0x20004") { eprintln!("FAIL:too_large"); }
            else { eprintln!("FAIL:{}", &msg[..msg.len().min(50)]); }
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
    let ms_tok = dur.as_secs_f64() * 1000.0 / tokens as f64;
    let tok_s = tokens as f64 / dur.as_secs_f64();
    let ms_layer = ms_tok / LAYERS as f64;
    
    // For prefill, effective tok/s = seq * tok_s (processing seq tokens per "token")
    let prefill_tok_s = seq as f64 * tok_s;
    
    eprintln!("{tok_s:.1} tok/s  (prefill: {prefill_tok_s:.0} tok/s, {ms_tok:.1}ms, {ms_layer:.3}ms/L, {dispatches}×d)");
}

fn build_ffn(g: &mut Graph, mut h: ane::Tensor, nl: usize) -> ane::Tensor {
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
}

fn main() {
    eprintln!("=== HILL CLIMB R2: Bandwidth & Prefill Optimization ===\n");
    
    // === Exp 1: Larger seq for prefill throughput ===
    // seq=64 is min, but seq=128/256 amortizes dispatch overhead and weight loading
    eprintln!("--- Seq length scaling (9L FFN, prefill mode) ---");
    for seq in [64, 128, 256, 512] {
        let label = format!("9L_ffn_s{seq}");
        bench_tok(&label, seq, 9, |g, h| build_ffn(g, h, 9));
    }
    
    eprintln!();
    
    // === Exp 2: Seq scaling at different fusion levels ===
    eprintln!("--- Seq scaling at 6L fusion ---");
    for seq in [64, 128, 256] {
        let label = format!("6L_ffn_s{seq}");
        bench_tok(&label, seq, 6, |g, h| build_ffn(g, h, 6));
    }

    eprintln!();
    
    // === Exp 3: Seq scaling at 4L fusion ===
    eprintln!("--- Seq scaling at 4L fusion ---");
    for seq in [64, 128, 256, 512] {
        let label = format!("4L_ffn_s{seq}");
        bench_tok(&label, seq, 4, |g, h| build_ffn(g, h, 4));
    }

    eprintln!();
    
    // === Exp 4: Seq scaling at 3L fusion ===
    eprintln!("--- Seq scaling at 3L fusion ---");
    for seq in [64, 128, 256, 512, 1024] {
        let label = format!("3L_ffn_s{seq}");
        bench_tok(&label, seq, 3, |g, h| build_ffn(g, h, 3));
    }

    eprintln!();
    
    // === Exp 5: Seq scaling at 2L and 1L ===
    eprintln!("--- Seq scaling at 2L fusion ---");
    for seq in [64, 128, 256, 512, 1024] {
        let label = format!("2L_ffn_s{seq}");
        bench_tok(&label, seq, 2, |g, h| build_ffn(g, h, 2));
    }
    
    eprintln!();
    eprintln!("--- Seq scaling at 1L fusion ---");
    for seq in [64, 128, 256, 512, 1024] {
        let label = format!("1L_ffn_s{seq}");
        bench_tok(&label, seq, 1, |g, h| build_ffn(g, h, 1));
    }
}
