/// Hill-climb R6: Push split-4 further. Try split-8, split-16.
/// Also try asymmetric splits and different fusion combinations.
use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

const DIM: usize = 2048;
const HIDDEN: usize = 11008;
const LAYERS: usize = 36;
const SEQ: usize = 64;

fn bench_model(label: &str, nl: usize, build: impl FnOnce(&mut Graph, ane::Tensor) -> ane::Tensor) {
    eprint!("  {label:<55} ");
    let mut g = Graph::new();
    let h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
    let _ = build(&mut g, h);
    let exec = match g.compile(NSQualityOfService::Default) {
        Ok(e) => e,
        Err(e) => { eprintln!("FAIL: {}", &e.to_string()[..e.to_string().len().min(50)]); return; }
    };
    let input = TensorData::with_f32(&vec![0.01; DIM * SEQ],
        Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
    let output = TensorData::new(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
    for _ in 0..10 { let _ = exec.run_cached_direct(&[&input], &[&output]); }
    let dispatches = (LAYERS + nl - 1) / nl;
    let tokens = 30;
    let start = Instant::now();
    for _ in 0..tokens {
        for _ in 0..dispatches { let _ = exec.run_cached_direct(&[&input], &[&output]); }
    }
    let dur = start.elapsed();
    let tok_s = tokens as f64 / dur.as_secs_f64();
    let ms_tok = dur.as_secs_f64() * 1000.0 / tokens as f64;
    let ms_layer = ms_tok / LAYERS as f64;
    eprintln!("{tok_s:.1} tok/s  ({ms_layer:.3}ms/L, {dispatches}d)");
}

fn split_ip(g: &mut Graph, h: ane::Tensor, dim: usize, out: usize, splits: usize) -> ane::Tensor {
    let chunk = out / splits;
    let parts: Vec<ane::Tensor> = (0..splits)
        .map(|_| g.inner_product(h, &vec![0.01; chunk * dim], dim, chunk))
        .collect();
    g.concat(&parts, 1)
}

fn split_ip_down(g: &mut Graph, h: ane::Tensor, hidden: usize, dim: usize, splits: usize) -> ane::Tensor {
    let chunk = dim / splits;
    let parts: Vec<ane::Tensor> = (0..splits)
        .map(|_| g.inner_product(h, &vec![0.01; chunk * hidden], hidden, chunk))
        .collect();
    g.concat(&parts, 1)
}

fn main() {
    eprintln!("=== HILL CLIMB R6: Optimal Tiling ===\n");

    // Best from R5: split-4 all projs, 9L = 10.8 tok/s (2.573ms/L)
    eprintln!("--- Baseline: split-4 all, 9L ---");
    bench_model("9L split-4-all (baseline from R5)", 9, |g, mut h| {
        for _ in 0..9 {
            let gate = split_ip(g, h, DIM, HIDDEN, 4);
            let up = split_ip(g, h, DIM, HIDDEN, 4);
            let gs = g.sigmoid(gate);
            let gl = g.multiplication(gate, gs);
            let mix = g.multiplication(gl, up);
            let down = split_ip_down(g, mix, HIDDEN, DIM, 4);
            h = g.addition(h, down);
        }
        h
    });

    eprintln!();

    // === Try split-8 (5.5MB per chunk — deep in SRAM) ===
    eprintln!("--- Split-8 gate+up (5.5MB per chunk) ---");
    for nl in [6, 9, 12] {
        let label = format!("{nl}L split-8 gate+up");
        bench_model(&label, nl, |g, mut h| {
            for _ in 0..nl {
                let gate = split_ip(g, h, DIM, HIDDEN, 8);
                let up = split_ip(g, h, DIM, HIDDEN, 8);
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

    // === Split-8 ALL projections ===
    eprintln!("--- Split-8 ALL projections ---");
    for nl in [6, 9, 12] {
        let label = format!("{nl}L split-8 all");
        bench_model(&label, nl, |g, mut h| {
            for _ in 0..nl {
                let gate = split_ip(g, h, DIM, HIDDEN, 8);
                let up = split_ip(g, h, DIM, HIDDEN, 8);
                let gs = g.sigmoid(gate);
                let gl = g.multiplication(gate, gs);
                let mix = g.multiplication(gl, up);
                let down = split_ip_down(g, mix, HIDDEN, DIM, 8);
                h = g.addition(h, down);
            }
            h
        });
    }

    eprintln!();

    // === Try split-16 (2.75MB per chunk) ===
    eprintln!("--- Split-16 ALL projections (2.75MB per chunk) ---");
    for nl in [4, 6, 9] {
        let label = format!("{nl}L split-16 all");
        bench_model(&label, nl, |g, mut h| {
            for _ in 0..nl {
                let gate = split_ip(g, h, DIM, HIDDEN, 16);
                let up = split_ip(g, h, DIM, HIDDEN, 16);
                let gs = g.sigmoid(gate);
                let gl = g.multiplication(gate, gs);
                let mix = g.multiplication(gl, up);
                let down = split_ip_down(g, mix, HIDDEN, DIM, 16);
                h = g.addition(h, down);
            }
            h
        });
    }

    eprintln!();

    // === Asymmetric: split gate+up into 4, keep down unsplit ===
    eprintln!("--- Asymmetric: split-4 gate+up only, full down ---");
    for nl in [9, 12] {
        let label = format!("{nl}L split-4 gate+up, full down");
        bench_model(&label, nl, |g, mut h| {
            for _ in 0..nl {
                let gate = split_ip(g, h, DIM, HIDDEN, 4);
                let up = split_ip(g, h, DIM, HIDDEN, 4);
                let gs = g.sigmoid(gate);
                let gl = g.multiplication(gate, gs);
                let mix = g.multiplication(gl, up);
                let down = g.inner_product(mix, &vec![0.01; DIM * HIDDEN], HIDDEN, DIM);
                h = g.addition(h, down);
            }
            h
        });
    }
}
