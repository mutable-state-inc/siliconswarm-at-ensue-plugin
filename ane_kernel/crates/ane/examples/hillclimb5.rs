/// Hill-climb R5: Simulate int8 via split inner_products.
/// The private API only does fp16 inner_product, but we can simulate int8
/// by splitting each projection into smaller chunks that fit in SRAM.
///
/// Key insight from R4: weights < ~16MB run at 15,000+ GB/s (SRAM cached).
/// Qwen2.5-3B gate/up projections are 45MB each (too big for SRAM).
/// If we split dim→hidden into 4× dim→hidden/4, each chunk is ~11MB (fits SRAM).

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
    let start = Instant::now();
    let exec = match g.compile(NSQualityOfService::Default) {
        Ok(e) => e,
        Err(e) => {
            let msg = e.to_string();
            eprintln!("FAIL: {}", &msg[..msg.len().min(60)]);
            return;
        }
    };
    let compile_s = start.elapsed().as_secs_f64();

    let input = TensorData::with_f32(&vec![0.01; DIM * SEQ],
        Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
    let output = TensorData::new(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
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
    eprintln!("{tok_s:.1} tok/s  ({ms_tok:.1}ms, {ms_layer:.3}ms/L, {dispatches}d, compile {compile_s:.1}s)");
}

fn main() {
    eprintln!("=== HILL CLIMB R5: Weight Tiling for SRAM Utilization ===\n");

    // Baseline: standard 12L FFN (45MB per projection)
    eprintln!("--- Baseline ---");
    bench_model("12L standard (45MB/proj)", 12, |g, mut h| {
        for _ in 0..12 {
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

    eprintln!();

    // === Strategy 1: Split gate/up into 2 halves ===
    // Each half: dim→hidden/2 = 2048→5504 = 22.5MB (closer to SRAM)
    eprintln!("--- Strategy 1: Split gate+up into 2 halves (22MB each) ---");
    for nl in [6, 9, 12] {
        let half = HIDDEN / 2;
        let label = format!("{nl}L split-2 gate+up (22MB/proj)");
        bench_model(&label, nl, |g, mut h| {
            for _ in 0..nl {
                let g1 = g.inner_product(h, &vec![0.01; half * DIM], DIM, half);
                let g2 = g.inner_product(h, &vec![0.01; half * DIM], DIM, half);
                let gate = g.concat(&[g1, g2], 1);
                let u1 = g.inner_product(h, &vec![0.01; half * DIM], DIM, half);
                let u2 = g.inner_product(h, &vec![0.01; half * DIM], DIM, half);
                let up = g.concat(&[u1, u2], 1);
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

    // === Strategy 2: Split gate/up into 4 quarters ===
    // Each quarter: dim→hidden/4 = 2048→2752 = 11.3MB (fits SRAM!)
    eprintln!("--- Strategy 2: Split gate+up into 4 quarters (11MB each) ---");
    for nl in [4, 6, 9, 12] {
        let quarter = HIDDEN / 4;
        let label = format!("{nl}L split-4 gate+up (11MB/proj)");
        bench_model(&label, nl, |g, mut h| {
            for _ in 0..nl {
                let g1 = g.inner_product(h, &vec![0.01; quarter * DIM], DIM, quarter);
                let g2 = g.inner_product(h, &vec![0.01; quarter * DIM], DIM, quarter);
                let g3 = g.inner_product(h, &vec![0.01; quarter * DIM], DIM, quarter);
                let g4 = g.inner_product(h, &vec![0.01; quarter * DIM], DIM, quarter);
                let gate = g.concat(&[g1, g2, g3, g4], 1);
                let u1 = g.inner_product(h, &vec![0.01; quarter * DIM], DIM, quarter);
                let u2 = g.inner_product(h, &vec![0.01; quarter * DIM], DIM, quarter);
                let u3 = g.inner_product(h, &vec![0.01; quarter * DIM], DIM, quarter);
                let u4 = g.inner_product(h, &vec![0.01; quarter * DIM], DIM, quarter);
                let up = g.concat(&[u1, u2, u3, u4], 1);
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

    // === Strategy 3: Split ALL projections (gate, up, AND down) ===
    eprintln!("--- Strategy 3: Split ALL projections into 4 (11MB each) ---");
    for nl in [4, 6, 9] {
        let quarter = HIDDEN / 4;
        let qdim = DIM / 4;
        let label = format!("{nl}L split-4 all projs (11MB each)");
        bench_model(&label, nl, |g, mut h| {
            for _ in 0..nl {
                // gate: 4× dim→hidden/4
                let g1 = g.inner_product(h, &vec![0.01; quarter * DIM], DIM, quarter);
                let g2 = g.inner_product(h, &vec![0.01; quarter * DIM], DIM, quarter);
                let g3 = g.inner_product(h, &vec![0.01; quarter * DIM], DIM, quarter);
                let g4 = g.inner_product(h, &vec![0.01; quarter * DIM], DIM, quarter);
                let gate = g.concat(&[g1, g2, g3, g4], 1);
                // up: 4× dim→hidden/4
                let u1 = g.inner_product(h, &vec![0.01; quarter * DIM], DIM, quarter);
                let u2 = g.inner_product(h, &vec![0.01; quarter * DIM], DIM, quarter);
                let u3 = g.inner_product(h, &vec![0.01; quarter * DIM], DIM, quarter);
                let u4 = g.inner_product(h, &vec![0.01; quarter * DIM], DIM, quarter);
                let up = g.concat(&[u1, u2, u3, u4], 1);
                let gs = g.sigmoid(gate);
                let gl = g.multiplication(gate, gs);
                let mix = g.multiplication(gl, up);
                // down: 4× hidden→dim/4 then concat
                let d1 = g.inner_product(mix, &vec![0.01; qdim * HIDDEN], HIDDEN, qdim);
                let d2 = g.inner_product(mix, &vec![0.01; qdim * HIDDEN], HIDDEN, qdim);
                let d3 = g.inner_product(mix, &vec![0.01; qdim * HIDDEN], HIDDEN, qdim);
                let d4 = g.inner_product(mix, &vec![0.01; qdim * HIDDEN], HIDDEN, qdim);
                let down = g.concat(&[d1, d2, d3, d4], 1);
                h = g.addition(h, down);
            }
            h
        });
    }

    eprintln!();

    // === Strategy 4: Reduce compute by factored projection ===
    // Instead of dim→hidden, do dim→bottleneck→hidden
    // This reduces total weights at the cost of accuracy
    eprintln!("--- Strategy 4: Factored projections (dim→512→hidden) ---");
    let bottleneck = 512;
    for nl in [9, 12] {
        let label = format!("{nl}L factored dim→{bottleneck}→hidden");
        bench_model(&label, nl, |g, mut h| {
            for _ in 0..nl {
                // Gate: dim→bottleneck→hidden (8MB + 11MB = 19MB vs 45MB)
                let g_low = g.inner_product(h, &vec![0.01; bottleneck * DIM], DIM, bottleneck);
                let gate = g.inner_product(g_low, &vec![0.01; HIDDEN * bottleneck], bottleneck, HIDDEN);
                // Up: same
                let u_low = g.inner_product(h, &vec![0.01; bottleneck * DIM], DIM, bottleneck);
                let up = g.inner_product(u_low, &vec![0.01; HIDDEN * bottleneck], bottleneck, HIDDEN);
                let gs = g.sigmoid(gate);
                let gl = g.multiplication(gate, gs);
                let mix = g.multiplication(gl, up);
                // Down: hidden→bottleneck→dim
                let d_low = g.inner_product(mix, &vec![0.01; bottleneck * HIDDEN], HIDDEN, bottleneck);
                let down = g.inner_product(d_low, &vec![0.01; DIM * bottleneck], bottleneck, DIM);
                h = g.addition(h, down);
            }
            h
        });
    }
}
