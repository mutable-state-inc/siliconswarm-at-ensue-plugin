/// Hill-climb R4: verify bandwidth ceiling, try split projections, different compute patterns
use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

const DIM: usize = 2048;
const HIDDEN: usize = 11008;
const LAYERS: usize = 36;
const SEQ: usize = 64;

fn bench_dispatch(label: &str, n_bench: usize, build: impl FnOnce() -> Option<ane::Executable>) -> f64 {
    eprint!("  {label:<50} ");
    let exec = match build() {
        Some(e) => e,
        None => { eprintln!("FAIL"); return 0.0; }
    };
    let input = TensorData::with_f32(&vec![0.01; DIM * SEQ],
        Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
    let output_dim = if label.contains("->H") { HIDDEN } else { DIM };
    let output = TensorData::new(Shape { batch: 1, channels: output_dim, height: 1, width: SEQ });
    for _ in 0..10 { let _ = exec.run_cached_direct(&[&input], &[&output]); }
    let start = Instant::now();
    for _ in 0..n_bench { let _ = exec.run_cached_direct(&[&input], &[&output]); }
    let us = start.elapsed().as_secs_f64() * 1e6 / n_bench as f64;
    let ms = us / 1000.0;
    eprintln!("{ms:.3}ms ({us:.0}µs)");
    ms
}

fn main() {
    eprintln!("=== HILL CLIMB R4: Bandwidth ceiling analysis ===\n");
    
    // === Exp 1: Measure raw inner_product throughput at different sizes ===
    eprintln!("--- Single inner_product at different sizes ---");
    
    // dim->dim (2048->2048): 2048*2048*2 = 8MB weights
    let t1 = bench_dispatch("IP dim->dim (8 MB weights)", 500, || {
        let mut g = Graph::new();
        let h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
        let _ = g.inner_product(h, &vec![0.01; DIM * DIM], DIM, DIM);
        g.compile(NSQualityOfService::Default).ok()
    });
    
    // dim->hidden/4 (2048->2752): 2048*2752*2 = 11.3MB weights
    let half_h = HIDDEN / 4;
    let t2 = bench_dispatch("IP dim->H/4 (11 MB weights)", 500, || {
        let mut g = Graph::new();
        let h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
        let _ = g.inner_product(h, &vec![0.01; half_h * DIM], DIM, half_h);
        g.compile(NSQualityOfService::Default).ok()
    });
    
    // dim->hidden/2 (2048->5504): 2048*5504*2 = 22.5MB weights
    let half_hidden = HIDDEN / 2;
    let t3 = bench_dispatch("IP dim->H/2 (22 MB weights)", 500, || {
        let mut g = Graph::new();
        let h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
        let _ = g.inner_product(h, &vec![0.01; half_hidden * DIM], DIM, half_hidden);
        g.compile(NSQualityOfService::Default).ok()
    });
    
    // dim->hidden (2048->11008): 2048*11008*2 = 45MB weights
    let t4 = bench_dispatch("IP dim->H (45 MB weights)", 500, || {
        let mut g = Graph::new();
        let h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
        let _ = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
        g.compile(NSQualityOfService::Default).ok()
    });
    
    eprintln!();
    
    // Bandwidth analysis
    let sizes_mb = [8.0, 11.0, 22.5, 45.0];
    let times = [t1, t2, t3, t4];
    eprintln!("--- Bandwidth analysis ---");
    for (sz, t) in sizes_mb.iter().zip(times.iter()) {
        if *t > 0.0 {
            let bw = sz / t * 1000.0; // GB/s
            eprintln!("  {sz:.1} MB in {t:.3}ms → {bw:.1} GB/s");
        }
    }
    
    eprintln!();
    
    // === Exp 2: Split hidden into 2 halves ===
    // Instead of 1 big IP dim->hidden, do 2× IP dim->hidden/2 then concat
    eprintln!("--- Split gate projection: 1×dim->H vs 2×dim->H/2+concat ---");
    
    bench_dispatch("1× IP dim->H", 500, || {
        let mut g = Graph::new();
        let h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
        let _ = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
        g.compile(NSQualityOfService::Default).ok()
    });
    
    bench_dispatch("2× IP dim->H/2 + concat", 500, || {
        let half = HIDDEN / 2;
        let mut g = Graph::new();
        let h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
        let a = g.inner_product(h, &vec![0.01; half * DIM], DIM, half);
        let b = g.inner_product(h, &vec![0.01; half * DIM], DIM, half);
        let _ = g.concat(&[a, b], 1); // concat along channels
        g.compile(NSQualityOfService::Default).ok()
    });
    
    eprintln!();
    
    // === Exp 3: Full FFN with split projections ===
    // Split each dim->hidden into 2 halves to potentially allow more fusion
    eprintln!("--- Full FFN: standard vs split projections (12L) ---");
    
    // Standard 12L
    bench_dispatch("12L standard FFN", 100, || {
        let mut g = Graph::new();
        let mut h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
        for _ in 0..12 {
            let gate = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
            let up = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
            let gs = g.sigmoid(gate);
            let gl = g.multiplication(gate, gs);
            let mix = g.multiplication(gl, up);
            let down = g.inner_product(mix, &vec![0.01; DIM * HIDDEN], HIDDEN, DIM);
            h = g.addition(h, down);
        }
        g.compile(NSQualityOfService::Default).ok()
    });
    
    // Try 12L with skip connection directly on gate (fewer ops)
    bench_dispatch("12L minimal FFN (no SiLU, gate*up)", 100, || {
        let mut g = Graph::new();
        let mut h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
        for _ in 0..12 {
            let gate = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
            let up = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
            let mix = g.multiplication(gate, up);
            let down = g.inner_product(mix, &vec![0.01; DIM * HIDDEN], HIDDEN, DIM);
            h = g.addition(h, down);
        }
        g.compile(NSQualityOfService::Default).ok()
    });
    
    // Try 12L with just gate projection (1 IP per layer instead of 3)
    bench_dispatch("12L gate-only (1 IP/layer dim->dim->dim)", 100, || {
        let mut g = Graph::new();
        let mut h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
        for _ in 0..12 {
            let proj = g.inner_product(h, &vec![0.01; DIM * DIM], DIM, DIM);
            h = g.addition(h, proj);
        }
        g.compile(NSQualityOfService::Default).ok()
    });

    eprintln!();
    
    // === Exp 4: Pipeline 3 dispatches with different I/O ===
    eprintln!("--- Pipeline: 3×12L dispatches back-to-back timing ---");
    let mut g1 = Graph::new();
    let mut h1 = g1.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
    for _ in 0..12 {
        let gate = g1.inner_product(h1, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
        let up = g1.inner_product(h1, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
        let gs = g1.sigmoid(gate);
        let gl = g1.multiplication(gate, gs);
        let mix = g1.multiplication(gl, up);
        let down = g1.inner_product(mix, &vec![0.01; DIM * HIDDEN], HIDDEN, DIM);
        h1 = g1.addition(h1, down);
    }
    if let Ok(exec) = g1.compile(NSQualityOfService::Default) {
        let input = TensorData::with_f32(&vec![0.01; DIM * SEQ],
            Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
        let output = TensorData::new(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
        for _ in 0..10 {
            for _ in 0..3 { let _ = exec.run_cached_direct(&[&input], &[&output]); }
        }
        
        // Time 3 back-to-back dispatches (= 36 layers)
        let n = 30;
        let start = Instant::now();
        for _ in 0..n {
            let _ = exec.run_cached_direct(&[&input], &[&output]);
            let _ = exec.run_cached_direct(&[&input], &[&output]);
            let _ = exec.run_cached_direct(&[&input], &[&output]);
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let tok_s = n as f64 / start.elapsed().as_secs_f64() * 1.0; // wrong, redo
        let tok_s2 = 1000.0 / ms;
        eprintln!("  3×12L pipeline: {ms:.1}ms/token → {tok_s2:.1} tok/s");
    }
}
