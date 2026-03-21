/// Hill-climbing ANE kernel optimization for Qwen2.5-3B.
/// Baseline: 10.6 tok/s (ip_ffn_12L_s64)
/// Try every optimization we can think of.

use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

const DIM: usize = 2048;
const HIDDEN: usize = 11008;
const LAYERS: usize = 36;
const SEQ: usize = 64;

fn bench(label: &str, build: impl FnOnce() -> Option<ane::Executable>) {
    eprint!("  {label:<45} ");
    let exec = match build() {
        Some(e) => e,
        None => { eprintln!("SKIP"); return; }
    };

    let input = TensorData::with_f32(&vec![0.01; DIM * SEQ],
        Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
    let output = TensorData::new(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });

    // Warmup
    for _ in 0..15 {
        if exec.run_cached_direct(&[&input], &[&output]).is_err() {
            eprintln!("FAIL:run"); return;
        }
    }

    // Benchmark: 30 tokens × dispatches_needed
    // We'll measure dispatch time directly, then compute model tok/s
    let n = 500;
    let start = Instant::now();
    for _ in 0..n {
        let _ = exec.run_cached_direct(&[&input], &[&output]);
    }
    let dispatch_us = start.elapsed().as_secs_f64() * 1_000_000.0 / n as f64;
    let dispatch_ms = dispatch_us / 1000.0;

    eprintln!("{dispatch_ms:.3}ms/dispatch  ({dispatch_us:.0}µs)");
}

fn bench_model(label: &str, dispatches: usize, build: impl FnOnce() -> Option<ane::Executable>) {
    eprint!("  {label:<45} ");
    let exec = match build() {
        Some(e) => e,
        None => { eprintln!("SKIP"); return; }
    };

    let input = TensorData::with_f32(&vec![0.01; DIM * SEQ],
        Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
    let output = TensorData::new(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });

    for _ in 0..10 {
        let _ = exec.run_cached_direct(&[&input], &[&output]);
    }

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
    eprintln!("{tok_s:.1} tok/s  ({ms_tok:.1}ms/tok, {ms_layer:.3}ms/L, {dispatches}×disp)");
}

fn build_ffn_ip(g: &mut Graph, h: ane::Tensor, nl: usize) -> ane::Tensor {
    let mut out = h;
    for _ in 0..nl {
        let gate = g.inner_product(out, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
        let up = g.inner_product(out, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
        let gs = g.sigmoid(gate);
        let gl = g.multiplication(gate, gs);
        let mix = g.multiplication(gl, up);
        let down = g.inner_product(mix, &vec![0.01; DIM * HIDDEN], HIDDEN, DIM);
        out = g.addition(out, down);
    }
    out
}

fn main() {
    eprintln!("=== HILL CLIMB: Qwen2.5-3B ANE Kernel Optimization ===");
    eprintln!("Baseline: 10.6 tok/s (ip_ffn_12L_s64, 3 dispatches)\n");

    // === EXPERIMENT 1: QoS levels ===
    eprintln!("--- Exp 1: QoS levels (12L FFN) ---");
    for (qos_label, qos) in [
        ("Default", NSQualityOfService::Default),
        ("UserInteractive", NSQualityOfService::UserInteractive),
        ("UserInitiated", NSQualityOfService::UserInitiated),
        ("Utility", NSQualityOfService::Utility),
    ] {
        let label = format!("12L_ffn_qos={qos_label}");
        bench_model(&label, 3, || {
            let mut g = Graph::new();
            let h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
            build_ffn_ip(&mut g, h, 12);
            g.compile(qos).ok()
        });
    }

    eprintln!();

    // === EXPERIMENT 2: Fusion sweet spot (9 vs 12 vs try 13-17) ===
    eprintln!("--- Exp 2: Find exact fusion limit ---");
    for nl in [9, 10, 11, 12, 13, 14, 15] {
        let dispatches = (LAYERS + nl - 1) / nl;
        let label = format!("{nl}L_ffn_s64 ({dispatches}×disp)");
        bench_model(&label, dispatches, || {
            let mut g = Graph::new();
            let h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
            build_ffn_ip(&mut g, h, nl);
            g.compile(NSQualityOfService::Default).ok()
        });
    }

    eprintln!();

    // === EXPERIMENT 3: Reduce gate+up to single fused IP ===
    // Instead of separate gate and up projections, try a single 2*hidden projection
    // then split. This halves the number of inner_product ops.
    eprintln!("--- Exp 3: Fused gate+up projection (1 IP instead of 2) ---");
    for nl in [6, 12] {
        let dispatches = (LAYERS + nl - 1) / nl;
        let label = format!("{nl}L_fused_gate_up ({dispatches}×disp)");
        bench_model(&label, dispatches, || {
            let mut g = Graph::new();
            let mut h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
            for _ in 0..nl {
                // Single projection: dim -> 2*hidden
                let gate_up = g.inner_product(h, &vec![0.01; 2 * HIDDEN * DIM], DIM, 2 * HIDDEN);
                // Split into gate and up via slicing
                let gate = g.slice(gate_up, [0, 0, 0, 0], [1, HIDDEN, 1, SEQ]);
                let up = g.slice(gate_up, [0, HIDDEN, 0, 0], [1, HIDDEN, 1, SEQ]);
                let gs = g.sigmoid(gate);
                let gl = g.multiplication(gate, gs);
                let mix = g.multiplication(gl, up);
                let down = g.inner_product(mix, &vec![0.01; DIM * HIDDEN], HIDDEN, DIM);
                h = g.addition(h, down);
            }
            g.compile(NSQualityOfService::Default).ok()
        });
    }

    eprintln!();

    // === EXPERIMENT 4: Dispatch overhead isolation ===
    eprintln!("--- Exp 4: Raw dispatch overhead (minimal kernel) ---");
    bench("noop_add_only", || {
        let mut g = Graph::new();
        let h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
        let _ = g.addition(h, h);
        g.compile(NSQualityOfService::Default).ok()
    });
    bench("1_inner_product_dim->dim", || {
        let mut g = Graph::new();
        let h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
        let _ = g.inner_product(h, &vec![0.01; DIM * DIM], DIM, DIM);
        g.compile(NSQualityOfService::Default).ok()
    });
    bench("1_inner_product_dim->hidden", || {
        let mut g = Graph::new();
        let h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
        let _ = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
        g.compile(NSQualityOfService::Default).ok()
    });
    bench("1L_ffn (3 IPs + silu + residual)", || {
        let mut g = Graph::new();
        let h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
        build_ffn_ip(&mut g, h, 1);
        g.compile(NSQualityOfService::Default).ok()
    });

    eprintln!();

    // === EXPERIMENT 5: Try leaky_relu instead of sigmoid for SiLU ===
    eprintln!("--- Exp 5: Activation variants ---");
    for nl in [12] {
        let dispatches = (LAYERS + nl - 1) / nl;
        
        // Standard SiLU: x * sigmoid(x)
        let label = format!("{nl}L_silu (baseline)");
        bench_model(&label, dispatches, || {
            let mut g = Graph::new();
            let h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
            build_ffn_ip(&mut g, h, nl);
            g.compile(NSQualityOfService::Default).ok()
        });

        // ReLU instead of SiLU (fewer ops)
        let label = format!("{nl}L_relu (simpler activation)");
        bench_model(&label, dispatches, || {
            let mut g = Graph::new();
            let mut h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
            for _ in 0..nl {
                let gate = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
                let up = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
                let gl = g.relu(gate); // ReLU instead of SiLU
                let mix = g.multiplication(gl, up);
                let down = g.inner_product(mix, &vec![0.01; DIM * HIDDEN], HIDDEN, DIM);
                h = g.addition(h, down);
            }
            g.compile(NSQualityOfService::Default).ok()
        });

        // No activation at all (just gate * up)
        let label = format!("{nl}L_no_act (gate*up only)");
        bench_model(&label, dispatches, || {
            let mut g = Graph::new();
            let mut h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
            for _ in 0..nl {
                let gate = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
                let up = g.inner_product(h, &vec![0.01; HIDDEN * DIM], DIM, HIDDEN);
                let mix = g.multiplication(gate, up);
                let down = g.inner_product(mix, &vec![0.01; DIM * HIDDEN], HIDDEN, DIM);
                h = g.addition(h, down);
            }
            g.compile(NSQualityOfService::Default).ok()
        });
    }
}
