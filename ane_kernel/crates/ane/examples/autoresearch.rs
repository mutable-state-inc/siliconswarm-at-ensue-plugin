/// ANE Autoresearch: push the limits of the Apple Neural Engine with the private API.
///
/// Fixed model: Qwen2.5-3B (dim=2048, hidden=11008, 36 layers)
/// Fixed hardware: whatever chip this runs on
/// Variable: kernel architecture (fusion, ops, seq, dispatch strategy)
/// Metric: tok/s — how many tokens the ANE can process per second through the full model
///
/// Run: cargo run --release --example autoresearch
///
/// This is analogous to bench.rustane.org — pure ANE inference throughput.

use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

// === MODEL CONFIG (fixed) ===
const MODEL: &str = "Qwen2.5-3B";
const DIM: usize = 2048;
const HIDDEN: usize = 11008;
const NUM_LAYERS: usize = 36;

#[derive(Clone)]
struct KernelConfig {
    label: String,
    fused_layers: usize,
    seq: usize,
    use_inner_product: bool,
    include_attn: bool,  // include Q+O attention projections
}

struct BenchResult {
    label: String,
    dispatch_ms: f64,
    dispatches_needed: usize,
    model_ms: f64,        // time for full model (all layers)
    model_tok_s: f64,     // tokens per second for full model
    per_layer_ms: f64,
    compile_ms: f64,
    status: String,
}

/// Build a fused FFN-only kernel using inner_product (constant weights).
fn build_ffn_ip(g: &mut Graph, h: ane::Tensor, dim: usize, hidden: usize) -> ane::Tensor {
    let gate = g.inner_product(h, &vec![0.01; hidden * dim], dim, hidden);
    let up = g.inner_product(h, &vec![0.01; hidden * dim], dim, hidden);
    let gs = g.sigmoid(gate);
    let gl = g.multiplication(gate, gs);
    let mix = g.multiplication(gl, up);
    let down = g.inner_product(mix, &vec![0.01; dim * hidden], hidden, dim);
    g.addition(h, down)
}

/// Build a fused attention (Q+O) + FFN kernel using inner_product.
fn build_attn_ffn_ip(g: &mut Graph, h: ane::Tensor, dim: usize, hidden: usize) -> ane::Tensor {
    // Attention: Q projection + O projection (simplified — no KV, no softmax)
    let q = g.inner_product(h, &vec![0.01; dim * dim], dim, dim);
    let o = g.inner_product(q, &vec![0.01; dim * dim], dim, dim);
    let h2 = g.addition(h, o);
    // FFN
    build_ffn_ip(g, h2, dim, hidden)
}

/// Build a fused FFN kernel using matrix_multiplication (dynamic weights).
fn build_ffn_matmul(g: &mut Graph, packed: ane::Tensor, h: ane::Tensor,
                    dim: usize, hidden: usize, seq: usize, wo: &mut usize) -> ane::Tensor {
    let hr = g.reshape(h, Shape { batch: 1, channels: 1, height: dim, width: seq });
    let ht = g.transpose(hr, [0, 1, 3, 2]);

    let gw = g.slice(packed, [0, 0, 0, *wo], [1, dim, 1, hidden]); *wo += hidden;
    let gr = g.reshape(gw, Shape { batch: 1, channels: 1, height: dim, width: hidden });
    let gate = g.matrix_multiplication(ht, gr, false, false);

    let uw = g.slice(packed, [0, 0, 0, *wo], [1, dim, 1, hidden]); *wo += hidden;
    let ur = g.reshape(uw, Shape { batch: 1, channels: 1, height: dim, width: hidden });
    let up = g.matrix_multiplication(ht, ur, false, false);

    let gs = g.sigmoid(gate);
    let gl = g.multiplication(gate, gs);
    let mix = g.multiplication(gl, up);

    let dw = g.slice(packed, [0, 0, 0, *wo], [1, dim, 1, dim]); *wo += dim;
    let dr = g.reshape(dw, Shape { batch: 1, channels: 1, height: hidden, width: dim });
    let f = g.matrix_multiplication(mix, dr, false, false);
    let ft = g.transpose(f, [0, 1, 3, 2]);
    let fo = g.reshape(ft, Shape { batch: 1, channels: dim, height: 1, width: seq });
    g.addition(h, fo)
}

fn benchmark(cfg: &KernelConfig) -> BenchResult {
    let dim = DIM;
    let hidden = HIDDEN;
    let seq = cfg.seq;
    let nl = cfg.fused_layers;
    let label = cfg.label.clone();

    // Check spatial limit for matmul path
    if !cfg.use_inner_product {
        let wpl = if cfg.include_attn { 2 * dim + 2 * hidden + dim } else { 2 * hidden + dim };
        let sp = seq + nl * wpl;
        if sp > 16384 {
            return BenchResult {
                label, dispatch_ms: 0.0, dispatches_needed: 0, model_ms: 0.0,
                model_tok_s: 0.0, per_layer_ms: 0.0, compile_ms: 0.0,
                status: "skip:spatial_overflow".into(),
            };
        }
    }

    let compile_start = Instant::now();
    let compile_result = if cfg.use_inner_product {
        let mut g = Graph::new();
        let mut h = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
        for _ in 0..nl {
            if cfg.include_attn {
                h = build_attn_ffn_ip(&mut g, h, dim, hidden);
            } else {
                h = build_ffn_ip(&mut g, h, dim, hidden);
            }
        }
        g.compile(NSQualityOfService::Default)
    } else {
        let wpl = 2 * hidden + dim;
        let sp = seq + nl * wpl;
        let mut g = Graph::new();
        let packed = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: sp });
        let mut h = g.slice(packed, [0, 0, 0, 0], [1, dim, 1, seq]);
        let mut wo = seq;
        for _ in 0..nl {
            h = build_ffn_matmul(&mut g, packed, h, dim, hidden, seq, &mut wo);
        }
        g.compile(NSQualityOfService::Default)
    };

    let exec = match compile_result {
        Ok(e) => e,
        Err(e) => {
            let status = if e.to_string().contains("0x20004") {
                "skip:too_large".into()
            } else {
                format!("crash:{}", e.to_string().chars().take(60).collect::<String>())
            };
            return BenchResult {
                label, dispatch_ms: 0.0, dispatches_needed: 0, model_ms: 0.0,
                model_tok_s: 0.0, per_layer_ms: 0.0,
                compile_ms: compile_start.elapsed().as_secs_f64() * 1000.0,
                status,
            };
        }
    };
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;

    // Create I/O
    let (input, output) = if cfg.use_inner_product {
        (
            TensorData::with_f32(&vec![0.01; dim * seq],
                Shape { batch: 1, channels: dim, height: 1, width: seq }),
            TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq }),
        )
    } else {
        let wpl = 2 * hidden + dim;
        let sp = seq + nl * wpl;
        (
            TensorData::with_f32(&vec![0.01; dim * sp],
                Shape { batch: 1, channels: dim, height: 1, width: sp }),
            TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq }),
        )
    };

    // Warmup
    for _ in 0..10 {
        if exec.run_cached_direct(&[&input], &[&output]).is_err() {
            return BenchResult {
                label, dispatch_ms: 0.0, dispatches_needed: 0, model_ms: 0.0,
                model_tok_s: 0.0, per_layer_ms: 0.0, compile_ms,
                status: "crash:run".into(),
            };
        }
    }

    // Benchmark: simulate full model inference
    let dispatches_needed = (NUM_LAYERS + nl - 1) / nl;
    let num_tokens = 30; // generate this many tokens
    let start = Instant::now();
    for _ in 0..num_tokens {
        for _ in 0..dispatches_needed {
            let _ = exec.run_cached_direct(&[&input], &[&output]);
        }
    }
    let elapsed = start.elapsed();
    let model_ms = elapsed.as_secs_f64() * 1000.0 / num_tokens as f64;
    let dispatch_ms = model_ms / dispatches_needed as f64;
    let model_tok_s = num_tokens as f64 / elapsed.as_secs_f64();
    let per_layer_ms = dispatch_ms / nl as f64;

    BenchResult {
        label, dispatch_ms, dispatches_needed, model_ms, model_tok_s,
        per_layer_ms, compile_ms, status: "ok".into(),
    }
}

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════╗");
    eprintln!("║  ANE Autoresearch: Private API Kernel Optimization  ║");
    eprintln!("╠══════════════════════════════════════════════════════╣");
    eprintln!("║  Model: {} (dim={}, hidden={}, {}L)", MODEL, DIM, HIDDEN, NUM_LAYERS);
    eprintln!("║  Goal:  Maximize pure ANE tok/s                     ║");
    eprintln!("║  API:   Private (_ANEInMemoryModel via inner_product)");
    eprintln!("╚══════════════════════════════════════════════════════╝");
    eprintln!();

    let mut configs: Vec<KernelConfig> = Vec::new();

    // Sweep 1: inner_product FFN-only, vary fusion
    for nl in [1, 2, 3, 4, 6, 9, 12] {
        configs.push(KernelConfig {
            label: format!("ip_ffn_{}L_s64", nl),
            fused_layers: nl, seq: 64,
            use_inner_product: true, include_attn: false,
        });
    }

    // Sweep 2: inner_product attn+FFN, vary fusion
    for nl in [1, 2, 3, 4, 6] {
        configs.push(KernelConfig {
            label: format!("ip_attn+ffn_{}L_s64", nl),
            fused_layers: nl, seq: 64,
            use_inner_product: true, include_attn: true,
        });
    }

    // Sweep 3: matmul FFN-only (dynamic weights), vary fusion
    // spatial limit: seq + nl * (2*hidden + dim) <= 16384
    // for dim=2048, hidden=11008: wpl = 2*11008 + 2048 = 24064
    // Even 1L: 64 + 24064 > 16384. So matmul won't work for this model.
    // Only works for smaller models.

    let mut results: Vec<BenchResult> = Vec::new();
    let mut best_tok_s: f64 = 0.0;
    let mut best_label = String::new();

    for cfg in &configs {
        eprint!("  {:<30} ", cfg.label);
        let r = benchmark(cfg);
        match r.status.as_str() {
            "ok" => {
                let marker = if r.model_tok_s > best_tok_s { " <<<< BEST" } else { "" };
                if r.model_tok_s > best_tok_s {
                    best_tok_s = r.model_tok_s;
                    best_label = r.label.clone();
                }
                eprintln!("{:.1} tok/s  ({:.1}ms/tok, {:.2}ms/layer, {}×dispatch, compile {:.0}ms){marker}",
                    r.model_tok_s, r.model_ms, r.per_layer_ms, r.dispatches_needed, r.compile_ms);
            }
            s => eprintln!("{s}"),
        }
        results.push(r);
    }

    // Leaderboard
    let mut ok: Vec<&BenchResult> = results.iter().filter(|r| r.status == "ok").collect();
    ok.sort_by(|a, b| b.model_tok_s.partial_cmp(&a.model_tok_s).unwrap());

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  LEADERBOARD: {} ({} layers, pure ANE)", MODEL, NUM_LAYERS);
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  {:<28} {:>8} {:>10} {:>10} {:>8}  ║", "Config", "tok/s", "ms/tok", "ms/layer", "compile");
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    for (i, r) in ok.iter().enumerate() {
        let marker = if i == 0 { " <-" } else { "" };
        eprintln!("║  {:<28} {:>7.1} {:>9.1}ms {:>9.3}ms {:>6.0}ms{marker}  ║",
            r.label, r.model_tok_s, r.model_ms, r.per_layer_ms, r.compile_ms);
    }
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("BEST: {} → {:.1} tok/s", best_label, best_tok_s);
    eprintln!();
    eprintln!("For comparison:");
    eprintln!("  GPU Qwen3.5-4B-4bit (mlx-go compiled forward): ~95 tok/s");
    eprintln!("  GPU Qwen2.5-3B-4bit (mlx-go compiled forward): ~130 tok/s");
}
