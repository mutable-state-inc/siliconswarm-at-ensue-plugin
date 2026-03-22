/// DistilBERT inference benchmark: private ANE API vs Apple's CoreML.
/// Model: distilbert-base-uncased-finetuned-sst-2-english (66M params, 6 layers, dim=768)
use std::time::Instant;

use ane::{Executable, Graph, NSQualityOfService, Shape, TensorData};
use half::{bf16, f16};
use hf_hub::api::sync::ApiBuilder;
use safetensors::{Dtype, SafeTensors};
use tokenizers::Tokenizer;

const REPO_ID: &str = "distilbert-base-uncased-finetuned-sst-2-english";
const SEQ: usize = 128;
const DIM: usize = 768;
const HEADS: usize = 12;
const HD: usize = 64;
const FFN: usize = 3072;
const LAYERS: usize = 6;
const CLS: usize = 2;

// ─── Weights ───────────────────────────────────────────────────────────────

struct LW {
    sa_ln_w: Box<[f32]>, sa_ln_b: Box<[f32]>,
    q_w: Box<[f32]>, q_b: Box<[f32]>,
    k_w: Box<[f32]>, k_b: Box<[f32]>,
    v_w: Box<[f32]>, v_b: Box<[f32]>,
    out_w: Box<[f32]>, out_b: Box<[f32]>,
    ffn_ln_w: Box<[f32]>, ffn_ln_b: Box<[f32]>,
    ffn1_w: Box<[f32]>, ffn1_b: Box<[f32]>,
    ffn2_w: Box<[f32]>, ffn2_b: Box<[f32]>,
}

struct MW {
    word_emb: Box<[f32]>, pos_emb: Box<[f32]>,
    emb_ln_w: Box<[f32]>, emb_ln_b: Box<[f32]>,
    layers: Box<[LW]>,
    pre_w: Box<[f32]>, pre_b: Box<[f32]>,
    cls_w: Box<[f32]>, cls_b: Box<[f32]>,
}

fn tf(st: &SafeTensors, name: &str) -> Box<[f32]> {
    let t = st.tensor(name).unwrap_or_else(|_| panic!("missing: {name}"));
    let b = t.data();
    match t.dtype() {
        Dtype::BF16 => b.chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0],c[1]])).to_f32()).collect(),
        Dtype::F16 => b.chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0],c[1]])).to_f32()).collect(),
        Dtype::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect(),
        other => panic!("unsupported: {other:?}"),
    }
}

fn load(st: &SafeTensors) -> MW {
    let layers: Box<[LW]> = (0..LAYERS).map(|i| {
        let p = format!("distilbert.transformer.layer.{i}");
        LW {
            sa_ln_w: tf(st, &format!("{p}.sa_layer_norm.weight")),
            sa_ln_b: tf(st, &format!("{p}.sa_layer_norm.bias")),
            q_w: tf(st, &format!("{p}.attention.q_lin.weight")),
            q_b: tf(st, &format!("{p}.attention.q_lin.bias")),
            k_w: tf(st, &format!("{p}.attention.k_lin.weight")),
            k_b: tf(st, &format!("{p}.attention.k_lin.bias")),
            v_w: tf(st, &format!("{p}.attention.v_lin.weight")),
            v_b: tf(st, &format!("{p}.attention.v_lin.bias")),
            out_w: tf(st, &format!("{p}.attention.out_lin.weight")),
            out_b: tf(st, &format!("{p}.attention.out_lin.bias")),
            ffn_ln_w: tf(st, &format!("{p}.output_layer_norm.weight")),
            ffn_ln_b: tf(st, &format!("{p}.output_layer_norm.bias")),
            ffn1_w: tf(st, &format!("{p}.ffn.lin1.weight")),
            ffn1_b: tf(st, &format!("{p}.ffn.lin1.bias")),
            ffn2_w: tf(st, &format!("{p}.ffn.lin2.weight")),
            ffn2_b: tf(st, &format!("{p}.ffn.lin2.bias")),
        }
    }).collect();
    MW {
        word_emb: tf(st, "distilbert.embeddings.word_embeddings.weight"),
        pos_emb: tf(st, "distilbert.embeddings.position_embeddings.weight"),
        emb_ln_w: tf(st, "distilbert.embeddings.LayerNorm.weight"),
        emb_ln_b: tf(st, "distilbert.embeddings.LayerNorm.bias"),
        layers,
        pre_w: tf(st, "pre_classifier.weight"),
        pre_b: tf(st, "pre_classifier.bias"),
        cls_w: tf(st, "classifier.weight"),
        cls_b: tf(st, "classifier.bias"),
    }
}

// ─── Graph helpers ─────────────────────────────────────────────────────────

fn s1() -> Shape { Shape { batch: 1, channels: 1, height: 1, width: 1 } }
fn sc(d: usize) -> Shape { Shape { batch: 1, channels: d, height: 1, width: 1 } }

fn layer_norm(g: &mut Graph, x: ane::Tensor, w: &[f32], b: &[f32], d: usize) -> ane::Tensor {
    let wt = g.constant(w, sc(d));
    let bt = g.constant(b, sc(d));
    let eps = g.constant_with_scalar(1e-12, s1());
    let nhalf = g.constant_with_scalar(-0.5, s1());
    let inv_d = g.constant_with_scalar(1.0 / d as f32, s1());
    let neg_one = g.constant_with_scalar(-1.0, s1());
    // Use reduce_sum * inv_d instead of reduce_mean for more control
    let sum = g.reduce_sum(x, 1);
    let mean = g.multiplication(sum, inv_d);
    let neg_mean = g.multiplication(mean, neg_one);
    let centered = g.addition(x, neg_mean);
    let sq = g.multiplication(centered, centered);
    let var_sum = g.reduce_sum(sq, 1);
    let var = g.multiplication(var_sum, inv_d);
    let var_eps = g.addition(var, eps);
    let rstd = g.power(var_eps, nhalf);
    let normed = g.multiplication(centered, rstd);
    let scaled = g.multiplication(normed, wt);
    g.addition(scaled, bt)
}

fn gelu(g: &mut Graph, x: ane::Tensor) -> ane::Tensor {
    let half = g.constant_with_scalar(0.5, s1());
    let one = g.constant_with_scalar(1.0, s1());
    let c = g.constant_with_scalar(0.044715, s1());
    let s = g.constant_with_scalar(0.797_884_6, s1());
    let x2 = g.multiplication(x, x);
    let x3 = g.multiplication(x2, x);
    let cx3 = g.multiplication(c, x3);
    let inner = g.addition(x, cx3);
    let arg = g.multiplication(s, inner);
    let th = g.tanh(arg);
    let one_plus = g.addition(one, th);
    let hx = g.multiplication(half, x);
    g.multiplication(hx, one_plus)
}

// ─── Compile ───────────────────────────────────────────────────────────────

/// Single encoder layer with attention mask. POST-LayerNorm.
fn compile_layer(w: &LW) -> Executable {
    let mut g = Graph::new();
    let x = g.placeholder(Shape::spatial(DIM, 1, SEQ));
    let mask = g.placeholder(Shape { batch: 1, channels: 1, height: SEQ, width: SEQ });

    // QKV
    let q = g.addition(g.inner_product(x, &w.q_w, DIM, DIM), g.constant(&w.q_b, sc(DIM)));
    let k = g.addition(g.inner_product(x, &w.k_w, DIM, DIM), g.constant(&w.k_b, sc(DIM)));
    let v = g.addition(g.inner_product(x, &w.v_w, DIM, DIM), g.constant(&w.v_b, sc(DIM)));

    // Per-head attention (Apple's ANE optimization: single-head ops for better precision)
    let scale = g.constant_with_scalar(1.0 / (HD as f32).sqrt(), s1());
    let mut head_outputs = Vec::new();
    for h in 0..HEADS {
        let qh = g.slice(q, [0, h * HD, 0, 0], [1, HD, 1, SEQ]); // [1, HD, 1, SEQ]
        let kh = g.slice(k, [0, h * HD, 0, 0], [1, HD, 1, SEQ]);
        let vh = g.slice(v, [0, h * HD, 0, 0], [1, HD, 1, SEQ]);
        // Transpose to [1, 1, SEQ, HD] for matmul — use channels=1 since single head
        let qh = g.reshape(qh, Shape { batch: 1, channels: 1, height: HD, width: SEQ });
        let qh = g.transpose(qh, [0, 1, 3, 2]); // [1, 1, SEQ, HD]
        let kh = g.reshape(kh, Shape { batch: 1, channels: 1, height: HD, width: SEQ });
        let kh = g.transpose(kh, [0, 1, 3, 2]); // [1, 1, SEQ, HD]
        let vh = g.reshape(vh, Shape { batch: 1, channels: 1, height: HD, width: SEQ });
        let vh = g.transpose(vh, [0, 1, 3, 2]); // [1, 1, SEQ, HD]
        // Q @ K^T / sqrt(d)
        let raw = g.matrix_multiplication(qh, kh, false, true); // [1, 1, SEQ, SEQ]
        let scores = g.multiplication(raw, scale);
        let masked = g.addition(scores, mask);
        let probs = g.soft_max(masked, -1);
        // attn @ V
        let ah = g.matrix_multiplication(probs, vh, false, false); // [1, 1, SEQ, HD]
        let ah = g.transpose(ah, [0, 1, 3, 2]); // [1, 1, HD, SEQ]
        let ah = g.reshape(ah, Shape::spatial(HD, 1, SEQ)); // [1, HD, 1, SEQ]
        head_outputs.push(ah);
    }
    let refs: Vec<ane::Tensor> = head_outputs.iter().copied().collect();
    let attn = g.concat(&refs, 1); // [1, DIM, 1, SEQ]

    // Output projection + residual + LayerNorm (POST-norm)
    let o = g.addition(g.inner_product(attn, &w.out_w, DIM, DIM), g.constant(&w.out_b, sc(DIM)));
    let h = g.addition(o, x);
    let h = layer_norm(&mut g, h, &w.sa_ln_w, &w.sa_ln_b, DIM);

    // FFN + residual + LayerNorm (POST-norm)
    let fc1 = g.addition(g.inner_product(h, &w.ffn1_w, DIM, FFN), g.constant(&w.ffn1_b, sc(FFN)));
    let fc1 = gelu(&mut g, fc1);
    let fc2 = g.addition(g.inner_product(fc1, &w.ffn2_w, FFN, DIM), g.constant(&w.ffn2_b, sc(DIM)));
    let out = g.addition(fc2, h);
    let _ = layer_norm(&mut g, out, &w.ffn_ln_w, &w.ffn_ln_b, DIM);

    g.compile(NSQualityOfService::UserInteractive).expect("layer compile")
}

fn compile_classifier(mw: &MW) -> Executable {
    let mut g = Graph::new();
    let x = g.placeholder(Shape::spatial(DIM, 1, SEQ));
    let pre = g.addition(g.inner_product(x, &mw.pre_w, DIM, DIM), g.constant(&mw.pre_b, sc(DIM)));
    let pre = g.relu(pre);
    let cls = g.addition(g.inner_product(pre, &mw.cls_w, DIM, CLS), g.constant(&mw.cls_b, sc(CLS)));
    let _ = cls;
    g.compile(NSQualityOfService::UserInteractive).expect("cls compile")
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  DistilBERT Benchmark: Private ANE API vs CoreML");
    eprintln!("═══════════════════════════════════════════════════════");

    // Download + load
    eprint!("  Downloading... ");
    let api = ApiBuilder::new().with_progress(false).build()?;
    let repo = api.model(REPO_ID.to_string());
    let sf = SafeTensors::deserialize(&std::fs::read(repo.get("model.safetensors")?)?)?;
    let mw = load(&sf);
    let tok = Tokenizer::from_file(
        api.model("distilbert-base-uncased".to_string()).get("tokenizer.json")?
    ).map_err(|e| format!("{e}"))?;
    eprintln!("ok");

    // Compile
    eprint!("  Compiling... ");
    let t0 = Instant::now();
    let layer_exes: Vec<Executable> = mw.layers.iter().map(|lw| compile_layer(lw)).collect();
    let cls_exe = compile_classifier(&mw);
    eprintln!("ok ({:.1}s, {} dispatches)", t0.elapsed().as_secs_f64(), layer_exes.len() + 1);

    // Allocate IOSurfaces
    let hidden_a = TensorData::new(Shape::spatial(DIM, 1, SEQ));
    let hidden_b = TensorData::new(Shape::spatial(DIM, 1, SEQ));
    let mask_td = TensorData::new(Shape { batch: 1, channels: 1, height: SEQ, width: SEQ });
    let cls_out = TensorData::new(Shape::spatial(CLS, 1, SEQ));

    // Embedding: token + position + LayerNorm on CPU
    let embed = |text: &str, seq_len: &mut usize| {
        let enc = tok.encode(text, true).expect("encode");
        let ids = enc.get_ids();
        *seq_len = ids.len().min(SEQ);
        let mut surf = hidden_a.as_f32_slice_mut();
        surf.fill(0.0);
        for pos in 0..SEQ {
            let t = if pos < *seq_len { ids[pos] as usize } else { 0 };
            for c in 0..DIM {
                surf[c * SEQ + pos] = mw.word_emb[t * DIM + c] + mw.pos_emb[pos * DIM + c];
            }
        }
        // LayerNorm
        for s in 0..SEQ {
            let mut mean = 0f32;
            for c in 0..DIM { mean += surf[c * SEQ + s]; }
            mean /= DIM as f32;
            let mut var = 0f32;
            for c in 0..DIM { let d = surf[c * SEQ + s] - mean; var += d * d; }
            var /= DIM as f32;
            let rstd = 1.0 / (var + 1e-12_f32).sqrt();
            for c in 0..DIM {
                surf[c * SEQ + s] = (surf[c * SEQ + s] - mean) * rstd * mw.emb_ln_w[c] + mw.emb_ln_b[c];
            }
        }
    };

    // Build attention mask for given sequence length
    let set_mask = |seq_len: usize| {
        let mut m = mask_td.as_f32_slice_mut();
        for row in 0..SEQ {
            for col in 0..SEQ {
                // Real tokens (row < seq_len) can attend to real tokens (col < seq_len)
                m[row * SEQ + col] = if row < seq_len && col < seq_len { 0.0 } else { -65504.0 };
            }
        }
    };

    // Forward pass
    let forward = || {
        for (i, exe) in layer_exes.iter().enumerate() {
            let (src, dst) = if i % 2 == 0 { (&hidden_a, &hidden_b) } else { (&hidden_b, &hidden_a) };
            exe.run(&[src, &mask_td], &[dst]).unwrap();
        }
        let final_h = if LAYERS % 2 == 0 { &hidden_a } else { &hidden_b };
        cls_exe.run(&[final_h], &[&cls_out]).unwrap();
    };

    let classify = || {
        let out = cls_out.as_f32_slice();
        let neg = out[0];
        let pos = out[1 * SEQ];
        if pos > neg { "POSITIVE" } else { "NEGATIVE" }
    };

    // Sanity check
    let mut slen = 0;
    embed("I love this movie!", &mut slen);
    set_mask(slen);
    forward();
    let label = classify();
    eprintln!("  Sanity: \"I love this movie!\" → {label}");

    // Benchmark (fixed mask for consistent timing)
    eprintln!("\n  Benchmarking (1000 iterations)...");
    embed("This is a test sentence for benchmarking purposes.", &mut slen);
    set_mask(slen);

    // Warmup
    for _ in 0..50 { forward(); }

    let mut times = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let start = Instant::now();
        forward();
        times.push(start.elapsed().as_micros() as f64 / 1000.0);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let p5 = times[times.len() / 20];
    let p95 = times[times.len() * 19 / 20];

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  Results (ANE private API, {LAYERS} layers, {LAYERS}+1 dispatches)");
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  Mean:   {mean:.3} ms");
    eprintln!("  Median: {median:.3} ms");
    eprintln!("  P5:     {p5:.3} ms");
    eprintln!("  P95:    {p95:.3} ms");
    eprintln!("  Min:    {:.3} ms", times[0]);
    eprintln!("═══════════════════════════════════════════════════════");

    Ok(())
}
