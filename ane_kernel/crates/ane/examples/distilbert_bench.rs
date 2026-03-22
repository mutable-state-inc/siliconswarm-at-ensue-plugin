/// DistilBERT inference benchmark: private ANE API vs Apple's CoreML.
///
/// Target: beat CoreML's 5.86ms on M1 Max for DistilBERT sentiment classification.
/// Model: distilbert-base-uncased-finetuned-sst-2-english (66M params, 6 layers, dim=768)
///
/// Run: cargo run --release --example distilbert_bench
use std::time::Instant;

use ane::{Executable, Graph, NSQualityOfService, Shape, TensorData};
use half::{bf16, f16};
use hf_hub::api::sync::ApiBuilder;
use safetensors::{Dtype, SafeTensors};
use tokenizers::Tokenizer;

const REPO_ID: &str = "distilbert-base-uncased-finetuned-sst-2-english";
const SEQ_LEN: usize = 128;
const DIM: usize = 768;
const NUM_HEADS: usize = 12;
const HEAD_DIM: usize = 64; // 768/12
const FFN_DIM: usize = 3072;
const NUM_LAYERS: usize = 6;
const VOCAB_SIZE: usize = 30522;
const NUM_CLASSES: usize = 2;

// ─── Weights ───────────────────────────────────────────────────────────────

struct LayerWeights {
    sa_ln_w: Box<[f32]>, sa_ln_b: Box<[f32]>,
    q_w: Box<[f32]>, q_b: Box<[f32]>,
    k_w: Box<[f32]>, k_b: Box<[f32]>,
    v_w: Box<[f32]>, v_b: Box<[f32]>,
    out_w: Box<[f32]>, out_b: Box<[f32]>,
    ffn_ln_w: Box<[f32]>, ffn_ln_b: Box<[f32]>,
    ffn1_w: Box<[f32]>, ffn1_b: Box<[f32]>,
    ffn2_w: Box<[f32]>, ffn2_b: Box<[f32]>,
}

struct ModelWeights {
    word_emb: Box<[f32]>,
    pos_emb: Box<[f32]>,
    emb_ln_w: Box<[f32]>, emb_ln_b: Box<[f32]>,
    layers: Box<[LayerWeights]>,
    pre_cls_w: Box<[f32]>, pre_cls_b: Box<[f32]>,
    cls_w: Box<[f32]>, cls_b: Box<[f32]>,
}

fn tensor_f32(st: &SafeTensors, name: &str) -> Box<[f32]> {
    let t = st.tensor(name).unwrap_or_else(|_| panic!("missing: {name}"));
    let b = t.data();
    match t.dtype() {
        Dtype::BF16 => b.chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0],c[1]])).to_f32()).collect(),
        Dtype::F16 => b.chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0],c[1]])).to_f32()).collect(),
        Dtype::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect(),
        other => panic!("unsupported dtype: {other:?}"),
    }
}

fn load_weights(st: &SafeTensors) -> ModelWeights {
    let layers: Box<[LayerWeights]> = (0..NUM_LAYERS).map(|i| {
        let p = format!("distilbert.transformer.layer.{i}");
        LayerWeights {
            sa_ln_w: tensor_f32(st, &format!("{p}.sa_layer_norm.weight")),
            sa_ln_b: tensor_f32(st, &format!("{p}.sa_layer_norm.bias")),
            q_w: tensor_f32(st, &format!("{p}.attention.q_lin.weight")),
            q_b: tensor_f32(st, &format!("{p}.attention.q_lin.bias")),
            k_w: tensor_f32(st, &format!("{p}.attention.k_lin.weight")),
            k_b: tensor_f32(st, &format!("{p}.attention.k_lin.bias")),
            v_w: tensor_f32(st, &format!("{p}.attention.v_lin.weight")),
            v_b: tensor_f32(st, &format!("{p}.attention.v_lin.bias")),
            out_w: tensor_f32(st, &format!("{p}.attention.out_lin.weight")),
            out_b: tensor_f32(st, &format!("{p}.attention.out_lin.bias")),
            ffn_ln_w: tensor_f32(st, &format!("{p}.output_layer_norm.weight")),
            ffn_ln_b: tensor_f32(st, &format!("{p}.output_layer_norm.bias")),
            ffn1_w: tensor_f32(st, &format!("{p}.ffn.lin1.weight")),
            ffn1_b: tensor_f32(st, &format!("{p}.ffn.lin1.bias")),
            ffn2_w: tensor_f32(st, &format!("{p}.ffn.lin2.weight")),
            ffn2_b: tensor_f32(st, &format!("{p}.ffn.lin2.bias")),
        }
    }).collect();
    ModelWeights {
        word_emb: tensor_f32(st, "distilbert.embeddings.word_embeddings.weight"),
        pos_emb: tensor_f32(st, "distilbert.embeddings.position_embeddings.weight"),
        emb_ln_w: tensor_f32(st, "distilbert.embeddings.LayerNorm.weight"),
        emb_ln_b: tensor_f32(st, "distilbert.embeddings.LayerNorm.bias"),
        layers,
        pre_cls_w: tensor_f32(st, "pre_classifier.weight"),
        pre_cls_b: tensor_f32(st, "pre_classifier.bias"),
        cls_w: tensor_f32(st, "classifier.weight"),
        cls_b: tensor_f32(st, "classifier.bias"),
    }
}

// ─── ANE Graph helpers ─────────────────────────────────────────────────────

fn scalar() -> Shape { Shape { batch: 1, channels: 1, height: 1, width: 1 } }
fn ch(d: usize) -> Shape { Shape { batch: 1, channels: d, height: 1, width: 1 } }

fn layer_norm(g: &mut Graph, input: ane::Tensor, weight: &[f32], bias: &[f32],
              d: usize, eps: f64) -> ane::Tensor {
    let w = g.constant(weight, ch(d));
    let b = g.constant(bias, ch(d));
    let eps_c = g.constant_with_scalar(eps as f32, scalar());
    let inv_d = g.constant_with_scalar(1.0 / d as f32, scalar());
    let neg_half = g.constant_with_scalar(-0.5, scalar());
    let neg_one = g.constant_with_scalar(-1.0, scalar());
    let sum = g.reduce_sum(input, 1);
    let mean = g.multiplication(sum, inv_d);
    let neg_mean = g.multiplication(mean, neg_one);
    let centered = g.addition(input, neg_mean);
    let sq = g.multiplication(centered, centered);
    let var_sum = g.reduce_sum(sq, 1);
    let var = g.multiplication(var_sum, inv_d);
    let var_eps = g.addition(var, eps_c);
    let rstd = g.power(var_eps, neg_half);
    let normed = g.multiplication(centered, rstd);
    let scaled = g.multiplication(normed, w);
    g.addition(scaled, b)
}

fn gelu(g: &mut Graph, input: ane::Tensor) -> ane::Tensor {
    let half = g.constant_with_scalar(0.5, scalar());
    let one = g.constant_with_scalar(1.0, scalar());
    let coeff = g.constant_with_scalar(0.044715, scalar());
    let sqrt_2pi = g.constant_with_scalar(0.797_884_6, scalar());
    let x2 = g.multiplication(input, input);
    let x3 = g.multiplication(x2, input);
    let scaled_cube = g.multiplication(coeff, x3);
    let inner = g.addition(input, scaled_cube);
    let tanh_arg = g.multiplication(sqrt_2pi, inner);
    let tanh_val = g.tanh(tanh_arg);
    let one_plus = g.addition(one, tanh_val);
    let half_x = g.multiplication(half, input);
    g.multiplication(half_x, one_plus)
}

// ─── Compile encoder layer ────────────────────────────────────────────────

/// Full encoder layer: LayerNorm + attention + residual + LayerNorm + FFN + residual
/// All in one ANE graph — maximizes ops per dispatch.
/// DistilBERT encoder layer — POST-LayerNorm architecture:
/// Attention(x) + x → LayerNorm → FFN → + → LayerNorm → output
fn compile_encoder_layer(w: &LayerWeights) -> Executable {
    let mut g = Graph::new();
    let x = g.placeholder(Shape::spatial(DIM, 1, SEQ_LEN));

    // Q, K, V projections (from raw input, NOT from LayerNorm — post-norm architecture)
    let q = g.inner_product(x, &w.q_w, DIM, DIM);
    let q_b = g.constant(&w.q_b, ch(DIM));
    let q = g.addition(q, q_b);
    let k = g.inner_product(x, &w.k_w, DIM, DIM);
    let k_b = g.constant(&w.k_b, ch(DIM));
    let k = g.addition(k, k_b);
    let v = g.inner_product(x, &w.v_w, DIM, DIM);
    let v_b = g.constant(&w.v_b, ch(DIM));
    let v = g.addition(v, v_b);

    let q = g.reshape(q, Shape { batch: 1, channels: NUM_HEADS, height: HEAD_DIM, width: SEQ_LEN });
    let k = g.reshape(k, Shape { batch: 1, channels: NUM_HEADS, height: HEAD_DIM, width: SEQ_LEN });
    let v = g.reshape(v, Shape { batch: 1, channels: NUM_HEADS, height: HEAD_DIM, width: SEQ_LEN });
    let hw = [0, 1, 3, 2];
    let q = g.transpose(q, hw);
    let k = g.transpose(k, hw);
    let v = g.transpose(v, hw);

    let scale = g.constant_with_scalar(1.0 / (HEAD_DIM as f32).sqrt(), scalar());
    let raw_scores = g.matrix_multiplication(q, k, false, true);
    let scores = g.multiplication(raw_scores, scale);
    let probs = g.soft_max(scores, -1);
    let attn_raw = g.matrix_multiplication(probs, v, false, false);
    let attn = g.transpose(attn_raw, hw);
    let attn = g.reshape(attn, Shape::spatial(DIM, 1, SEQ_LEN));

    // Output projection
    let o = g.inner_product(attn, &w.out_w, DIM, DIM);
    let o_b = g.constant(&w.out_b, ch(DIM));
    let o = g.addition(o, o_b);

    // POST-norm: residual THEN LayerNorm
    let sa_out = g.addition(o, x);
    let sa_normed = layer_norm(&mut g, sa_out, &w.sa_ln_w, &w.sa_ln_b, DIM, 1e-12);

    // FFN
    let fc1 = g.inner_product(sa_normed, &w.ffn1_w, DIM, FFN_DIM);
    let fc1_b = g.constant(&w.ffn1_b, ch(FFN_DIM));
    let fc1 = g.addition(fc1, fc1_b);
    let fc1 = gelu(&mut g, fc1);
    let fc2 = g.inner_product(fc1, &w.ffn2_w, FFN_DIM, DIM);
    let fc2_b = g.constant(&w.ffn2_b, ch(DIM));
    let ffn_out = g.addition(fc2, fc2_b);

    // POST-norm: residual THEN LayerNorm
    let ffn_res = g.addition(ffn_out, sa_normed);
    let _ = layer_norm(&mut g, ffn_res, &w.ffn_ln_w, &w.ffn_ln_b, DIM, 1e-12);

    g.compile(NSQualityOfService::UserInteractive).expect("encoder layer compile")
}

/// Two POST-norm encoder layers fused into one ANE graph.
fn compile_fused_two_layers(w0: &LayerWeights, w1: &LayerWeights) -> Result<Executable, ane::Error> {
    let mut g = Graph::new();
    let x = g.placeholder(Shape::spatial(DIM, 1, SEQ_LEN));
    let hw = [0, 1, 3, 2];
    let scale = g.constant_with_scalar(1.0 / (HEAD_DIM as f32).sqrt(), scalar());

    // === Layer 0 (POST-norm) ===
    let q = g.inner_product(x, &w0.q_w, DIM, DIM);
    let q_b = g.constant(&w0.q_b, ch(DIM)); let q = g.addition(q, q_b);
    let k = g.inner_product(x, &w0.k_w, DIM, DIM);
    let k_b = g.constant(&w0.k_b, ch(DIM)); let k = g.addition(k, k_b);
    let v = g.inner_product(x, &w0.v_w, DIM, DIM);
    let v_b = g.constant(&w0.v_b, ch(DIM)); let v = g.addition(v, v_b);
    let q = g.reshape(q, Shape { batch: 1, channels: NUM_HEADS, height: HEAD_DIM, width: SEQ_LEN });
    let k = g.reshape(k, Shape { batch: 1, channels: NUM_HEADS, height: HEAD_DIM, width: SEQ_LEN });
    let v = g.reshape(v, Shape { batch: 1, channels: NUM_HEADS, height: HEAD_DIM, width: SEQ_LEN });
    let q = g.transpose(q, hw); let k = g.transpose(k, hw); let v = g.transpose(v, hw);
    let raw = g.matrix_multiplication(q, k, false, true);
    let scores = g.multiplication(raw, scale);
    let probs = g.soft_max(scores, -1);
    let attn_raw = g.matrix_multiplication(probs, v, false, false);
    let attn = g.transpose(attn_raw, hw);
    let attn = g.reshape(attn, Shape::spatial(DIM, 1, SEQ_LEN));
    let o = g.inner_product(attn, &w0.out_w, DIM, DIM);
    let o_b = g.constant(&w0.out_b, ch(DIM)); let o = g.addition(o, o_b);
    let sa0 = g.addition(o, x);
    let sa0 = layer_norm(&mut g, sa0, &w0.sa_ln_w, &w0.sa_ln_b, DIM, 1e-12);
    let fc1 = g.inner_product(sa0, &w0.ffn1_w, DIM, FFN_DIM);
    let fc1_b = g.constant(&w0.ffn1_b, ch(FFN_DIM)); let fc1 = g.addition(fc1, fc1_b);
    let fc1 = gelu(&mut g, fc1);
    let fc2 = g.inner_product(fc1, &w0.ffn2_w, FFN_DIM, DIM);
    let fc2_b = g.constant(&w0.ffn2_b, ch(DIM)); let fc2 = g.addition(fc2, fc2_b);
    let x1 = g.addition(fc2, sa0);
    let x1 = layer_norm(&mut g, x1, &w0.ffn_ln_w, &w0.ffn_ln_b, DIM, 1e-12);

    // === Layer 1 (POST-norm) ===
    let q = g.inner_product(x1, &w1.q_w, DIM, DIM);
    let q_b = g.constant(&w1.q_b, ch(DIM)); let q = g.addition(q, q_b);
    let k = g.inner_product(x1, &w1.k_w, DIM, DIM);
    let k_b = g.constant(&w1.k_b, ch(DIM)); let k = g.addition(k, k_b);
    let v = g.inner_product(x1, &w1.v_w, DIM, DIM);
    let v_b = g.constant(&w1.v_b, ch(DIM)); let v = g.addition(v, v_b);
    let q = g.reshape(q, Shape { batch: 1, channels: NUM_HEADS, height: HEAD_DIM, width: SEQ_LEN });
    let k = g.reshape(k, Shape { batch: 1, channels: NUM_HEADS, height: HEAD_DIM, width: SEQ_LEN });
    let v = g.reshape(v, Shape { batch: 1, channels: NUM_HEADS, height: HEAD_DIM, width: SEQ_LEN });
    let q = g.transpose(q, hw); let k = g.transpose(k, hw); let v = g.transpose(v, hw);
    let raw = g.matrix_multiplication(q, k, false, true);
    let scores = g.multiplication(raw, scale);
    let probs = g.soft_max(scores, -1);
    let attn_raw = g.matrix_multiplication(probs, v, false, false);
    let attn = g.transpose(attn_raw, hw);
    let attn = g.reshape(attn, Shape::spatial(DIM, 1, SEQ_LEN));
    let o = g.inner_product(attn, &w1.out_w, DIM, DIM);
    let o_b = g.constant(&w1.out_b, ch(DIM)); let o = g.addition(o, o_b);
    let sa1 = g.addition(o, x1);
    let sa1 = layer_norm(&mut g, sa1, &w1.sa_ln_w, &w1.sa_ln_b, DIM, 1e-12);
    let fc1 = g.inner_product(sa1, &w1.ffn1_w, DIM, FFN_DIM);
    let fc1_b = g.constant(&w1.ffn1_b, ch(FFN_DIM)); let fc1 = g.addition(fc1, fc1_b);
    let fc1 = gelu(&mut g, fc1);
    let fc2 = g.inner_product(fc1, &w1.ffn2_w, FFN_DIM, DIM);
    let fc2_b = g.constant(&w1.ffn2_b, ch(DIM)); let fc2 = g.addition(fc2, fc2_b);
    let out = g.addition(fc2, sa1);
    let _ = layer_norm(&mut g, out, &w1.ffn_ln_w, &w1.ffn_ln_b, DIM, 1e-12);

    g.compile(NSQualityOfService::UserInteractive)
}

/// Classifier head: pre_classifier (768→768, ReLU) + classifier (768→2)
fn compile_classifier(w: &ModelWeights) -> Executable {
    let mut g = Graph::new();
    let x = g.placeholder(Shape::spatial(DIM, 1, SEQ_LEN));

    // Take [CLS] token (position 0) — inner_product will process all positions,
    // we just read position 0 from the output
    let pre = g.inner_product(x, &w.pre_cls_w, DIM, DIM);
    let pre_b = g.constant(&w.pre_cls_b, ch(DIM));
    let pre = g.addition(pre, pre_b);
    let pre = g.relu(pre);

    let cls = g.inner_product(pre, &w.cls_w, DIM, NUM_CLASSES);
    let cls_b = g.constant(&w.cls_b, ch(NUM_CLASSES));
    let _ = g.addition(cls, cls_b);

    g.compile(NSQualityOfService::UserInteractive).expect("classifier compile")
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn run_inference(
    layer_exes: &[Executable], cls_exe: &Executable,
    hidden_a: &TensorData, hidden_b: &TensorData, cls_out: &TensorData,
) {
    for (i, exe) in layer_exes.iter().enumerate() {
        let (src, dst) = if i % 2 == 0 { (hidden_a, hidden_b) } else { (hidden_b, hidden_a) };
        exe.run(&[src], &[dst]).unwrap();
    }
    let final_h = if layer_exes.len() % 2 == 0 { hidden_a } else { hidden_b };
    cls_exe.run(&[final_h], &[cls_out]).unwrap();
}

fn classify(cls_out: &TensorData) -> (f32, f32, &'static str) {
    let out = cls_out.as_f32_slice();
    let neg = out[0];
    let pos = out[1 * SEQ_LEN];
    let label = if pos > neg { "POSITIVE" } else { "NEGATIVE" };
    (neg, pos, label)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  DistilBERT Benchmark: Private ANE API vs CoreML");
    eprintln!("═══════════════════════════════════════════════════════");

    // Download weights
    eprint!("  Downloading model... ");
    let api = ApiBuilder::new().with_progress(false).build()?;
    let repo = api.model(REPO_ID.to_string());
    let sf_path = repo.get("model.safetensors")?;
    let sf_bytes = std::fs::read(&sf_path)?;
    eprintln!("ok");

    // Load weights
    eprint!("  Loading weights... ");
    let st = SafeTensors::deserialize(&sf_bytes)?;
    let weights = load_weights(&st);
    eprintln!("ok ({}M params)", 66);

    // Compile ANE graphs — try fused 2-layer graphs first
    eprint!("  Compiling ANE graphs... ");
    let compile_start = Instant::now();

    // Try 2-layer fusion: 6 layers → 3 graphs + 1 classifier = 4 dispatches
    let layer_exes: Vec<Executable> = vec![
        compile_fused_two_layers(&weights.layers[0], &weights.layers[1]).expect("fuse 0-1"),
        compile_fused_two_layers(&weights.layers[2], &weights.layers[3]).expect("fuse 2-3"),
        compile_fused_two_layers(&weights.layers[4], &weights.layers[5]).expect("fuse 4-5"),
    ];
    let fused = true;
    let cls_exe = compile_classifier(&weights);
    let num_dispatches = layer_exes.len() + 1;
    eprintln!("ok ({:.1}s, {} dispatches{})",
        compile_start.elapsed().as_secs_f64(), num_dispatches,
        if fused { ", 2-layer fusion" } else { "" });

    // Allocate IOSurfaces (reused across inferences)
    let mut hidden_a = TensorData::new(Shape::spatial(DIM, 1, SEQ_LEN));
    let mut hidden_b = TensorData::new(Shape::spatial(DIM, 1, SEQ_LEN));
    let cls_out = TensorData::new(Shape::spatial(NUM_CLASSES, 1, SEQ_LEN));

    // Prepare input: embedding lookup (CPU — one-time per inference)
    // For benchmark: use zeros as token IDs (content doesn't affect latency)
    let token_ids: Vec<u32> = vec![0; SEQ_LEN];
    let embed = |hidden: &TensorData| {
        let mut surf = hidden.as_f32_slice_mut();
        for (pos, &tok) in token_ids.iter().enumerate() {
            let t = tok as usize;
            for c in 0..DIM {
                surf[c * SEQ_LEN + pos] =
                    weights.word_emb[t * DIM + c] + weights.pos_emb[pos * DIM + c];
            }
        }
    };

    // Apply embedding LayerNorm on CPU (one-time, not timed in hot loop)
    let embed_and_norm = |hidden: &TensorData| {
        embed(hidden);
        // LayerNorm on CPU for embeddings
        let mut surf = hidden.as_f32_slice_mut();
        for s in 0..SEQ_LEN {
            let mut mean = 0f32;
            for c in 0..DIM { mean += surf[c * SEQ_LEN + s]; }
            mean /= DIM as f32;
            let mut var = 0f32;
            for c in 0..DIM {
                let d = surf[c * SEQ_LEN + s] - mean;
                var += d * d;
            }
            var /= DIM as f32;
            let rstd = 1.0 / (var + 1e-12_f32).sqrt();
            for c in 0..DIM {
                surf[c * SEQ_LEN + s] = (surf[c * SEQ_LEN + s] - mean) * rstd
                    * weights.emb_ln_w[c] + weights.emb_ln_b[c];
            }
        }
    };

    // Download tokenizer
    let tok_path = repo.get("tokenizer.json").or_else(|_| {
        api.model("distilbert-base-uncased".to_string()).get("tokenizer.json")
    })?;
    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| format!("tokenizer: {e}"))?;

    // Helper: tokenize and embed a sentence
    let embed_sentence = |text: &str, hidden: &TensorData| {
        let enc = tokenizer.encode(text, true).expect("encode");
        let ids = enc.get_ids();
        let len = ids.len().min(SEQ_LEN);
        let mut surf = hidden.as_f32_slice_mut();
        surf.fill(0.0);
        for pos in 0..SEQ_LEN {
            let tok = if pos < len { ids[pos] as usize } else { 0 }; // pad with 0
            for c in 0..DIM {
                surf[c * SEQ_LEN + pos] =
                    weights.word_emb[tok * DIM + c] + weights.pos_emb[pos * DIM + c];
            }
        }
        // Embedding LayerNorm
        for s in 0..SEQ_LEN {
            let mut mean = 0f32;
            for c in 0..DIM { mean += surf[c * SEQ_LEN + s]; }
            mean /= DIM as f32;
            let mut var = 0f32;
            for c in 0..DIM { let d = surf[c * SEQ_LEN + s] - mean; var += d * d; }
            var /= DIM as f32;
            let rstd = 1.0 / (var + 1e-12_f32).sqrt();
            for c in 0..DIM {
                surf[c * SEQ_LEN + s] = (surf[c * SEQ_LEN + s] - mean) * rstd
                    * weights.emb_ln_w[c] + weights.emb_ln_b[c];
            }
        }
    };

    // === Quick sanity check ===
    embed_sentence("I love this movie!", &hidden_a);
    run_inference(&layer_exes, &cls_exe, &hidden_a, &hidden_b, &cls_out);
    let (neg, pos, label) = classify(&cls_out);
    eprintln!("  Sanity: \"I love this movie!\" → {label} (neg={neg:.2}, pos={pos:.2})");

    // === Benchmark: ANE inference latency ===
    eprintln!("\n  Benchmarking (1000 iterations)...");

    // Warmup with a real sentence
    embed_sentence("This is a test sentence for warmup.", &hidden_a);
    for _ in 0..50 {
        run_inference(&layer_exes, &cls_exe, &hidden_a, &hidden_b, &cls_out);
    }

    // Timed runs (embedding is pre-computed, we time only the ANE forward pass)
    let mut times = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let start = Instant::now();
        run_inference(&layer_exes, &cls_exe, &hidden_a, &hidden_b, &cls_out);
        times.push(start.elapsed().as_micros() as f64 / 1000.0);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let p5 = times[times.len() / 20];
    let p95 = times[times.len() * 19 / 20];

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  Results: DistilBERT seq=128 (ANE private API)");
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  Mean:   {mean:.3} ms");
    eprintln!("  Median: {median:.3} ms");
    eprintln!("  P5:     {p5:.3} ms");
    eprintln!("  P95:    {p95:.3} ms");
    eprintln!("  Min:    {:.3} ms", times[0]);
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  CoreML baseline (M1 Max):  5.858 ms");
    eprintln!("  Apple published (iPhone):  3.470 ms");
    if median < 5.858 {
        eprintln!("  >>> FASTER than CoreML by {:.1}% <<<", (5.858 - median) / 5.858 * 100.0);
    } else {
        eprintln!("  Slower than CoreML by {:.1}%", (median - 5.858) / 5.858 * 100.0);
    }
    eprintln!("═══════════════════════════════════════════════════════");

    Ok(())
}
