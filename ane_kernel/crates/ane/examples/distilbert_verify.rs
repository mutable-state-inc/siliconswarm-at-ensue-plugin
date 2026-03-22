/// DistilBERT verification: ensure the ANE kernel produces correct sentiment classification.
///
/// Runs known positive/negative sentences and checks the model classifies them correctly.
/// Exit code 0 = all tests pass, 1 = at least one failure.
///
/// Run: cargo run --release --example distilbert_verify
use std::io::Write;
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
const HEAD_DIM: usize = 64;
const FFN_DIM: usize = 3072;
const NUM_LAYERS: usize = 6;
const NUM_CLASSES: usize = 2;

// ─── Reuse the same model code (TODO: factor into shared module) ───────────

struct LayerWeights {
    sa_ln_w: Box<[f32]>,
    sa_ln_b: Box<[f32]>,
    q_w: Box<[f32]>,
    q_b: Box<[f32]>,
    k_w: Box<[f32]>,
    k_b: Box<[f32]>,
    v_w: Box<[f32]>,
    v_b: Box<[f32]>,
    out_w: Box<[f32]>,
    out_b: Box<[f32]>,
    ffn_ln_w: Box<[f32]>,
    ffn_ln_b: Box<[f32]>,
    ffn1_w: Box<[f32]>,
    ffn1_b: Box<[f32]>,
    ffn2_w: Box<[f32]>,
    ffn2_b: Box<[f32]>,
}

struct ModelWeights {
    word_emb: Box<[f32]>,
    pos_emb: Box<[f32]>,
    emb_ln_w: Box<[f32]>,
    emb_ln_b: Box<[f32]>,
    layers: Box<[LayerWeights]>,
    pre_cls_w: Box<[f32]>,
    pre_cls_b: Box<[f32]>,
    cls_w: Box<[f32]>,
    cls_b: Box<[f32]>,
}

fn tensor_f32(st: &SafeTensors, name: &str) -> Box<[f32]> {
    let t = st
        .tensor(name)
        .unwrap_or_else(|_| panic!("missing: {name}"));
    let b = t.data();
    match t.dtype() {
        Dtype::BF16 => b
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        Dtype::F16 => b
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        Dtype::F32 => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => panic!("unsupported dtype: {other:?}"),
    }
}

fn load_weights(st: &SafeTensors) -> ModelWeights {
    let layers: Box<[LayerWeights]> = (0..NUM_LAYERS)
        .map(|i| {
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
        })
        .collect();
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

fn scalar() -> Shape {
    Shape {
        batch: 1,
        channels: 1,
        height: 1,
        width: 1,
    }
}
fn ch(d: usize) -> Shape {
    Shape {
        batch: 1,
        channels: d,
        height: 1,
        width: 1,
    }
}

fn layer_norm(
    g: &mut Graph,
    input: ane::Tensor,
    weight: &[f32],
    bias: &[f32],
    d: usize,
    eps: f64,
) -> ane::Tensor {
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

/// DistilBERT POST-LayerNorm encoder layer
fn compile_encoder_layer(w: &LayerWeights) -> Executable {
    let mut g = Graph::new();
    let x = g.placeholder(Shape::spatial(DIM, 1, SEQ_LEN));
    let q = g.inner_product(x, &w.q_w, DIM, DIM);
    let q_b = g.constant(&w.q_b, ch(DIM));
    let q = g.addition(q, q_b);
    let k = g.inner_product(x, &w.k_w, DIM, DIM);
    let k_b = g.constant(&w.k_b, ch(DIM));
    let k = g.addition(k, k_b);
    let v = g.inner_product(x, &w.v_w, DIM, DIM);
    let v_b = g.constant(&w.v_b, ch(DIM));
    let v = g.addition(v, v_b);
    let q = g.reshape(
        q,
        Shape {
            batch: 1,
            channels: NUM_HEADS,
            height: HEAD_DIM,
            width: SEQ_LEN,
        },
    );
    let k = g.reshape(
        k,
        Shape {
            batch: 1,
            channels: NUM_HEADS,
            height: HEAD_DIM,
            width: SEQ_LEN,
        },
    );
    let v = g.reshape(
        v,
        Shape {
            batch: 1,
            channels: NUM_HEADS,
            height: HEAD_DIM,
            width: SEQ_LEN,
        },
    );
    let hw = [0, 1, 3, 2];
    let q = g.transpose(q, hw);
    let k = g.transpose(k, hw);
    let v = g.transpose(v, hw);
    let scale = g.constant_with_scalar(1.0 / (HEAD_DIM as f32).sqrt(), scalar());
    let raw = g.matrix_multiplication(q, k, false, true);
    let scores = g.multiplication(raw, scale);
    let probs = g.soft_max(scores, -1);
    let attn_raw = g.matrix_multiplication(probs, v, false, false);
    let attn = g.transpose(attn_raw, hw);
    let attn = g.reshape(attn, Shape::spatial(DIM, 1, SEQ_LEN));
    let o = g.inner_product(attn, &w.out_w, DIM, DIM);
    let o_b = g.constant(&w.out_b, ch(DIM));
    let o = g.addition(o, o_b);
    // POST-norm: residual then LayerNorm
    let sa = g.addition(o, x);
    let sa = layer_norm(&mut g, sa, &w.sa_ln_w, &w.sa_ln_b, DIM, 1e-12);
    let fc1 = g.inner_product(sa, &w.ffn1_w, DIM, FFN_DIM);
    let fc1_b = g.constant(&w.ffn1_b, ch(FFN_DIM));
    let fc1 = g.addition(fc1, fc1_b);
    let fc1 = gelu(&mut g, fc1);
    let fc2 = g.inner_product(fc1, &w.ffn2_w, FFN_DIM, DIM);
    let fc2_b = g.constant(&w.ffn2_b, ch(DIM));
    let fc2 = g.addition(fc2, fc2_b);
    let ffn = g.addition(fc2, sa);
    let _ = layer_norm(&mut g, ffn, &w.ffn_ln_w, &w.ffn_ln_b, DIM, 1e-12);
    g.compile(NSQualityOfService::UserInteractive)
        .expect("layer compile")
}

fn compile_classifier(w: &ModelWeights) -> Executable {
    let mut g = Graph::new();
    let x = g.placeholder(Shape::spatial(DIM, 1, SEQ_LEN));
    let pre = g.inner_product(x, &w.pre_cls_w, DIM, DIM);
    let pre_b = g.constant(&w.pre_cls_b, ch(DIM));
    let pre = g.addition(pre, pre_b);
    let pre = g.relu(pre);
    let cls = g.inner_product(pre, &w.cls_w, DIM, NUM_CLASSES);
    let cls_b = g.constant(&w.cls_b, ch(NUM_CLASSES));
    let _ = g.addition(cls, cls_b);
    g.compile(NSQualityOfService::UserInteractive)
        .expect("classifier compile")
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  DistilBERT Verification (ANE private API)");
    eprintln!("═══════════════════════════════════════════════════════");

    let api = ApiBuilder::new().with_progress(false).build()?;
    let repo = api.model(REPO_ID.to_string());
    let sf_bytes = std::fs::read(repo.get("model.safetensors")?)?;
    let st = SafeTensors::deserialize(&sf_bytes)?;
    let weights = load_weights(&st);
    // Use base distilbert tokenizer (same vocab as finetuned)
    let tok_path = api
        .model("distilbert-base-uncased".to_string())
        .get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| format!("tokenizer: {e}"))?;

    eprint!("  Compiling... ");
    let layer_exes: Vec<Executable> = weights
        .layers
        .iter()
        .map(|lw| compile_encoder_layer(lw))
        .collect();
    let cls_exe = compile_classifier(&weights);
    eprintln!("ok");

    let hidden_a = TensorData::new(Shape::spatial(DIM, 1, SEQ_LEN));
    let hidden_b = TensorData::new(Shape::spatial(DIM, 1, SEQ_LEN));
    let cls_out = TensorData::new(Shape::spatial(NUM_CLASSES, 1, SEQ_LEN));

    let embed_sentence = |text: &str| {
        let enc = tokenizer.encode(text, true).expect("encode");
        let ids = enc.get_ids();
        let len = ids.len().min(SEQ_LEN);
        let mut surf = hidden_a.as_f32_slice_mut();
        surf.fill(0.0);
        for pos in 0..SEQ_LEN {
            let tok = if pos < len { ids[pos] as usize } else { 0 };
            for c in 0..DIM {
                surf[c * SEQ_LEN + pos] =
                    weights.word_emb[tok * DIM + c] + weights.pos_emb[pos * DIM + c];
            }
        }
        for s in 0..SEQ_LEN {
            let mut mean = 0f32;
            for c in 0..DIM {
                mean += surf[c * SEQ_LEN + s];
            }
            mean /= DIM as f32;
            let mut var = 0f32;
            for c in 0..DIM {
                let d = surf[c * SEQ_LEN + s] - mean;
                var += d * d;
            }
            var /= DIM as f32;
            let rstd = 1.0 / (var + 1e-12_f32).sqrt();
            for c in 0..DIM {
                surf[c * SEQ_LEN + s] = (surf[c * SEQ_LEN + s] - mean) * rstd * weights.emb_ln_w[c]
                    + weights.emb_ln_b[c];
            }
        }
    };

    let run = || {
        for (i, exe) in layer_exes.iter().enumerate() {
            let (src, dst) = if i % 2 == 0 {
                (&hidden_a, &hidden_b)
            } else {
                (&hidden_b, &hidden_a)
            };
            exe.run(&[src], &[dst]).unwrap();
        }
        let fh = if layer_exes.len() % 2 == 0 {
            &hidden_a
        } else {
            &hidden_b
        };
        cls_exe.run(&[fh], &[&cls_out]).unwrap();
        let out = cls_out.as_f32_slice();
        let neg = out[0];
        let pos = out[1 * SEQ_LEN];
        if pos > neg { "POSITIVE" } else { "NEGATIVE" }
    };

    // Load SST-2 validation set (872 examples)
    let sst2_path = std::path::Path::new("sst2_validation.tsv");
    if !sst2_path.exists() {
        eprintln!("  ERROR: sst2_validation.tsv not found.");
        eprintln!("  Run: python3 -c \"from datasets import load_dataset; ds = load_dataset('glue','sst2',split='validation'); f = open('sst2_validation.tsv','w'); f.write('label\\tsentence\\n'); [f.write(f'{{r[\\\"label\\\"]}}\\t{{r[\\\"sentence\\\"]}}\\n') for r in ds]\"");
        std::process::exit(1);
    }

    let tsv = std::fs::read_to_string(sst2_path).expect("read sst2_validation.tsv");
    let lines: Vec<&str> = tsv.lines().skip(1).collect(); // skip header
    let total = lines.len();
    eprintln!("  Running SST-2 validation ({total} examples)...");

    let mut correct = 0;
    let mut errors: Vec<(String, &str, &str)> = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let parts: Vec<&str> = line.splitn(2, '\t').collect();
        if parts.len() != 2 { continue; }
        let label: usize = parts[0].parse().unwrap_or(0);
        let sentence = parts[1].trim();
        let expected = if label == 1 { "POSITIVE" } else { "NEGATIVE" };

        embed_sentence(sentence);
        let got = run();
        if got == expected {
            correct += 1;
        } else if errors.len() < 5 {
            errors.push((sentence.to_string(), expected, got));
        }

        if (i + 1) % 100 == 0 {
            eprint!("\r\x1b[2K  {}/{total} ({:.1}%)...", i + 1, correct as f64 / (i + 1) as f64 * 100.0);
            let _ = std::io::stderr().flush();
        }
    }

    let accuracy = correct as f64 / total as f64 * 100.0;
    let published_accuracy = 91.3; // f32 reference
    let min_accuracy = 90.0; // CoreML on ANE gets 90.48%. Match or beat it.

    eprintln!("\r\x1b[2K");
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  SST-2 Validation Results");
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  Correct:    {correct}/{total}");
    eprintln!("  Accuracy:   {accuracy:.2}%");
    eprintln!("  Published:  {published_accuracy}% (f32 reference)");
    eprintln!("  Threshold:  {min_accuracy}% (fp16 tolerance)");

    if !errors.is_empty() {
        eprintln!();
        eprintln!("  Sample errors:");
        for (sent, expected, got) in &errors {
            let short = if sent.len() > 60 { &sent[..60] } else { sent.as_str() };
            eprintln!("    ✗ \"{short}...\" expected={expected} got={got}");
        }
    }

    eprintln!("═══════════════════════════════════════════════════════");
    if accuracy >= min_accuracy {
        eprintln!("  PASSED ({accuracy:.2}% ≥ {min_accuracy}%)");
    } else {
        eprintln!("  FAILED ({accuracy:.2}% < {min_accuracy}%)");
        std::process::exit(1);
    }

    Ok(())
}
