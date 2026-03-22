/// GPT-2 Large (774M) inference on Apple Neural Engine via private API.
/// Single-file implementation: download → compile → prefill → decode → benchmark.
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use ane::{Executable, Graph, NSQualityOfService, Shape, TensorData};
use half::{bf16, f16};
use hf_hub::api::sync::ApiBuilder;
use rand::{Rng, RngExt};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use tokenizers::Tokenizer;

// ─── Config ────────────────────────────────────────────────────────────────

const REPO_ID: &str = "vicgalle/gpt2-open-instruct-v1"; // GPT-2 instruct (124M)
const MAX_NEW_TOKENS: usize = 60;
const MAX_SEQ: usize = 128;
const SPATIAL: usize = 64;
const TEMPERATURE: f32 = 0.8;
const TOP_P: f32 = 0.95;
const REPETITION_PENALTY: f32 = 1.2;

#[derive(Debug, Clone, Deserialize)]
struct Config {
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    vocab_size: usize,
    #[allow(dead_code)]
    n_positions: usize,
    #[serde(default = "default_eps")]
    layer_norm_epsilon: f64,
}
fn default_eps() -> f64 {
    1e-5
}
impl Config {
    fn head_size(&self) -> usize {
        self.n_embd / self.n_head
    }
}

// ─── Weights ───────────────────────────────────────────────────────────────

struct LayerWeights {
    ln1_weight: Box<[f32]>,
    ln1_bias: Box<[f32]>,
    qkv_weight: Box<[f32]>,
    qkv_bias: Box<[f32]>,
    attn_proj_weight: Box<[f32]>,
    attn_proj_bias: Box<[f32]>,
    ln2_weight: Box<[f32]>,
    ln2_bias: Box<[f32]>,
    fc_weight: Box<[f32]>,
    fc_bias: Box<[f32]>,
    fc_proj_weight: Box<[f32]>,
    fc_proj_bias: Box<[f32]>,
}

struct ModelWeights {
    wte: Box<[f32]>,
    wpe: Box<[f32]>,
    layers: Box<[LayerWeights]>,
    ln_f_weight: Box<[f32]>,
    ln_f_bias: Box<[f32]>,
}

fn tensor_f32(st: &SafeTensors, name: &str) -> Box<[f32]> {
    // Try with and without "transformer." prefix
    let t = st
        .tensor(name)
        .or_else(|_| st.tensor(&format!("transformer.{name}")))
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

fn tensor_f32_t(st: &SafeTensors, name: &str, rows: usize, cols: usize) -> Box<[f32]> {
    let raw = tensor_f32(st, name);
    assert_eq!(raw.len(), rows * cols);
    let mut t = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = raw[r * cols + c];
        }
    }
    t.into()
}

struct Downloaded {
    config: Config,
    tokenizer_path: PathBuf,
    safetensors_bytes: Vec<u8>,
}

fn download(repo_id: &str) -> Result<Downloaded, Box<dyn std::error::Error>> {
    let api = ApiBuilder::new().with_progress(true).build()?;
    let repo = api.model(repo_id.to_string());
    eprint_status("Downloading config.json");
    let config: Config = serde_json::from_reader(std::fs::File::open(repo.get("config.json")?)?)?;
    eprint_status("Downloading tokenizer");
    let tokenizer_path = repo.get("tokenizer.json").or_else(|_| {
        // Fallback: use GPT-2 tokenizer (same vocab for Cerebras-GPT etc.)
        let gpt2 = api.model("openai-community/gpt2".to_string());
        gpt2.get("tokenizer.json")
    })?;
    eprint_status("Downloading model.safetensors");
    let sb = std::fs::read(repo.get("model.safetensors")?)?;
    eprint_done("Downloaded model files");
    Ok(Downloaded {
        config,
        tokenizer_path,
        safetensors_bytes: sb,
    })
}

fn load_weights(st: &SafeTensors, cfg: &Config) -> ModelWeights {
    let d = cfg.n_embd;
    let layers: Box<[LayerWeights]> = (0..cfg.n_layer)
        .map(|i| {
            let p = format!("h.{i}");
            LayerWeights {
                ln1_weight: tensor_f32(st, &format!("{p}.ln_1.weight")),
                ln1_bias: tensor_f32(st, &format!("{p}.ln_1.bias")),
                qkv_weight: tensor_f32_t(st, &format!("{p}.attn.c_attn.weight"), d, 3 * d),
                qkv_bias: tensor_f32(st, &format!("{p}.attn.c_attn.bias")),
                attn_proj_weight: tensor_f32_t(st, &format!("{p}.attn.c_proj.weight"), d, d),
                attn_proj_bias: tensor_f32(st, &format!("{p}.attn.c_proj.bias")),
                ln2_weight: tensor_f32(st, &format!("{p}.ln_2.weight")),
                ln2_bias: tensor_f32(st, &format!("{p}.ln_2.bias")),
                fc_weight: tensor_f32_t(st, &format!("{p}.mlp.c_fc.weight"), d, 4 * d),
                fc_bias: tensor_f32(st, &format!("{p}.mlp.c_fc.bias")),
                fc_proj_weight: tensor_f32_t(st, &format!("{p}.mlp.c_proj.weight"), 4 * d, d),
                fc_proj_bias: tensor_f32(st, &format!("{p}.mlp.c_proj.bias")),
            }
        })
        .collect();
    ModelWeights {
        wte: tensor_f32(st, "wte.weight"),
        wpe: tensor_f32(st, "wpe.weight"),
        layers,
        ln_f_weight: tensor_f32(st, "ln_f.weight"),
        ln_f_bias: tensor_f32(st, "ln_f.bias"),
    }
}

// ─── ANE Graph Compilation ─────────────────────────────────────────────────

fn scalar() -> Shape {
    Shape {
        batch: 1,
        channels: 1,
        height: 1,
        width: 1,
    }
}

fn layer_norm(
    g: &mut Graph,
    input: ane::Tensor,
    gamma: &[f32],
    beta: &[f32],
    d: usize,
    eps: f64,
) -> ane::Tensor {
    let inv_d = g.constant_with_scalar(1.0 / d as f32, scalar());
    let eps_c = g.constant_with_scalar(eps as f32, scalar());
    let neg_half = g.constant_with_scalar(-0.5, scalar());
    let neg_one = g.constant_with_scalar(-1.0, scalar());
    let gamma_c = g.constant(
        gamma,
        Shape {
            batch: 1,
            channels: d,
            height: 1,
            width: 1,
        },
    );
    let beta_c = g.constant(
        beta,
        Shape {
            batch: 1,
            channels: d,
            height: 1,
            width: 1,
        },
    );
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
    let scaled = g.multiplication(normed, gamma_c);
    g.addition(scaled, beta_c)
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

fn causal_mask(len: usize) -> Box<[f32]> {
    (0..len * len)
        .map(|i| {
            if (i % len) <= (i / len) {
                0.0
            } else {
                -65504.0
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn attention(
    g: &mut Graph,
    normed: ane::Tensor,
    w: &LayerWeights,
    cfg: &Config,
    q_seq: usize,
    k_seq: usize,
    k_cache: Option<ane::Tensor>,
    v_cache: Option<ane::Tensor>,
    mask: Option<ane::Tensor>,
) -> (ane::Tensor, ane::Tensor, ane::Tensor) {
    let d = cfg.n_embd;
    let nh = cfg.n_head;
    let hs = cfg.head_size();
    let qkv = g.inner_product(normed, &w.qkv_weight, d, 3 * d);
    let qkv_b = g.constant(
        &w.qkv_bias,
        Shape {
            batch: 1,
            channels: 3 * d,
            height: 1,
            width: 1,
        },
    );
    let qkv = g.addition(qkv, qkv_b);
    let q_flat = g.slice(qkv, [0, 0, 0, 0], [1, d, 1, q_seq]);
    let k_new = g.slice(qkv, [0, d, 0, 0], [1, d, 1, q_seq]);
    let v_new = g.slice(qkv, [0, 2 * d, 0, 0], [1, d, 1, q_seq]);
    let hw = [0, 1, 3, 2];
    let q_r = g.reshape(
        q_flat,
        Shape {
            batch: 1,
            channels: nh,
            height: hs,
            width: q_seq,
        },
    );
    let q = g.transpose(q_r, hw);
    let (k, v, aks) = match (k_cache, v_cache) {
        (Some(kc), Some(vc)) => {
            let kr = g.reshape(
                kc,
                Shape {
                    batch: 1,
                    channels: nh,
                    height: hs,
                    width: k_seq,
                },
            );
            let k = g.transpose(kr, hw);
            let vr = g.reshape(
                vc,
                Shape {
                    batch: 1,
                    channels: nh,
                    height: hs,
                    width: k_seq,
                },
            );
            let v = g.transpose(vr, hw);
            (k, v, k_seq)
        }
        _ => {
            let kr = g.reshape(
                k_new,
                Shape {
                    batch: 1,
                    channels: nh,
                    height: hs,
                    width: q_seq,
                },
            );
            let k = g.transpose(kr, hw);
            let vr = g.reshape(
                v_new,
                Shape {
                    batch: 1,
                    channels: nh,
                    height: hs,
                    width: q_seq,
                },
            );
            let v = g.transpose(vr, hw);
            (k, v, q_seq)
        }
    };
    let scale = g.constant_with_scalar(1.0 / (hs as f32).sqrt(), scalar());
    let raw_scores = g.matrix_multiplication(q, k, false, true);
    let scores = g.multiplication(raw_scores, scale);
    let scores = match mask {
        Some(m) => g.addition(scores, m),
        None => {
            let m = g.constant(
                &causal_mask(q_seq),
                Shape {
                    batch: 1,
                    channels: 1,
                    height: q_seq,
                    width: aks,
                },
            );
            g.addition(scores, m)
        }
    };
    let probs = g.soft_max(scores, -1);
    let attn_raw = g.matrix_multiplication(probs, v, false, false);
    let attn = g.transpose(attn_raw, hw);
    let attn = g.reshape(attn, Shape::spatial(d, 1, q_seq));
    let proj = g.inner_product(attn, &w.attn_proj_weight, d, d);
    let proj_b = g.constant(
        &w.attn_proj_bias,
        Shape {
            batch: 1,
            channels: d,
            height: 1,
            width: 1,
        },
    );
    (g.addition(proj, proj_b), k_new, v_new)
}

fn ffn(g: &mut Graph, input: ane::Tensor, w: &LayerWeights, cfg: &Config) -> ane::Tensor {
    let d = cfg.n_embd;
    let normed = layer_norm(
        g,
        input,
        &w.ln2_weight,
        &w.ln2_bias,
        d,
        cfg.layer_norm_epsilon,
    );
    let h = g.inner_product(normed, &w.fc_weight, d, 4 * d);
    let fc_b = g.constant(
        &w.fc_bias,
        Shape {
            batch: 1,
            channels: 4 * d,
            height: 1,
            width: 1,
        },
    );
    let h_biased = g.addition(h, fc_b);
    let h = gelu(g, h_biased);
    let proj = g.inner_product(h, &w.fc_proj_weight, 4 * d, d);
    let proj_b = g.constant(
        &w.fc_proj_bias,
        Shape {
            batch: 1,
            channels: d,
            height: 1,
            width: 1,
        },
    );
    let ffn_out = g.addition(proj, proj_b);
    g.addition(ffn_out, input)
}

fn compile_decode_attn(w: &LayerWeights, cfg: &Config) -> Executable {
    let d = cfg.n_embd;
    let mut g = Graph::new();
    let x = g.placeholder(Shape::spatial(d, 1, SPATIAL));
    let kc = g.placeholder(Shape::spatial(d, 1, MAX_SEQ));
    let vc = g.placeholder(Shape::spatial(d, 1, MAX_SEQ));
    let mask = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: SPATIAL,
        width: MAX_SEQ,
    });
    let normed = layer_norm(
        &mut g,
        x,
        &w.ln1_weight,
        &w.ln1_bias,
        d,
        cfg.layer_norm_epsilon,
    );
    let (o, k_new, v_new) = attention(
        &mut g,
        normed,
        w,
        cfg,
        SPATIAL,
        MAX_SEQ,
        Some(kc),
        Some(vc),
        Some(mask),
    );
    let res = g.addition(o, x);
    let _ = g.concat(&[res, k_new, v_new], 1);
    g.compile(NSQualityOfService::Default)
        .expect("decode attn compile")
}

fn compile_decode_ffn(w: &LayerWeights, cfg: &Config) -> Executable {
    let d = cfg.n_embd;
    let mut g = Graph::new();
    let inp = g.placeholder(Shape::spatial(d, 1, SPATIAL));
    let _ = ffn(&mut g, inp, w, cfg);
    g.compile(NSQualityOfService::Default)
        .expect("decode ffn compile")
}

fn compile_lm_head(wte: &[f32], ln_w: &[f32], ln_b: &[f32], cfg: &Config) -> Executable {
    let d = cfg.n_embd;
    let v = cfg.vocab_size;
    let mut g = Graph::new();
    let inp = g.placeholder(Shape::spatial(d, 1, SPATIAL));
    let normed = layer_norm(&mut g, inp, ln_w, ln_b, d, cfg.layer_norm_epsilon);
    let max_chunk = 16 * 1024 * 1024 / (d * 2);
    let n = v.div_ceil(max_chunk);
    let cs = v / n;
    let chunks: Vec<_> = (0..n)
        .map(|i| {
            let s = i * cs;
            let e = if i == n - 1 { v } else { (i + 1) * cs };
            g.inner_product(normed, &wte[s * d..e * d], d, e - s)
        })
        .collect();
    let _ = g.concat(&chunks, 1);
    g.compile(NSQualityOfService::Default)
        .expect("lm_head compile")
}

// ─── KV Cache ──────────────────────────────────────────────────────────────

struct KvCache {
    keys: Box<[TensorData]>,
    values: Box<[TensorData]>,
    dim: usize,
}

impl KvCache {
    fn new(n_layer: usize, dim: usize) -> Self {
        let shape = Shape::spatial(dim, 1, MAX_SEQ);
        Self {
            keys: (0..n_layer).map(|_| TensorData::new(shape)).collect(),
            values: (0..n_layer).map(|_| TensorData::new(shape)).collect(),
            dim,
        }
    }
    fn write_decode(&self, layer: usize, attn: &[f32], pos: usize) {
        let mut ks = self.keys[layer].as_f32_slice_mut();
        let mut vs = self.values[layer].as_f32_slice_mut();
        for c in 0..self.dim {
            ks[c * MAX_SEQ + pos] = attn[(self.dim + c) * SPATIAL];
            vs[c * MAX_SEQ + pos] = attn[(2 * self.dim + c) * SPATIAL];
        }
    }
}

// ─── Sampling ──────────────────────────────────────────────────────────────

fn sample(logits: &[f32], history: &[u32], rng: &mut impl Rng) -> u32 {
    let mut l: Box<[f32]> = logits.into();
    for &t in history {
        let i = t as usize;
        if i < l.len() {
            if l[i] > 0.0 {
                l[i] /= REPETITION_PENALTY;
            } else {
                l[i] *= REPETITION_PENALTY;
            }
        }
    }
    if TEMPERATURE <= 0.0 {
        return l
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
    }
    let max = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = l
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, ((v - max) / TEMPERATURE).exp()))
        .collect();
    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let total: f32 = probs.iter().map(|p| p.1).sum();
    let cutoff = TOP_P * total;
    let mut cum = 0f32;
    let mut cands = Vec::new();
    for &(idx, p) in &probs {
        cum += p;
        cands.push((idx, p));
        if cum >= cutoff {
            break;
        }
    }
    let ct: f32 = cands.iter().map(|p| p.1).sum();
    let thresh = rng.random::<f32>() * ct;
    let mut acc = 0f32;
    for &(idx, p) in &cands {
        acc += p;
        if acc >= thresh {
            return idx as u32;
        }
    }
    cands.last().unwrap().0 as u32
}

// ─── Status ────────────────────────────────────────────────────────────────

fn eprint_status(msg: &str) {
    eprint!("\r\x1b[2K\x1b[1;36m⠋\x1b[0m {msg}");
    let _ = io::stderr().flush();
}
fn eprint_done(msg: &str) {
    eprintln!("\r\x1b[2K\x1b[1;32m✓\x1b[0m {msg}");
}

// ─── Decode one token ──────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn decode_one(
    dc_hidden: &mut TensorData,
    dc_ffn_out: &mut TensorData,
    dc_attn_out: &TensorData,
    dc_mask: &TensorData,
    lm_out: &TensorData,
    decode_attn: &[Executable],
    decode_ffn: &[Executable],
    lm_head: &Executable,
    kv: &KvCache,
    weights: &ModelWeights,
    logits_buf: &mut [f32],
    token: u32,
    position: usize,
    d: usize,
    n_layer: usize,
    vocab_size: usize,
) {
    // Embed
    {
        let mut s = dc_hidden.as_f32_slice_mut();
        let t = token as usize;
        for c in 0..d {
            s[c * SPATIAL] = weights.wte[t * d + c] + weights.wpe[position * d + c];
        }
    }
    {
        dc_mask.as_f32_slice_mut()[position] = 0.0;
    }

    // Layers
    for i in 0..n_layer {
        decode_attn[i]
            .run(
                &[dc_hidden, &kv.keys[i], &kv.values[i], dc_mask],
                &[dc_attn_out],
            )
            .unwrap();
        {
            let a = dc_attn_out.as_f32_slice();
            kv.write_decode(i, &a, position);
            dc_hidden
                .as_f32_slice_mut()
                .copy_from_slice(&a[..d * SPATIAL]);
        }
        decode_ffn[i].run(&[dc_hidden], &[dc_ffn_out]).unwrap();
        std::mem::swap(dc_hidden, dc_ffn_out);
    }

    // LM head
    lm_head.run(&[dc_hidden], &[lm_out]).unwrap();
    let o = lm_out.as_f32_slice();
    for v in 0..vocab_size {
        logits_buf[v] = o[v * SPATIAL];
    }
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();

    // Download & load
    let dl = download(REPO_ID)?;
    let cfg = dl.config;
    let d = cfg.n_embd;
    eprintln!(
        "  Model: {} ({} layers, dim={}, {} heads)",
        REPO_ID, cfg.n_layer, d, cfg.n_head
    );

    let tokenizer =
        Tokenizer::from_file(&dl.tokenizer_path).map_err(|e| format!("tokenizer: {e}"))?;

    eprint_status("Loading weights");
    let st = SafeTensors::deserialize(&dl.safetensors_bytes)?;
    let weights = load_weights(&st, &cfg);

    // Compile ANE executables (decode only — prefill reuses decode path token-by-token)
    let mut decode_attn = Vec::new();
    let mut decode_ffn = Vec::new();
    for (i, lw) in weights.layers.iter().enumerate() {
        eprint_status(&format!("Compiling layer {}/{}", i + 1, cfg.n_layer));
        decode_attn.push(compile_decode_attn(lw, &cfg));
        decode_ffn.push(compile_decode_ffn(lw, &cfg));
    }
    eprint_status("Compiling LM head");
    let lm_head = compile_lm_head(&weights.wte, &weights.ln_f_weight, &weights.ln_f_bias, &cfg);
    eprint_done(&format!(
        "Compiled ANE model ({:.1}s)",
        start.elapsed().as_secs_f64()
    ));

    // Allocate IOSurfaces
    let mut dc_hidden = TensorData::new(Shape::spatial(d, 1, SPATIAL));
    let dc_attn_out = TensorData::new(Shape::spatial(3 * d, 1, SPATIAL));
    let mut dc_ffn_out = TensorData::new(Shape::spatial(d, 1, SPATIAL));
    let dc_mask = TensorData::new(Shape {
        batch: 1,
        channels: 1,
        height: SPATIAL,
        width: MAX_SEQ,
    });
    let lm_out = TensorData::new(Shape::spatial(cfg.vocab_size, 1, SPATIAL));
    let mut logits_buf = vec![0f32; cfg.vocab_size];
    let kv = KvCache::new(cfg.n_layer, d);
    let mut rng = rand::rng();

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  GPT-2 Instruct (124M) — Apple Neural Engine");
    eprintln!("  Type an instruction, press Enter. Ctrl-C to quit.");
    eprintln!("═══════════════════════════════════════════════════════");

    loop {
        // Reset KV cache and mask for each prompt
        for i in 0..cfg.n_layer {
            kv.keys[i].as_f32_slice_mut().fill(0.0);
            kv.values[i].as_f32_slice_mut().fill(0.0);
        }
        dc_mask.as_f32_slice_mut().fill(-65504.0);

        eprintln!();
        eprint!("\x1b[1;32m> \x1b[0m");
        io::stderr().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        } // EOF
        let prompt = input.trim();
        if prompt.is_empty() {
            continue;
        }

        // Alpaca instruct template
        let formatted = format!(
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n"
        );
        let enc = tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| format!("encode: {e}"))?;
        let prompt_ids = enc.get_ids();
        let prompt_len = prompt_ids.len();

        if prompt_len + MAX_NEW_TOKENS >= MAX_SEQ {
            eprintln!("  (prompt too long, max {} tokens total)", MAX_SEQ);
            continue;
        }

        // ── Prefill: token-by-token through decode path ──
        let prefill_start = Instant::now();
        let mut position = 0usize;
        let mut all_tokens: Vec<u32> = Vec::new();

        for &tok in prompt_ids {
            all_tokens.push(tok);
            decode_one(
                &mut dc_hidden,
                &mut dc_ffn_out,
                &dc_attn_out,
                &dc_mask,
                &lm_out,
                &decode_attn,
                &decode_ffn,
                &lm_head,
                &kv,
                &weights,
                &mut logits_buf,
                tok,
                position,
                d,
                cfg.n_layer,
                cfg.vocab_size,
            );
            position += 1;
        }

        let first_tok = sample(&logits_buf, &all_tokens, &mut rng);
        let _prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        // Print first generated token
        all_tokens.push(first_tok);
        let base_text = tokenizer
            .decode(prompt_ids, true)
            .map_err(|e| format!("{e}"))?;
        let mut prev_text = tokenizer
            .decode(&all_tokens, true)
            .map_err(|e| format!("{e}"))?;
        if let Some(delta) = prev_text.strip_prefix(&base_text) {
            print!("{delta}");
        }
        io::stdout().flush()?;

        // ── Decode loop ──
        let gen_start = Instant::now();
        let mut gen_count = 1usize;
        let eos_id = tokenizer.token_to_id("<|endoftext|>");

        for _ in 0..MAX_NEW_TOKENS - 1 {
            if position >= MAX_SEQ - 1 {
                break;
            }

            let tok = *all_tokens.last().unwrap();
            // Stop on EOS
            if eos_id == Some(tok) {
                break;
            }

            decode_one(
                &mut dc_hidden,
                &mut dc_ffn_out,
                &dc_attn_out,
                &dc_mask,
                &lm_out,
                &decode_attn,
                &decode_ffn,
                &lm_head,
                &kv,
                &weights,
                &mut logits_buf,
                tok,
                position,
                d,
                cfg.n_layer,
                cfg.vocab_size,
            );
            position += 1;

            let next = sample(&logits_buf, &all_tokens, &mut rng);
            all_tokens.push(next);
            gen_count += 1;

            let cur = tokenizer
                .decode(&all_tokens, true)
                .map_err(|e| format!("{e}"))?;
            if let Some(delta) = cur.strip_prefix(&prev_text) {
                // Stop if we see the next instruction marker
                if delta.contains("###") {
                    let clean = delta.split("###").next().unwrap_or("");
                    print!("{clean}");
                    break;
                }
                print!("{delta}");
            }
            io::stdout().flush()?;
            prev_text = cur;
        }

        let gen_elapsed = gen_start.elapsed().as_secs_f64();
        let tok_s = gen_count as f64 / gen_elapsed;

        println!();
        eprintln!("  \x1b[2m[{gen_count} tokens, {tok_s:.1} tok/s]\x1b[0m");
    }

    Ok(())
}
