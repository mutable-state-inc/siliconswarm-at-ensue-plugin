/// SmolLM-135M (instruction-tuned) on Apple Neural Engine via private API.
/// LLaMA architecture: half-split RoPE (CPU), GQA (CPU), SwiGLU + RMSNorm (ANE).
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

const REPO_ID: &str = "HuggingFaceTB/SmolLM-135M-Instruct";
const MAX_NEW_TOKENS: usize = 100;
const MAX_SEQ: usize = 128;
const SPATIAL: usize = 64;
const TEMPERATURE: f32 = 0.7;
const TOP_P: f32 = 0.9;
const REPETITION_PENALTY: f32 = 1.1;

#[derive(Debug, Clone, Deserialize)]
struct Config {
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    vocab_size: usize,
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
    #[serde(default = "default_rms_eps")]
    rms_norm_eps: f64,
    #[serde(default)]
    tie_word_embeddings: bool,
}
fn default_rope_theta() -> f64 { 10000.0 }
fn default_rms_eps() -> f64 { 1e-5 }
impl Config {
    fn head_dim(&self) -> usize { self.hidden_size / self.num_attention_heads }
    fn kv_dim(&self) -> usize { self.num_key_value_heads * self.head_dim() }
    fn num_groups(&self) -> usize { self.num_attention_heads / self.num_key_value_heads }
}

// ─── Weights ───────────────────────────────────────────────────────────────

struct LayerWeights {
    input_ln_w: Box<[f32]>,
    q_proj_w: Box<[f32]>, k_proj_w: Box<[f32]>, v_proj_w: Box<[f32]>,
    o_proj_w: Box<[f32]>,
    post_ln_w: Box<[f32]>,
    gate_proj_w: Box<[f32]>, up_proj_w: Box<[f32]>, down_proj_w: Box<[f32]>,
}

struct ModelWeights {
    embed_tokens: Box<[f32]>,
    layers: Box<[LayerWeights]>,
    final_ln_w: Box<[f32]>,
    lm_head: Box<[f32]>,
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

fn download(repo_id: &str) -> Result<(Config, PathBuf, Vec<u8>), Box<dyn std::error::Error>> {
    let api = ApiBuilder::new().with_progress(true).build()?;
    let repo = api.model(repo_id.to_string());
    eprint_status("Downloading config.json");
    let config: Config = serde_json::from_reader(std::fs::File::open(repo.get("config.json")?)?)?;
    eprint_status("Downloading tokenizer");
    let tok_path = repo.get("tokenizer.json").or_else(|_| {
        api.model("openai-community/gpt2".to_string()).get("tokenizer.json")
    })?;
    eprint_status("Downloading model.safetensors");
    let sb = std::fs::read(repo.get("model.safetensors")?)?;
    eprint_done("Downloaded model files");
    Ok((config, tok_path, sb))
}

fn load_weights(st: &SafeTensors, cfg: &Config) -> ModelWeights {
    let d = cfg.hidden_size;
    // LLaMA weights are [out, in] — same as ANE conv expects, no transpose needed
    let layers: Box<[LayerWeights]> = (0..cfg.num_hidden_layers).map(|i| {
        let p = format!("model.layers.{i}");
        LayerWeights {
            input_ln_w: tensor_f32(st, &format!("{p}.input_layernorm.weight")),
            q_proj_w: tensor_f32(st, &format!("{p}.self_attn.q_proj.weight")),
            k_proj_w: tensor_f32(st, &format!("{p}.self_attn.k_proj.weight")),
            v_proj_w: tensor_f32(st, &format!("{p}.self_attn.v_proj.weight")),
            o_proj_w: tensor_f32(st, &format!("{p}.self_attn.o_proj.weight")),
            post_ln_w: tensor_f32(st, &format!("{p}.post_attention_layernorm.weight")),
            gate_proj_w: tensor_f32(st, &format!("{p}.mlp.gate_proj.weight")),
            up_proj_w: tensor_f32(st, &format!("{p}.mlp.up_proj.weight")),
            down_proj_w: tensor_f32(st, &format!("{p}.mlp.down_proj.weight")),
        }
    }).collect();
    let embed = tensor_f32(st, "model.embed_tokens.weight");
    let lm_head = if cfg.tie_word_embeddings { embed.clone() } else { tensor_f32(st, "lm_head.weight") };
    ModelWeights {
        embed_tokens: embed, layers,
        final_ln_w: tensor_f32(st, "model.norm.weight"), lm_head,
    }
}

// ─── RoPE: HALF-SPLIT (LLaMA convention) ───────────────────────────────────

fn precompute_rope(head_dim: usize, max_seq: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0f32; max_seq * half];
    let mut sin = vec![0f32; max_seq * half];
    for pos in 0..max_seq {
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let angle = pos as f64 * freq;
            cos[pos * half + i] = angle.cos() as f32;
            sin[pos * half + i] = angle.sin() as f32;
        }
    }
    (cos, sin)
}

/// Half-split RoPE: element i pairs with element i + half (NOT interleaved i, i+1)
fn apply_rope_cpu(data: &mut [f32], num_heads: usize, head_dim: usize, pos: usize,
                  cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half {
            let c = cos[pos * half + i];
            let s = sin[pos * half + i];
            let x0 = data[base + i];          // first half
            let x1 = data[base + i + half];   // second half
            data[base + i]        = x0 * c - x1 * s;
            data[base + i + half] = x0 * s + x1 * c;
        }
    }
}

// ─── ANE Graph helpers ─────────────────────────────────────────────────────

fn scalar() -> Shape { Shape { batch: 1, channels: 1, height: 1, width: 1 } }

fn rms_norm(g: &mut Graph, input: ane::Tensor, weight: &[f32], d: usize, eps: f64) -> ane::Tensor {
    let w = g.constant(weight, Shape { batch: 1, channels: d, height: 1, width: 1 });
    let neg_half = g.constant_with_scalar(-0.5, scalar());
    // fp16-safe RMSNorm: scale down before squaring to avoid overflow.
    // x_s = x * (1/16), then mean(x_s²) + eps_s, rsqrt, x_s * rsqrt * weight.
    // The 1/16 factor cancels: x_s * rsqrt(mean(x_s²) + eps_s) = x * rsqrt(mean(x²) + eps).
    let scale = g.constant_with_scalar(1.0 / 16.0, scalar());
    let eps_scaled = g.constant_with_scalar((eps / (16.0 * 16.0) as f64) as f32, scalar());
    let x_s = g.multiplication(input, scale);
    let x_s2 = g.multiplication(x_s, x_s);
    let mean = g.reduce_mean(x_s2, 1);
    let mean_eps = g.addition(mean, eps_scaled);
    let rstd = g.power(mean_eps, neg_half);
    let normed = g.multiplication(x_s, rstd);
    g.multiplication(normed, w)
}

fn silu(g: &mut Graph, input: ane::Tensor) -> ane::Tensor {
    let s = g.sigmoid(input);
    g.multiplication(input, s)
}

// ─── ANE Executables ───────────────────────────────────────────────────────

/// Full single-layer graph: RMSNorm + QKV + RoPE + Attention + FFN + residual.
/// If this compiles, we get 1 dispatch per layer = 31 total.
fn compile_full_layer(w: &LayerWeights, cfg: &Config) -> Result<Executable, ane::Error> {
    let d = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let hd = cfg.head_dim();
    let kv_dim = cfg.kv_dim();
    let ff = cfg.intermediate_size;
    let mut g = Graph::new();

    let x = g.placeholder(Shape::spatial(d, 1, SPATIAL));
    let k_cache = g.placeholder(Shape { batch: 1, channels: nh, height: MAX_SEQ, width: hd });
    let v_cache = g.placeholder(Shape { batch: 1, channels: nh, height: MAX_SEQ, width: hd });
    let mask = g.placeholder(Shape { batch: 1, channels: 1, height: SPATIAL, width: MAX_SEQ });
    let rope_cos = g.placeholder(Shape { batch: 1, channels: hd / 2, height: 1, width: SPATIAL });
    let rope_sin = g.placeholder(Shape { batch: 1, channels: hd / 2, height: 1, width: SPATIAL });

    // RMSNorm
    let normed = rms_norm(&mut g, x, &w.input_ln_w, d, cfg.rms_norm_eps);

    // QKV projection
    let q = g.inner_product(normed, &w.q_proj_w, d, d);
    let k_new = g.inner_product(normed, &w.k_proj_w, d, kv_dim);
    let v_new = g.inner_product(normed, &w.v_proj_w, d, kv_dim);

    // RoPE on Q
    let q_mh = g.reshape(q, Shape { batch: nh, channels: hd, height: 1, width: SPATIAL });
    let q_rot = rope_in_graph(&mut g, q_mh, rope_cos, rope_sin, nh, hd);

    // RoPE on K
    let k_mh = g.reshape(k_new, Shape { batch: nkv, channels: hd, height: 1, width: SPATIAL });
    let k_rot = rope_in_graph(&mut g, k_mh, rope_cos, rope_sin, nkv, hd);
    let k_new_flat = g.reshape(k_rot, Shape::spatial(kv_dim, 1, SPATIAL));

    // Attention: reshape Q for matmul
    let hw = [0, 1, 3, 2];
    let q_attn = g.transpose(q_rot, hw); // [nh, SPATIAL, 1, hd] — already in [nh, hd, 1, SP] from rope
    // Wait, q_rot is [nh, hd, 1, SPATIAL]. Transpose HW → [nh, hd, SPATIAL, 1]... no.
    // Need to go from [nh, hd, 1, SPATIAL] → [1, nh, SPATIAL, hd] for matmul.
    // First reshape to [1, nh, hd, SPATIAL], then transpose HW.
    let q_4d = g.reshape(q_rot, Shape { batch: 1, channels: nh, height: hd, width: SPATIAL });
    let q_attn = g.transpose(q_4d, hw); // [1, nh, SPATIAL, hd]

    let scale = g.constant_with_scalar(1.0 / (hd as f32).sqrt(), scalar());
    let raw_scores = g.matrix_multiplication(q_attn, k_cache, false, true);
    let scores = g.multiplication(raw_scores, scale);
    let scores = g.addition(scores, mask);
    let probs = g.soft_max(scores, -1);
    let attn_raw = g.matrix_multiplication(probs, v_cache, false, false);
    let attn = g.transpose(attn_raw, hw);
    let attn = g.reshape(attn, Shape::spatial(d, 1, SPATIAL));

    // Output projection + residual
    let o = g.inner_product(attn, &w.o_proj_w, d, d);
    let h = g.addition(o, x);

    // FFN: RMSNorm + SwiGLU
    let normed2 = rms_norm(&mut g, h, &w.post_ln_w, d, cfg.rms_norm_eps);
    let gate = g.inner_product(normed2, &w.gate_proj_w, d, ff);
    let gate = silu(&mut g, gate);
    let up = g.inner_product(normed2, &w.up_proj_w, d, ff);
    let hidden = g.multiplication(gate, up);
    let down = g.inner_product(hidden, &w.down_proj_w, ff, d);
    let layer_out = g.addition(down, h);

    // Output: [layer_out, k_new_rotated, v_new] for cache update
    let _ = g.concat(&[layer_out, k_new_flat, v_new], 1);
    g.compile(NSQualityOfService::Default)
}

/// Half-split RoPE inside ANE graph. Input shape: [num_heads, head_dim, 1, SPATIAL].
/// cos/sin shape: [1, half_dim, 1, SPATIAL] — broadcasts over batch (heads).
fn rope_in_graph(
    g: &mut Graph, x: ane::Tensor, cos: ane::Tensor, sin: ane::Tensor,
    num_heads: usize, head_dim: usize,
) -> ane::Tensor {
    let half = head_dim / 2;
    // Split into first and second half of head_dim
    let x_first = g.slice(x, [0, 0, 0, 0], [num_heads, half, 1, SPATIAL]);
    let x_second = g.slice(x, [0, half, 0, 0], [num_heads, half, 1, SPATIAL]);
    // Rotate: first' = first * cos - second * sin
    let fc = g.multiplication(x_first, cos);
    let ss = g.multiplication(x_second, sin);
    let rot_first = g.subtraction(fc, ss);
    // second' = first * sin + second * cos
    let fs = g.multiplication(x_first, sin);
    let sc = g.multiplication(x_second, cos);
    let rot_second = g.addition(fs, sc);
    // Concat back to [num_heads, head_dim, 1, SPATIAL]
    g.concat(&[rot_first, rot_second], 1)
}

/// QKV projection + RoPE, all on ANE. Eliminates CPU RoPE round-trip.
fn compile_qkv_rope(w: &LayerWeights, cfg: &Config) -> Executable {
    let d = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let hd = cfg.head_dim();
    let kv_dim = cfg.kv_dim();
    let mut g = Graph::new();

    let x = g.placeholder(Shape::spatial(d, 1, SPATIAL));
    let rope_cos = g.placeholder(Shape { batch: 1, channels: hd / 2, height: 1, width: SPATIAL });
    let rope_sin = g.placeholder(Shape { batch: 1, channels: hd / 2, height: 1, width: SPATIAL });

    let normed = rms_norm(&mut g, x, &w.input_ln_w, d, cfg.rms_norm_eps);

    // Q/K/V projections
    let q = g.inner_product(normed, &w.q_proj_w, d, d);
    let k = g.inner_product(normed, &w.k_proj_w, d, kv_dim);
    let v = g.inner_product(normed, &w.v_proj_w, d, kv_dim);

    // Reshape Q to [nh, hd, 1, SPATIAL] for RoPE
    let q_mh = g.reshape(q, Shape { batch: nh, channels: hd, height: 1, width: SPATIAL });
    let q_rot = rope_in_graph(&mut g, q_mh, rope_cos, rope_sin, nh, hd);
    let q_flat = g.reshape(q_rot, Shape::spatial(d, 1, SPATIAL));

    // Reshape K to [nkv, hd, 1, SPATIAL] for RoPE
    let k_mh = g.reshape(k, Shape { batch: nkv, channels: hd, height: 1, width: SPATIAL });
    let k_rot = rope_in_graph(&mut g, k_mh, rope_cos, rope_sin, nkv, hd);
    let k_flat = g.reshape(k_rot, Shape::spatial(kv_dim, 1, SPATIAL));

    // Output: [rotated_Q, rotated_K, V] concatenated
    let _ = g.concat(&[q_flat, k_flat, v], 1);
    g.compile(NSQualityOfService::Default).expect("qkv_rope compile")
}

/// Merged attention + FFN: attention scores + softmax + V + output proj + residual + RMSNorm + SwiGLU + residual
fn compile_attn_ffn(w: &LayerWeights, cfg: &Config) -> Result<Executable, ane::Error> {
    let d = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let hd = cfg.head_dim();
    let ff = cfg.intermediate_size;
    let mut g = Graph::new();

    let q_flat = g.placeholder(Shape::spatial(d, 1, SPATIAL));
    let k_full = g.placeholder(Shape { batch: 1, channels: nh, height: MAX_SEQ, width: hd });
    let v_full = g.placeholder(Shape { batch: 1, channels: nh, height: MAX_SEQ, width: hd });
    let mask = g.placeholder(Shape { batch: 1, channels: 1, height: SPATIAL, width: MAX_SEQ });
    let x_residual = g.placeholder(Shape::spatial(d, 1, SPATIAL));

    let q = g.reshape(q_flat, Shape { batch: 1, channels: nh, height: hd, width: SPATIAL });
    let hw = [0, 1, 3, 2];
    let q = g.transpose(q, hw);
    let scale = g.constant_with_scalar(1.0 / (hd as f32).sqrt(), scalar());
    let raw_scores = g.matrix_multiplication(q, k_full, false, true);
    let scores = g.multiplication(raw_scores, scale);
    let scores = g.addition(scores, mask);
    let probs = g.soft_max(scores, -1);
    let attn_raw = g.matrix_multiplication(probs, v_full, false, false);
    let attn = g.transpose(attn_raw, hw);
    let attn = g.reshape(attn, Shape::spatial(d, 1, SPATIAL));
    let o = g.inner_product(attn, &w.o_proj_w, d, d);
    let h = g.addition(o, x_residual);

    // FFN
    let normed = rms_norm(&mut g, h, &w.post_ln_w, d, cfg.rms_norm_eps);
    let gate = g.inner_product(normed, &w.gate_proj_w, d, ff);
    let gate = silu(&mut g, gate);
    let up = g.inner_product(normed, &w.up_proj_w, d, ff);
    let hidden = g.multiplication(gate, up);
    let down = g.inner_product(hidden, &w.down_proj_w, ff, d);
    let _ = g.addition(down, h);

    g.compile(NSQualityOfService::Default)
}

fn compile_attn(w: &LayerWeights, cfg: &Config) -> Executable {
    let d = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let hd = cfg.head_dim();
    let mut g = Graph::new();
    let q_flat = g.placeholder(Shape::spatial(d, 1, SPATIAL));
    let k_full = g.placeholder(Shape { batch: 1, channels: nh, height: MAX_SEQ, width: hd });
    let v_full = g.placeholder(Shape { batch: 1, channels: nh, height: MAX_SEQ, width: hd });
    let mask = g.placeholder(Shape { batch: 1, channels: 1, height: SPATIAL, width: MAX_SEQ });
    let x_residual = g.placeholder(Shape::spatial(d, 1, SPATIAL));

    let q = g.reshape(q_flat, Shape { batch: 1, channels: nh, height: hd, width: SPATIAL });
    let hw = [0, 1, 3, 2];
    let q = g.transpose(q, hw); // [1, nh, SPATIAL, hd]
    let scale = g.constant_with_scalar(1.0 / (hd as f32).sqrt(), scalar());
    let raw_scores = g.matrix_multiplication(q, k_full, false, true);
    let scores = g.multiplication(raw_scores, scale);
    let scores = g.addition(scores, mask);
    let probs = g.soft_max(scores, -1);
    let attn_raw = g.matrix_multiplication(probs, v_full, false, false);
    let attn = g.transpose(attn_raw, hw);
    let attn = g.reshape(attn, Shape::spatial(d, 1, SPATIAL));
    let o = g.inner_product(attn, &w.o_proj_w, d, d);
    let _ = g.addition(o, x_residual);
    g.compile(NSQualityOfService::Default).expect("attn compile")
}

fn compile_ffn(w: &LayerWeights, cfg: &Config) -> Executable {
    let d = cfg.hidden_size;
    let ff = cfg.intermediate_size;
    let mut g = Graph::new();
    let x = g.placeholder(Shape::spatial(d, 1, SPATIAL));
    let normed = rms_norm(&mut g, x, &w.post_ln_w, d, cfg.rms_norm_eps);
    let gate = g.inner_product(normed, &w.gate_proj_w, d, ff);
    let gate = silu(&mut g, gate);
    let up = g.inner_product(normed, &w.up_proj_w, d, ff);
    let hidden = g.multiplication(gate, up);
    let down = g.inner_product(hidden, &w.down_proj_w, ff, d);
    let _ = g.addition(down, x);
    g.compile(NSQualityOfService::Default).expect("ffn compile")
}

fn compile_lm_head(weights: &ModelWeights, cfg: &Config) -> Executable {
    let d = cfg.hidden_size;
    let v = cfg.vocab_size;
    let mut g = Graph::new();
    let inp = g.placeholder(Shape::spatial(d, 1, SPATIAL));
    let normed = rms_norm(&mut g, inp, &weights.final_ln_w, d, cfg.rms_norm_eps);
    let max_chunk = 16 * 1024 * 1024 / (d * 2);
    let n = v.div_ceil(max_chunk);
    let cs = v / n;
    let chunks: Vec<_> = (0..n).map(|i| {
        let s = i * cs;
        let e = if i == n - 1 { v } else { (i + 1) * cs };
        g.inner_product(normed, &weights.lm_head[s * d..e * d], d, e - s)
    }).collect();
    let _ = g.concat(&chunks, 1);
    g.compile(NSQualityOfService::Default).expect("lm_head compile")
}

// ─── KV Cache ──────────────────────────────────────────────────────────────

/// KV cache stored as IOSurfaces in attention-ready layout [1, nh, MAX_SEQ, hd].
/// Only the new position is written each token — no full-cache reshape.
struct KvCache {
    keys: Vec<TensorData>,   // [n_layer], each shape [1, nh, MAX_SEQ, hd]
    values: Vec<TensorData>,
    nh: usize,
    nkv: usize,
    hd: usize,
    groups: usize,
}

impl KvCache {
    fn new(n_layer: usize, cfg: &Config) -> Self {
        let nh = cfg.num_attention_heads;
        let hd = cfg.head_dim();
        let shape = Shape { batch: 1, channels: nh, height: MAX_SEQ, width: hd };
        Self {
            keys: (0..n_layer).map(|_| TensorData::new(shape)).collect(),
            values: (0..n_layer).map(|_| TensorData::new(shape)).collect(),
            nh,
            nkv: cfg.num_key_value_heads,
            hd,
            groups: cfg.num_groups(),
        }
    }

    /// Write one position to the KV cache directly in fp16 (no full-surface conversion).
    /// Expands GQA inline: each KV head is written to all its Q head slots.
    fn write(&self, layer: usize, k: &[f32], v: &[f32], pos: usize) {
        let n = self.nh * self.hd; // total values to write per cache
        let mut k_indices = Vec::with_capacity(n);
        let mut k_values = Vec::with_capacity(n);
        let mut v_indices = Vec::with_capacity(n);
        let mut v_values = Vec::with_capacity(n);
        for kv_h in 0..self.nkv {
            for g in 0..self.groups {
                let q_h = kv_h * self.groups + g;
                for dd in 0..self.hd {
                    let idx = q_h * MAX_SEQ * self.hd + pos * self.hd + dd;
                    k_indices.push(idx);
                    k_values.push(k[kv_h * self.hd + dd]);
                    v_indices.push(idx);
                    v_values.push(v[kv_h * self.hd + dd]);
                }
            }
        }
        self.keys[layer].write_f32_sparse(&k_indices, &k_values);
        self.values[layer].write_f32_sparse(&v_indices, &v_values);
    }

    fn reset(&self) {
        for k in &self.keys { k.as_f32_slice_mut().fill(0.0); }
        for v in &self.values { v.as_f32_slice_mut().fill(0.0); }
    }
}

// ─── Sampling ──────────────────────────────────────────────────────────────

fn sample(logits: &[f32], history: &[u32], rng: &mut impl Rng) -> u32 {
    let mut l: Box<[f32]> = logits.into();
    for &t in history {
        let i = t as usize;
        if i < l.len() {
            if l[i] > 0.0 { l[i] /= REPETITION_PENALTY; } else { l[i] *= REPETITION_PENALTY; }
        }
    }
    if TEMPERATURE <= 0.0 {
        return l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
    }
    let max = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = l.iter().enumerate()
        .map(|(i, &v)| (i, ((v - max) / TEMPERATURE).exp())).collect();
    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let total: f32 = probs.iter().map(|p| p.1).sum();
    let cutoff = TOP_P * total;
    let mut cum = 0f32;
    let mut cands = Vec::new();
    for &(idx, p) in &probs { cum += p; cands.push((idx, p)); if cum >= cutoff { break; } }
    let ct: f32 = cands.iter().map(|p| p.1).sum();
    let thresh = rng.random::<f32>() * ct;
    let mut acc = 0f32;
    for &(idx, p) in &cands { acc += p; if acc >= thresh { return idx as u32; } }
    cands.last().unwrap().0 as u32
}

fn eprint_status(msg: &str) { eprint!("\r\x1b[2K\x1b[1;36m⠋\x1b[0m {msg}"); let _ = io::stderr().flush(); }
fn eprint_done(msg: &str) { eprintln!("\r\x1b[2K\x1b[1;32m✓\x1b[0m {msg}"); }

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let (cfg, tok_path, sb) = download(REPO_ID)?;
    let d = cfg.hidden_size;
    let kv_dim = cfg.kv_dim();
    let nh = cfg.num_attention_heads;
    let hd = cfg.head_dim();
    eprintln!("  Model: {} ({} layers, dim={}, {}Q/{}KV heads)",
        REPO_ID, cfg.num_hidden_layers, d, nh, cfg.num_key_value_heads);

    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| format!("tokenizer: {e}"))?;
    eprint_status("Loading weights");
    let st = SafeTensors::deserialize(&sb)?;
    let weights = load_weights(&st, &cfg);
    let (cos_table, sin_table) = precompute_rope(hd, MAX_SEQ, cfg.rope_theta);

    // 2 dispatches per layer: QKV+RoPE then attn+FFN = 61 total dispatches
    let mut qkv_rope_exes = Vec::new();
    let mut attn_ffn_exes = Vec::new();
    for (i, lw) in weights.layers.iter().enumerate() {
        eprint_status(&format!("Compiling layer {}/{}", i + 1, cfg.num_hidden_layers));
        qkv_rope_exes.push(compile_qkv_rope(lw, &cfg));
        attn_ffn_exes.push(compile_attn_ffn(lw, &cfg).expect("attn_ffn compile"));
    }
    eprint_status("Compiling LM head");
    let lm_head_exe = compile_lm_head(&weights, &cfg);
    eprint_done(&format!("Compiled ANE model ({:.1}s)", start.elapsed().as_secs_f64()));

    let hidden_td = TensorData::new(Shape::spatial(d, 1, SPATIAL));
    let qkv_out_td = TensorData::new(Shape::spatial(d + 2 * kv_dim, 1, SPATIAL));
    let q_td = TensorData::new(Shape::spatial(d, 1, SPATIAL));
    // KV cache IOSurfaces are inside the KvCache struct now
    let mask_td = TensorData::new(Shape { batch: 1, channels: 1, height: SPATIAL, width: MAX_SEQ });
    let residual_td = TensorData::new(Shape::spatial(d, 1, SPATIAL));
    let rope_cos_td = TensorData::new(Shape { batch: 1, channels: hd / 2, height: 1, width: SPATIAL });
    let rope_sin_td = TensorData::new(Shape { batch: 1, channels: hd / 2, height: 1, width: SPATIAL });
    let attn_out_td = TensorData::new(Shape::spatial(d, 1, SPATIAL));
    let ffn_out_td = TensorData::new(Shape::spatial(d, 1, SPATIAL));
    let lm_out_td = TensorData::new(Shape::spatial(cfg.vocab_size, 1, SPATIAL));
    let mut logits_buf = vec![0f32; cfg.vocab_size];
    let kv = KvCache::new(cfg.num_hidden_layers, &cfg);
    let mut rng = rand::rng();
    let mut q_buf = vec![0f32; d];
    let mut k_buf = vec![0f32; kv_dim];
    let mut v_buf = vec![0f32; kv_dim];

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  SmolLM-135M Instruct — Apple Neural Engine");
    eprintln!("  Type a message, press Enter. Ctrl-C to quit.");
    eprintln!("═══════════════════════════════════════════════════════");

    loop {
        kv.reset();
        mask_td.as_f32_slice_mut().fill(-65504.0);

        eprintln!();
        eprint!("\x1b[1;32m> \x1b[0m");
        io::stderr().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 { break; }
        let prompt = input.trim();
        if prompt.is_empty() { continue; }

        let formatted = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
        let enc = tokenizer.encode(formatted.as_str(), false).map_err(|e| format!("encode: {e}"))?;
        let prompt_ids = enc.get_ids();

        if prompt_ids.len() + MAX_NEW_TOKENS >= MAX_SEQ {
            eprintln!("  (prompt too long)");
            continue;
        }

        let prefill_start = Instant::now();
        let mut position = 0usize;
        let mut all_tokens: Vec<u32> = Vec::new();

        // Process each token
        #[allow(clippy::too_many_arguments)]
        let mut process_token = |tok: u32, pos: usize,
            logits_buf: &mut [f32], q_buf: &mut [f32], k_buf: &mut [f32], v_buf: &mut [f32]| {
            // Embed
            {
                let mut s = hidden_td.as_f32_slice_mut();
                s.fill(0.0);
                let t = tok as usize;
                for c in 0..d { s[c * SPATIAL] = weights.embed_tokens[t * d + c]; }
            }
            { mask_td.as_f32_slice_mut()[pos] = 0.0; }

            // Set RoPE cos/sin once per token
            {
                let half = hd / 2;
                let mut cos_vals = vec![0f32; half * SPATIAL];
                let mut sin_vals = vec![0f32; half * SPATIAL];
                for ii in 0..half {
                    cos_vals[ii * SPATIAL] = cos_table[pos * half + ii];
                    sin_vals[ii * SPATIAL] = sin_table[pos * half + ii];
                }
                rope_cos_td.copy_from_f32(&cos_vals);
                rope_sin_td.copy_from_f32(&sin_vals);
            }

            for i in 0..cfg.num_hidden_layers {
                // Save residual
                {
                    let src = hidden_td.as_f32_slice();
                    let mut dst = residual_td.as_f32_slice_mut();
                    dst.copy_from_slice(&src);
                }

                // Dispatch 1: QKV + RoPE
                qkv_rope_exes[i].run(&[&hidden_td, &rope_cos_td, &rope_sin_td], &[&qkv_out_td]).unwrap();

                // CPU: extract K/V for cache, copy Q
                {
                    let qkv = qkv_out_td.as_f32_slice();
                    for c in 0..kv_dim { k_buf[c] = qkv[(d + c) * SPATIAL]; }
                    for c in 0..kv_dim { v_buf[c] = qkv[(d + kv_dim + c) * SPATIAL]; }
                }
                kv.write(i, k_buf, v_buf, pos);
                {
                    let qkv = qkv_out_td.as_f32_slice();
                    let indices: Vec<usize> = (0..d).map(|c| c * SPATIAL).collect();
                    let q_vals: Vec<f32> = (0..d).map(|c| qkv[c * SPATIAL]).collect();
                    q_td.write_f32_sparse(&indices, &q_vals);
                }

                // Dispatch 2: Attention + FFN (merged)
                attn_ffn_exes[i].run(
                    &[&q_td, &kv.keys[i], &kv.values[i], &mask_td, &residual_td],
                    &[&ffn_out_td],
                ).unwrap();

                // Copy to hidden for next layer
                {
                    let src = ffn_out_td.as_f32_slice();
                    let mut dst = hidden_td.as_f32_slice_mut();
                    dst.fill(0.0);
                    for c in 0..d { dst[c * SPATIAL] = src[c * SPATIAL]; }
                }
            }

            // LM head
            lm_head_exe.run(&[&hidden_td], &[&lm_out_td]).unwrap();
            let o = lm_out_td.as_f32_slice();
            for v in 0..cfg.vocab_size { logits_buf[v] = o[v * SPATIAL]; }
        };

        // Prefill
        for &tok in prompt_ids {
            all_tokens.push(tok);
            process_token(tok, position, &mut logits_buf, &mut q_buf, &mut k_buf, &mut v_buf);
            position += 1;
        }

        let first_tok = sample(&logits_buf, &all_tokens, &mut rng);
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        eprint!("\r\x1b[2K");

        all_tokens.push(first_tok);
        let base_text = tokenizer.decode(prompt_ids, true).map_err(|e| format!("{e}"))?;
        let mut prev_text = tokenizer.decode(&all_tokens, true).map_err(|e| format!("{e}"))?;
        if let Some(delta) = prev_text.strip_prefix(&base_text) {
            let clean = delta.replace("<|im_end|>", "").replace("<|im_start|>", "");
            print!("{clean}");
        }
        io::stdout().flush()?;

        let eos_token = tokenizer.encode("<|im_end|>", false).ok().and_then(|e| e.get_ids().first().copied());
        let gen_start = Instant::now();
        let mut gen_count = 1usize;

        for _ in 0..MAX_NEW_TOKENS - 1 {
            if position >= MAX_SEQ - 1 { break; }
            let tok = *all_tokens.last().unwrap();
            if eos_token == Some(tok) { break; }

            process_token(tok, position, &mut logits_buf, &mut q_buf, &mut k_buf, &mut v_buf);
            position += 1;

            let next = sample(&logits_buf, &all_tokens, &mut rng);
            all_tokens.push(next);
            gen_count += 1;

            let cur = tokenizer.decode(&all_tokens, true).map_err(|e| format!("{e}"))?;
            if let Some(delta) = cur.strip_prefix(&prev_text) {
                let clean = delta.replace("<|im_end|>", "").replace("<|im_start|>", "");
                print!("{clean}");
            }
            io::stdout().flush()?;
            prev_text = cur;
        }

        let gen_elapsed = gen_start.elapsed().as_secs_f64();
        let tok_s = gen_count as f64 / gen_elapsed;
        println!();
        eprintln!("  \x1b[2m[{gen_count} tokens, {tok_s:.1} tok/s, prefill {prefill_ms:.0}ms]\x1b[0m");
    }

    Ok(())
}
