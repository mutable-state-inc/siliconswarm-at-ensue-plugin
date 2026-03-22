/// Phi-1.5 Instruct (1.4B) on Apple Neural Engine via private API.
/// Phi architecture: partial RoPE (50%), LayerNorm, GELU, MHA — close to GPT-2.
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

const REPO_ID: &str = "rasyosef/Phi-1_5-Instruct-v0.1";
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
    num_hidden_layers: usize,
    intermediate_size: usize,
    vocab_size: usize,
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
    #[serde(default = "default_eps")]
    layer_norm_eps: f64,
    #[serde(default = "default_rotary")]
    partial_rotary_factor: f64,
}
fn default_rope_theta() -> f64 { 10000.0 }
fn default_eps() -> f64 { 1e-5 }
fn default_rotary() -> f64 { 0.5 }
impl Config {
    fn head_dim(&self) -> usize { self.hidden_size / self.num_attention_heads }
    fn rotary_dim(&self) -> usize { (self.head_dim() as f64 * self.partial_rotary_factor) as usize }
}

// ─── Weights ───────────────────────────────────────────────────────────────

struct LayerWeights {
    ln_weight: Box<[f32]>, ln_bias: Box<[f32]>,
    q_proj_w: Box<[f32]>, q_proj_b: Box<[f32]>,
    k_proj_w: Box<[f32]>, k_proj_b: Box<[f32]>,
    v_proj_w: Box<[f32]>, v_proj_b: Box<[f32]>,
    dense_w: Box<[f32]>, dense_b: Box<[f32]>,  // output projection
    fc1_w: Box<[f32]>, fc1_b: Box<[f32]>,
    fc2_w: Box<[f32]>, fc2_b: Box<[f32]>,
}

struct ModelWeights {
    embed_tokens: Box<[f32]>,
    layers: Box<[LayerWeights]>,
    final_ln_w: Box<[f32]>, final_ln_b: Box<[f32]>,
    lm_head_w: Box<[f32]>, lm_head_b: Box<[f32]>,
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
    let tok_path = repo.get("tokenizer.json")?;
    eprint_status("Downloading model.safetensors");
    let sb = std::fs::read(repo.get("model.safetensors")?)?;
    eprint_done("Downloaded model files");
    Ok((config, tok_path, sb))
}

fn load_weights(st: &SafeTensors, cfg: &Config) -> ModelWeights {
    let d = cfg.hidden_size;
    let ff = cfg.intermediate_size;
    let nh = cfg.num_attention_heads;
    let hd = cfg.head_dim();
    // Phi weights are [out, in] — same as ANE conv expects
    let layers: Box<[LayerWeights]> = (0..cfg.num_hidden_layers).map(|i| {
        let p = format!("model.layers.{i}");
        LayerWeights {
            ln_weight: tensor_f32(st, &format!("{p}.input_layernorm.weight")),
            ln_bias: tensor_f32(st, &format!("{p}.input_layernorm.bias")),
            q_proj_w: tensor_f32(st, &format!("{p}.self_attn.q_proj.weight")),
            q_proj_b: tensor_f32(st, &format!("{p}.self_attn.q_proj.bias")),
            k_proj_w: tensor_f32(st, &format!("{p}.self_attn.k_proj.weight")),
            k_proj_b: tensor_f32(st, &format!("{p}.self_attn.k_proj.bias")),
            v_proj_w: tensor_f32(st, &format!("{p}.self_attn.v_proj.weight")),
            v_proj_b: tensor_f32(st, &format!("{p}.self_attn.v_proj.bias")),
            dense_w: tensor_f32(st, &format!("{p}.self_attn.dense.weight")),
            dense_b: tensor_f32(st, &format!("{p}.self_attn.dense.bias")),
            fc1_w: tensor_f32(st, &format!("{p}.mlp.fc1.weight")),
            fc1_b: tensor_f32(st, &format!("{p}.mlp.fc1.bias")),
            fc2_w: tensor_f32(st, &format!("{p}.mlp.fc2.weight")),
            fc2_b: tensor_f32(st, &format!("{p}.mlp.fc2.bias")),
        }
    }).collect();
    ModelWeights {
        embed_tokens: tensor_f32(st, "model.embed_tokens.weight"),
        layers,
        final_ln_w: tensor_f32(st, "model.final_layernorm.weight"),
        final_ln_b: tensor_f32(st, "model.final_layernorm.bias"),
        lm_head_w: tensor_f32(st, "lm_head.weight"),
        lm_head_b: tensor_f32(st, "lm_head.bias"),
    }
}

// ─── Partial RoPE (CPU, half-split) ────────────────────────────────────────

fn precompute_rope(rotary_dim: usize, max_seq: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = rotary_dim / 2;
    let mut cos = vec![0f32; max_seq * half];
    let mut sin = vec![0f32; max_seq * half];
    for pos in 0..max_seq {
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / rotary_dim as f64);
            let angle = pos as f64 * freq;
            cos[pos * half + i] = angle.cos() as f32;
            sin[pos * half + i] = angle.sin() as f32;
        }
    }
    (cos, sin)
}

/// Apply half-split RoPE to first rotary_dim dimensions only
fn apply_partial_rope_cpu(data: &mut [f32], num_heads: usize, head_dim: usize,
                          rotary_dim: usize, pos: usize, cos: &[f32], sin: &[f32]) {
    let half = rotary_dim / 2;
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half {
            let c = cos[pos * half + i];
            let s = sin[pos * half + i];
            let x0 = data[base + i];
            let x1 = data[base + i + half];
            data[base + i]        = x0 * c - x1 * s;
            data[base + i + half] = x0 * s + x1 * c;
        }
        // dims rotary_dim..head_dim are left unchanged (non-rotary)
    }
}

// ─── ANE Graph helpers ─────────────────────────────────────────────────────

fn scalar() -> Shape { Shape { batch: 1, channels: 1, height: 1, width: 1 } }

fn layer_norm(g: &mut Graph, input: ane::Tensor, weight: &[f32], bias: &[f32],
              d: usize, eps: f64) -> ane::Tensor {
    let w = g.constant(weight, Shape { batch: 1, channels: d, height: 1, width: 1 });
    let b = g.constant(bias, Shape { batch: 1, channels: d, height: 1, width: 1 });
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

// ─── ANE Executables ───────────────────────────────────────────────────────

/// Phi has a parallel attention+MLP architecture: both operate on the LayerNorm output,
/// then their results are summed with the residual.
/// Graph 1: LayerNorm + QKV projection (+ MLP fc1)
fn compile_proj(w: &LayerWeights, cfg: &Config) -> Executable {
    let d = cfg.hidden_size;
    let ff = cfg.intermediate_size;
    let mut g = Graph::new();
    let x = g.placeholder(Shape::spatial(d, 1, SPATIAL));
    let normed = layer_norm(&mut g, x, &w.ln_weight, &w.ln_bias, d, cfg.layer_norm_eps);
    let q = g.inner_product(normed, &w.q_proj_w, d, d);
    let q_b = g.constant(&w.q_proj_b, Shape { batch: 1, channels: d, height: 1, width: 1 });
    let q = g.addition(q, q_b);
    let k = g.inner_product(normed, &w.k_proj_w, d, d);
    let k_b = g.constant(&w.k_proj_b, Shape { batch: 1, channels: d, height: 1, width: 1 });
    let k = g.addition(k, k_b);
    let v = g.inner_product(normed, &w.v_proj_w, d, d);
    let v_b = g.constant(&w.v_proj_b, Shape { batch: 1, channels: d, height: 1, width: 1 });
    let v = g.addition(v, v_b);
    // Also compute MLP fc1 (gelu) since Phi runs attention and MLP in parallel on same input
    let fc1 = g.inner_product(normed, &w.fc1_w, d, ff);
    let fc1_b = g.constant(&w.fc1_b, Shape { batch: 1, channels: ff, height: 1, width: 1 });
    let fc1 = g.addition(fc1, fc1_b);
    let fc1 = gelu(&mut g, fc1);
    // Output: [Q(d) + K(d) + V(d) + fc1_gelu(ff)] = [d*3 + ff]
    let _ = g.concat(&[q, k, v, fc1], 1);
    g.compile(NSQualityOfService::Default).expect("proj compile")
}

/// Graph 2: Attention (post-RoPE) + output proj + MLP fc2 + residual
fn compile_attn_mlp(w: &LayerWeights, cfg: &Config) -> Executable {
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
    let fc1_gelu = g.placeholder(Shape::spatial(ff, 1, SPATIAL));

    // Attention
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

    // Output projection (dense)
    let o = g.inner_product(attn, &w.dense_w, d, d);
    let o_b = g.constant(&w.dense_b, Shape { batch: 1, channels: d, height: 1, width: 1 });
    let attn_out = g.addition(o, o_b);

    // MLP fc2
    let mlp_out = g.inner_product(fc1_gelu, &w.fc2_w, ff, d);
    let mlp_b = g.constant(&w.fc2_b, Shape { batch: 1, channels: d, height: 1, width: 1 });
    let mlp_out = g.addition(mlp_out, mlp_b);

    // Residual: output = residual + attn_out + mlp_out
    let sum = g.addition(attn_out, mlp_out);
    let _ = g.addition(sum, x_residual);

    g.compile(NSQualityOfService::Default).expect("attn_mlp compile")
}

fn compile_lm_head(weights: &ModelWeights, cfg: &Config) -> Executable {
    let d = cfg.hidden_size;
    let v = cfg.vocab_size;
    let mut g = Graph::new();
    let inp = g.placeholder(Shape::spatial(d, 1, SPATIAL));
    let normed = layer_norm(&mut g, inp, &weights.final_ln_w, &weights.final_ln_b, d, cfg.layer_norm_eps);
    let max_chunk = 16 * 1024 * 1024 / (d * 2);
    let n = v.div_ceil(max_chunk);
    let cs = v / n;
    let chunks: Vec<_> = (0..n).map(|i| {
        let s = i * cs;
        let e = if i == n - 1 { v } else { (i + 1) * cs };
        let proj = g.inner_product(normed, &weights.lm_head_w[s * d..e * d], d, e - s);
        let bias_data: Vec<f32> = weights.lm_head_b[s..e].to_vec();
        let bias = g.constant(&bias_data, Shape { batch: 1, channels: e - s, height: 1, width: 1 });
        g.addition(proj, bias)
    }).collect();
    let _ = g.concat(&chunks, 1);
    g.compile(NSQualityOfService::Default).expect("lm_head compile")
}

// ─── KV Cache (simple MHA — no GQA expansion needed) ──────────────────────

struct KvCache {
    keys: Vec<Vec<f32>>,   // [n_layer][d * MAX_SEQ] in NCHW
    values: Vec<Vec<f32>>,
    dim: usize,
}

impl KvCache {
    fn new(n_layer: usize, dim: usize) -> Self {
        Self {
            keys: (0..n_layer).map(|_| vec![0f32; dim * MAX_SEQ]).collect(),
            values: (0..n_layer).map(|_| vec![0f32; dim * MAX_SEQ]).collect(),
            dim,
        }
    }
    fn write(&mut self, layer: usize, k: &[f32], v: &[f32], pos: usize) {
        for c in 0..self.dim {
            self.keys[layer][c * MAX_SEQ + pos] = k[c];
            self.values[layer][c * MAX_SEQ + pos] = v[c];
        }
    }
    /// Return KV in [1, nh, MAX_SEQ, hd] NCHW layout for attention graph
    fn as_nchw(&self, layer: usize, nh: usize, hd: usize) -> Vec<f32> {
        let mut out = vec![0f32; nh * MAX_SEQ * hd];
        for h in 0..nh {
            for pos in 0..MAX_SEQ {
                for dd in 0..hd {
                    out[h * MAX_SEQ * hd + pos * hd + dd] =
                        self.keys[layer][(h * hd + dd) * MAX_SEQ + pos];
                }
            }
        }
        out
    }
    fn v_as_nchw(&self, layer: usize, nh: usize, hd: usize) -> Vec<f32> {
        let mut out = vec![0f32; nh * MAX_SEQ * hd];
        for h in 0..nh {
            for pos in 0..MAX_SEQ {
                for dd in 0..hd {
                    out[h * MAX_SEQ * hd + pos * hd + dd] =
                        self.values[layer][(h * hd + dd) * MAX_SEQ + pos];
                }
            }
        }
        out
    }
    fn reset(&mut self) {
        for k in &mut self.keys { k.fill(0.0); }
        for v in &mut self.values { v.fill(0.0); }
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
    let nh = cfg.num_attention_heads;
    let hd = cfg.head_dim();
    let ff = cfg.intermediate_size;
    let rot_dim = cfg.rotary_dim();
    eprintln!("  Model: {} ({} layers, dim={}, {} heads, rot_dim={})",
        REPO_ID, cfg.num_hidden_layers, d, nh, rot_dim);

    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| format!("tokenizer: {e}"))?;
    eprint_status("Loading weights");
    let st = SafeTensors::deserialize(&sb)?;
    let weights = load_weights(&st, &cfg);
    let (cos_table, sin_table) = precompute_rope(rot_dim, MAX_SEQ, cfg.rope_theta);

    // Compile: 2 executables per layer + 1 lm_head = 49 total
    let mut proj_exes = Vec::new();
    let mut attn_mlp_exes = Vec::new();
    for (i, lw) in weights.layers.iter().enumerate() {
        eprint_status(&format!("Compiling layer {}/{}", i + 1, cfg.num_hidden_layers));
        proj_exes.push(compile_proj(lw, &cfg));
        attn_mlp_exes.push(compile_attn_mlp(lw, &cfg));
    }
    eprint_status("Compiling LM head");
    let lm_head_exe = compile_lm_head(&weights, &cfg);
    eprint_done(&format!("Compiled ANE model ({:.1}s)", start.elapsed().as_secs_f64()));

    // Allocate surfaces
    let hidden_td = TensorData::new(Shape::spatial(d, 1, SPATIAL));
    let proj_out_td = TensorData::new(Shape::spatial(d * 3 + ff, 1, SPATIAL));
    let q_td = TensorData::new(Shape::spatial(d, 1, SPATIAL));
    let k_td = TensorData::new(Shape { batch: 1, channels: nh, height: MAX_SEQ, width: hd });
    let v_td = TensorData::new(Shape { batch: 1, channels: nh, height: MAX_SEQ, width: hd });
    let mask_td = TensorData::new(Shape { batch: 1, channels: 1, height: SPATIAL, width: MAX_SEQ });
    let residual_td = TensorData::new(Shape::spatial(d, 1, SPATIAL));
    let fc1_td = TensorData::new(Shape::spatial(ff, 1, SPATIAL));
    let layer_out_td = TensorData::new(Shape::spatial(d, 1, SPATIAL));
    let lm_out_td = TensorData::new(Shape::spatial(cfg.vocab_size, 1, SPATIAL));
    let mut logits_buf = vec![0f32; cfg.vocab_size];
    let mut kv = KvCache::new(cfg.num_hidden_layers, d);
    let mut rng = rand::rng();
    let mut q_buf = vec![0f32; d];
    let mut k_buf = vec![0f32; d];
    let mut v_buf = vec![0f32; d];

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  Phi-1.5 Instruct (1.4B) — Apple Neural Engine");
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

            for i in 0..cfg.num_hidden_layers {
                // Save residual
                { let h = hidden_td.read_f32(); residual_td.copy_from_f32(&h); }

                // Step 1: ANE projection (LayerNorm + QKV + fc1_gelu)
                proj_exes[i].run(&[&hidden_td], &[&proj_out_td]).unwrap();

                // Step 2: CPU — extract Q/K/V/fc1, apply partial RoPE
                {
                    let proj = proj_out_td.as_f32_slice();
                    for c in 0..d { q_buf[c] = proj[c * SPATIAL]; }
                    for c in 0..d { k_buf[c] = proj[(d + c) * SPATIAL]; }
                    for c in 0..d { v_buf[c] = proj[(2 * d + c) * SPATIAL]; }
                    let mut fc1_data = vec![0f32; ff * SPATIAL];
                    for c in 0..ff { fc1_data[c * SPATIAL] = proj[(3 * d + c) * SPATIAL]; }
                    fc1_td.copy_from_f32(&fc1_data);
                }
                apply_partial_rope_cpu(q_buf, nh, hd, rot_dim, pos, &cos_table, &sin_table);
                apply_partial_rope_cpu(k_buf, nh, hd, rot_dim, pos, &cos_table, &sin_table);
                kv.write(i, k_buf, v_buf, pos);

                // Write Q to surface
                {
                    let mut q_full = vec![0f32; d * SPATIAL];
                    for c in 0..d { q_full[c * SPATIAL] = q_buf[c]; }
                    q_td.copy_from_f32(&q_full);
                }
                // KV cache to surfaces (MHA, no GQA expansion)
                k_td.copy_from_f32(&kv.as_nchw(i, nh, hd));
                v_td.copy_from_f32(&kv.v_as_nchw(i, nh, hd));

                // Step 3: ANE attention + MLP fc2 + residual
                attn_mlp_exes[i].run(
                    &[&q_td, &k_td, &v_td, &mask_td, &residual_td, &fc1_td],
                    &[&layer_out_td],
                ).unwrap();

                // Copy to hidden for next layer
                {
                    let src = layer_out_td.as_f32_slice();
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

        let eos_token = tokenizer.token_to_id("<|im_end|>");
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
