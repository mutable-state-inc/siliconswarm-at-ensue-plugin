// Shared DistilBERT model: weights, graph construction, forward pass.
// Used by both distilbert_bench.rs and distilbert_verify.rs.

use ane::{Executable, Graph, NSQualityOfService, Shape, TensorData};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use std::cell::RefCell;

pub const SEQ: usize = 128;
pub const DIM: usize = 768;
pub const HEADS: usize = 12;
pub const HD: usize = 64;
pub const FFN: usize = 3072;
pub const LAYERS: usize = 6;
pub const CLS: usize = 2;

// ─── Weights ───────────────────────────────────────────────────────────────

pub struct LW {
    pub sa_ln_w: Box<[f32]>,
    pub sa_ln_b: Box<[f32]>,
    pub q_w: Box<[f32]>,
    pub q_b: Box<[f32]>,
    pub k_w: Box<[f32]>,
    pub k_b: Box<[f32]>,
    pub v_w: Box<[f32]>,
    pub v_b: Box<[f32]>,
    pub out_w: Box<[f32]>,
    pub out_b: Box<[f32]>,
    pub ffn_ln_w: Box<[f32]>,
    pub ffn_ln_b: Box<[f32]>,
    pub ffn1_w: Box<[f32]>,
    pub ffn1_b: Box<[f32]>,
    pub ffn2_w: Box<[f32]>,
    pub ffn2_b: Box<[f32]>,
}

pub struct MW {
    pub word_emb: Box<[f32]>,
    pub pos_emb: Box<[f32]>,
    pub emb_ln_w: Box<[f32]>,
    pub emb_ln_b: Box<[f32]>,
    pub layers: Box<[LW]>,
    pub pre_w: Box<[f32]>,
    pub pre_b: Box<[f32]>,
    pub cls_w: Box<[f32]>,
    pub cls_b: Box<[f32]>,
}

pub fn tf(st: &SafeTensors, name: &str) -> Box<[f32]> {
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
        other => panic!("unsupported: {other:?}"),
    }
}

pub fn load(st: &SafeTensors) -> MW {
    let layers: Box<[LW]> = (0..LAYERS)
        .map(|i| {
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
        })
        .collect();
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

fn s1() -> Shape {
    Shape {
        batch: 1,
        channels: 1,
        height: 1,
        width: 1,
    }
}
fn sc(d: usize) -> Shape {
    Shape {
        batch: 1,
        channels: d,
        height: 1,
        width: 1,
    }
}

pub fn layer_norm(g: &mut Graph, x: ane::Tensor, w: &[f32], b: &[f32], d: usize) -> ane::Tensor {
    let wt = g.constant(w, sc(d));
    let bt = g.constant(b, sc(d));
    let eps = g.constant_with_scalar(1e-12, s1());
    let nhalf = g.constant_with_scalar(-0.5, s1());
    let mean = g.reduce_mean(x, 1);
    let centered = g.subtraction(x, mean);
    let sq = g.multiplication(centered, centered);
    let var = g.reduce_mean(sq, 1);
    let var_eps = g.addition(var, eps);
    let rstd = g.power(var_eps, nhalf);
    let normed = g.multiplication(centered, rstd);
    let scaled = g.multiplication(normed, wt);
    g.addition(scaled, bt)
}

pub fn gelu(g: &mut Graph, x: ane::Tensor) -> ane::Tensor {
    let c = g.constant_with_scalar(0.044715, s1());
    let s = g.constant_with_scalar(0.797_884_6, s1());
    let x2 = g.multiplication(x, x);
    let x3 = g.multiplication(x2, x);
    let cx3 = g.multiplication(c, x3);
    let inner = g.addition(x, cx3);
    let arg = g.multiplication(s, inner);
    let th = g.tanh(arg);
    let coeff = g.linear(th, 0.5, 0.5); // 0.5*tanh + 0.5 = 0.5*(1+tanh)
    g.multiplication(x, coeff)
}

// ─── Layer graph builder ──────────────────────────────────────────────────

fn add_layer_to_graph(g: &mut Graph, x: ane::Tensor, mask: ane::Tensor, w: &LW) -> ane::Tensor {
    // Fused QKV projection with pre-scaled Q weights (eliminates runtime scale multiply)
    let q_scale = 1.0 / (HD as f32).sqrt();
    let mut qkv_w = Vec::with_capacity(3 * DIM * DIM);
    qkv_w.extend(w.q_w.iter().map(|&x| x * q_scale));
    qkv_w.extend_from_slice(&w.k_w);
    qkv_w.extend_from_slice(&w.v_w);
    let mut qkv_b = Vec::with_capacity(3 * DIM);
    qkv_b.extend(w.q_b.iter().map(|&x| x * q_scale));
    qkv_b.extend_from_slice(&w.k_b);
    qkv_b.extend_from_slice(&w.v_b);
    let qkv_proj = g.inner_product(x, &qkv_w, DIM, 3 * DIM);
    let qkv_bias = g.constant(&qkv_b, sc(3 * DIM));
    let qkv = g.addition(qkv_proj, qkv_bias);
    // Reshape then slice: combine reshape+slice into fewer ops
    let qkv = g.reshape(
        qkv,
        Shape {
            batch: 1,
            channels: 3 * HEADS,
            height: HD,
            width: SEQ,
        },
    );
    let q = g.slice(qkv, [0, 0, 0, 0], [1, HEADS, HD, SEQ]);
    let k = g.slice(qkv, [0, HEADS, 0, 0], [1, HEADS, HD, SEQ]);
    let v = g.slice(qkv, [0, 2 * HEADS, 0, 0], [1, HEADS, HD, SEQ]);

    // Attention with mask (scale pre-baked into Q weights, transposes in matmul flags)
    let raw = g.matrix_multiplication(q, k, true, false); // Q^T × K = [SEQ,HD]×[HD,SEQ]=[SEQ,SEQ]
    let masked = g.addition(raw, mask);
    let probs = g.soft_max(masked, -1);
    let attn = g.matrix_multiplication(probs, v, false, true); // probs × V^T = [SEQ,SEQ]×[SEQ,HD]=[SEQ,HD]
    let attn = g.transpose(attn, [0, 1, 3, 2]); // [HEADS,SEQ,HD] → [HEADS,HD,SEQ]
    let attn = g.reshape(attn, Shape::spatial(DIM, 1, SEQ));

    // Output projection + residual + LayerNorm (POST-norm)
    let o_proj = g.inner_product(attn, &w.out_w, DIM, DIM);
    let o_bias = g.constant(&w.out_b, sc(DIM));
    let o = g.addition(o_proj, o_bias);
    let h = g.addition(o, x);
    let h = layer_norm(g, h, &w.sa_ln_w, &w.sa_ln_b, DIM);

    // FFN + residual + LayerNorm (POST-norm)
    let fc1_proj = g.inner_product(h, &w.ffn1_w, DIM, FFN);
    let fc1_bias = g.constant(&w.ffn1_b, sc(FFN));
    let fc1 = g.addition(fc1_proj, fc1_bias);
    let fc1 = gelu(g, fc1);
    let fc2_proj = g.inner_product(fc1, &w.ffn2_w, FFN, DIM);
    let fc2_bias = g.constant(&w.ffn2_b, sc(DIM));
    let fc2 = g.addition(fc2_proj, fc2_bias);
    let out = g.addition(fc2, h);
    layer_norm(g, out, &w.ffn_ln_w, &w.ffn_ln_b, DIM)
}

fn compile_two_layers(w0: &LW, w1: &LW) -> Executable {
    let mut g = Graph::new();
    let x = g.placeholder(Shape::spatial(DIM, 1, SEQ));
    let mask = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: SEQ,
        width: SEQ,
    });
    let h = add_layer_to_graph(&mut g, x, mask, w0);
    let _ = add_layer_to_graph(&mut g, h, mask, w1);
    g.compile(NSQualityOfService::UserInteractive)
        .expect("two-layer compile")
}

fn compile_two_layers_with_cls(
    w0: &LW,
    w1: &LW,
    pre_w: &[f32],
    pre_b: &[f32],
    cls_w: &[f32],
    cls_b: &[f32],
) -> Executable {
    let mut g = Graph::new();
    let x = g.placeholder(Shape::spatial(DIM, 1, SEQ));
    let mask = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: SEQ,
        width: SEQ,
    });
    let h = add_layer_to_graph(&mut g, x, mask, w0);
    let h = add_layer_to_graph(&mut g, h, mask, w1);
    // Classifier head
    let pre_proj = g.inner_product(h, pre_w, DIM, DIM);
    let pre_bias = g.constant(pre_b, sc(DIM));
    let pre = g.addition(pre_proj, pre_bias);
    let pre = g.relu(pre);
    let cls_proj = g.inner_product(pre, cls_w, DIM, CLS);
    let cls_bt = g.constant(cls_b, sc(CLS));
    let _ = g.addition(cls_proj, cls_bt);
    g.compile(NSQualityOfService::UserInteractive)
        .expect("two-layer+cls compile")
}

fn clone_lw(w: &LW) -> LW {
    LW {
        sa_ln_w: w.sa_ln_w.clone(),
        sa_ln_b: w.sa_ln_b.clone(),
        q_w: w.q_w.clone(),
        q_b: w.q_b.clone(),
        k_w: w.k_w.clone(),
        k_b: w.k_b.clone(),
        v_w: w.v_w.clone(),
        v_b: w.v_b.clone(),
        out_w: w.out_w.clone(),
        out_b: w.out_b.clone(),
        ffn_ln_w: w.ffn_ln_w.clone(),
        ffn_ln_b: w.ffn_ln_b.clone(),
        ffn1_w: w.ffn1_w.clone(),
        ffn1_b: w.ffn1_b.clone(),
        ffn2_w: w.ffn2_w.clone(),
        ffn2_b: w.ffn2_b.clone(),
    }
}

// ─── Fused layer cache ────────────────────────────────────────────────────

thread_local! {
    static FUSED_PAIR_EXES: RefCell<Vec<Executable>> = RefCell::new(Vec::new());
    static PENDING_LAYER: RefCell<Option<LW>> = RefCell::new(None);
    static ALL_LAYER_WEIGHTS: RefCell<Vec<LW>> = RefCell::new(Vec::new());
    static CLS_FUSED: RefCell<bool> = RefCell::new(false);
}

// ─── Compile ───────────────────────────────────────────────────────────────

/// Compiles layers, fusing pairs of 2 layers into single ANE dispatches.
pub fn compile_layer(w: &LW) -> Executable {
    // Store weights for potential classifier fusion later
    ALL_LAYER_WEIGHTS.with(|alw| alw.borrow_mut().push(clone_lw(w)));

    // Build fused 2-layer executables via thread-local state
    let fused = PENDING_LAYER.with(|pl| {
        let mut pending = pl.borrow_mut();
        if let Some(prev) = pending.take() {
            Some(compile_two_layers(&prev, w))
        } else {
            *pending = Some(clone_lw(w));
            None
        }
    });
    if let Some(exe) = fused {
        FUSED_PAIR_EXES.with(|f| f.borrow_mut().push(exe));
    }

    // Return pass-through executable (fused path is used at runtime)
    let mut g = Graph::new();
    let x = g.placeholder(Shape::spatial(DIM, 1, SEQ));
    let _mask = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: SEQ,
        width: SEQ,
    });
    let zero = g.constant_with_scalar(0.0, s1());
    let _ = g.addition(x, zero);
    g.compile(NSQualityOfService::UserInteractive)
        .expect("passthrough compile")
}

pub fn compile_classifier(mw: &MW) -> Executable {
    // Try to fuse classifier into the last 2-layer pair (3 dispatches instead of 4)
    ALL_LAYER_WEIGHTS.with(|alw| {
        let weights = alw.borrow();
        if weights.len() == LAYERS {
            FUSED_PAIR_EXES.with(|f| {
                let mut fused = f.borrow_mut();
                if fused.len() == LAYERS / 2 {
                    // Recompile last pair with classifier fused in
                    let fused_exe = compile_two_layers_with_cls(
                        &weights[LAYERS - 2],
                        &weights[LAYERS - 1],
                        &mw.pre_w,
                        &mw.pre_b,
                        &mw.cls_w,
                        &mw.cls_b,
                    );
                    *fused.last_mut().unwrap() = fused_exe;
                    CLS_FUSED.with(|c| *c.borrow_mut() = true);
                }
            });
        }
    });

    // Return pass-through (classifier is fused into last pair at runtime)
    let mut g = Graph::new();
    let x = g.placeholder(Shape::spatial(DIM, 1, SEQ));
    let zero = g.constant_with_scalar(0.0, s1());
    let _ = g.addition(x, zero);
    g.compile(NSQualityOfService::UserInteractive)
        .expect("cls compile")
}

// ─── Inference ─────────────────────────────────────────────────────────────

pub fn embed(mw: &MW, tok: &tokenizers::Tokenizer, text: &str, hidden: &TensorData) -> usize {
    let enc = tok.encode(text, true).expect("encode");
    let ids = enc.get_ids();
    let len = ids.len().min(SEQ);
    let mut surf = hidden.as_f32_slice_mut();
    surf.fill(0.0);
    for pos in 0..SEQ {
        let t = if pos < len { ids[pos] as usize } else { 0 };
        for c in 0..DIM {
            surf[c * SEQ + pos] = mw.word_emb[t * DIM + c] + mw.pos_emb[pos * DIM + c];
        }
    }
    // Embedding LayerNorm (CPU, f32)
    for s in 0..SEQ {
        let mut mean = 0f32;
        for c in 0..DIM {
            mean += surf[c * SEQ + s];
        }
        mean /= DIM as f32;
        let mut var = 0f32;
        for c in 0..DIM {
            let d = surf[c * SEQ + s] - mean;
            var += d * d;
        }
        var /= DIM as f32;
        let rstd = 1.0 / (var + 1e-12_f32).sqrt();
        for c in 0..DIM {
            surf[c * SEQ + s] = (surf[c * SEQ + s] - mean) * rstd * mw.emb_ln_w[c] + mw.emb_ln_b[c];
        }
    }
    len
}

pub fn set_mask(mask: &TensorData, seq_len: usize) {
    let mut m = mask.as_f32_slice_mut();
    for row in 0..SEQ {
        for col in 0..SEQ {
            m[row * SEQ + col] = if row < seq_len && col < seq_len {
                0.0
            } else {
                -65504.0
            };
        }
    }
}

pub fn forward(
    layer_exes: &[Executable],
    cls_exe: &Executable,
    hidden_a: &TensorData,
    hidden_b: &TensorData,
    mask: &TensorData,
    cls_out: &TensorData,
) {
    let cls_is_fused = CLS_FUSED.with(|c| *c.borrow());

    let used_fused = FUSED_PAIR_EXES.with(|f| {
        let fused = f.borrow();
        if fused.len() == LAYERS / 2 {
            let last_idx = fused.len() - 1;
            for (i, exe) in fused.iter().enumerate() {
                if i == last_idx && cls_is_fused {
                    // Last pair has classifier fused — output to cls_out
                    let src = if i % 2 == 0 { hidden_a } else { hidden_b };
                    exe.run(&[src, mask], &[cls_out]).unwrap();
                } else {
                    let (src, dst) = if i % 2 == 0 {
                        (hidden_a, hidden_b)
                    } else {
                        (hidden_b, hidden_a)
                    };
                    exe.run(&[src, mask], &[dst]).unwrap();
                }
            }
            true
        } else {
            false
        }
    });
    if !used_fused {
        for (i, exe) in layer_exes.iter().enumerate() {
            let (src, dst) = if i % 2 == 0 {
                (hidden_a, hidden_b)
            } else {
                (hidden_b, hidden_a)
            };
            exe.run(&[src, mask], &[dst]).unwrap();
        }
    }
    if !cls_is_fused {
        let num = if used_fused { LAYERS / 2 } else { LAYERS };
        let final_h = if num % 2 == 0 { hidden_a } else { hidden_b };
        cls_exe.run(&[final_h], &[cls_out]).unwrap();
    }
}

pub fn classify(cls_out: &TensorData) -> (f32, f32, &'static str) {
    let out = cls_out.as_f32_slice();
    let neg = out[0];
    let pos = out[1 * SEQ];
    (neg, pos, if pos > neg { "POSITIVE" } else { "NEGATIVE" })
}
