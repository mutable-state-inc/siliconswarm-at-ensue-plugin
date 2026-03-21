use std::sync::Mutex;
use std::time::Instant;
use crate::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;

static DRAFT_MODEL: Mutex<Option<DraftModel>> = Mutex::new(None);

struct DraftModel {
    exec: crate::Executable,
    input: TensorData,
    output: TensorData,
    dim: usize,
    sp: usize,
    seq: usize,
}

/// Initialize the ANE draft model. Returns 0 on success, -1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn ane_draft_init(dim: u32, hidden: u32, seq: u32, num_layers: u32) -> i32 {
    let dim = dim as usize;
    let hidden = hidden as usize;
    let seq = seq as usize;
    let nl = num_layers as usize;
    
    let wpl = 4 * dim + 3 * hidden;
    let sp = seq + nl * wpl;
    
    if sp > 16384 {
        eprintln!("ane_draft_init: sp={sp} exceeds 16384 limit");
        return -1;
    }
    
    let mut g = Graph::new();
    let packed = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: sp });
    let mut h = g.slice(packed, [0, 0, 0, 0], [1, dim, 1, seq]);
    let mut wo = seq;
    
    for _ in 0..nl {
        let hr = g.reshape(h, Shape { batch: 1, channels: 1, height: dim, width: seq });
        let ht = g.transpose(hr, [0, 1, 3, 2]);
        let qw = g.slice(packed, [0, 0, 0, wo], [1, dim, 1, dim]); wo += dim * 4;
        let qr = g.reshape(qw, Shape { batch: 1, channels: 1, height: dim, width: dim });
        let q = g.matrix_multiplication(ht, qr, false, false);
        let qt = g.transpose(q, [0, 1, 3, 2]);
        let ao = g.reshape(qt, Shape { batch: 1, channels: dim, height: 1, width: seq });
        h = g.addition(h, ao);
        
        let hr2 = g.reshape(h, Shape { batch: 1, channels: 1, height: dim, width: seq });
        let ht2 = g.transpose(hr2, [0, 1, 3, 2]);
        let gw = g.slice(packed, [0, 0, 0, wo], [1, dim, 1, hidden]); wo += hidden;
        let gr = g.reshape(gw, Shape { batch: 1, channels: 1, height: dim, width: hidden });
        let gate = g.matrix_multiplication(ht2, gr, false, false);
        let uw = g.slice(packed, [0, 0, 0, wo], [1, dim, 1, hidden]); wo += hidden;
        let ur = g.reshape(uw, Shape { batch: 1, channels: 1, height: dim, width: hidden });
        let up = g.matrix_multiplication(ht2, ur, false, false);
        let gs = g.sigmoid(gate);
        let gl = g.multiplication(gate, gs);
        let mix = g.multiplication(gl, up);
        let dw = g.slice(packed, [0, 0, 0, wo], [1, dim, 1, dim]); wo += dim;
        let dr = g.reshape(dw, Shape { batch: 1, channels: 1, height: hidden, width: dim });
        let f = g.matrix_multiplication(mix, dr, false, false);
        let ft = g.transpose(f, [0, 1, 3, 2]);
        let fo = g.reshape(ft, Shape { batch: 1, channels: dim, height: 1, width: seq });
        h = g.addition(h, fo);
    }
    
    let exec = match g.compile(NSQualityOfService::Default) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("ane_draft_init: compile failed: {e}");
            return -1;
        }
    };
    
    let input = TensorData::with_f32(&vec![0.01; dim * sp], 
        Shape { batch: 1, channels: dim, height: 1, width: sp });
    let output = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });
    
    // Warmup
    for _ in 0..3 {
        let _ = exec.run_cached_direct(&[&input], &[&output]);
    }
    
    *DRAFT_MODEL.lock().unwrap() = Some(DraftModel { exec, input, output, dim, sp, seq });
    0
}

/// Run one draft forward pass. Returns latency in microseconds.
#[unsafe(no_mangle)]
pub extern "C" fn ane_draft_run() -> i64 {
    let guard = DRAFT_MODEL.lock().unwrap();
    let model = match guard.as_ref() {
        Some(m) => m,
        None => return -1,
    };
    let start = Instant::now();
    match model.exec.run_cached_direct(&[&model.input], &[&model.output]) {
        Ok(()) => start.elapsed().as_micros() as i64,
        Err(_) => -1,
    }
}

/// Benchmark N runs, return average microseconds per run.
#[unsafe(no_mangle)]  
pub extern "C" fn ane_draft_bench(n: u32) -> i64 {
    let guard = DRAFT_MODEL.lock().unwrap();
    let model = match guard.as_ref() {
        Some(m) => m,
        None => return -1,
    };
    let start = Instant::now();
    for _ in 0..n {
        let _ = model.exec.run_cached_direct(&[&model.input], &[&model.output]);
    }
    (start.elapsed().as_micros() / n as u128) as i64
}

/// Cleanup.
#[unsafe(no_mangle)]
pub extern "C" fn ane_draft_free() {
    *DRAFT_MODEL.lock().unwrap() = None;
}

// === Elementwise pipeline for Qwen3.5-4B ===
// Offload all norms, activations, residuals to ANE
// while GPU focuses on matmuls

static ELEM_PIPELINE: Mutex<Option<ElemPipeline>> = Mutex::new(None);

struct ElemPipeline {
    exec: crate::Executable,
    input: TensorData,
    output: TensorData,
}

/// Initialize elementwise pipeline for given model dimensions.
/// Compiles 8 layers of (RMSNorm + SiLU + residual) fused into one ANE graph.
#[unsafe(no_mangle)]
pub extern "C" fn ane_elem_init(dim: u32, num_fused_layers: u32) -> i32 {
    let dim = dim as usize;
    let nl = num_fused_layers as usize;
    let seq = 64;
    
    let mut g = Graph::new();
    let mut h = g.placeholder(Shape { channels: dim, height: 1, width: seq, batch: 1 });
    let neg_half = g.constant_with_scalar(-0.5, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let eps = g.constant_with_scalar(1e-6, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    
    for _ in 0..nl {
        let sq = g.multiplication(h, h);
        let m = g.reduce_mean(sq, 1);
        let me = g.addition(m, eps);
        let rms = g.power(me, neg_half);
        let n = g.multiplication(h, rms);
        h = g.addition(h, n);
        let sq2 = g.multiplication(h, h);
        let m2 = g.reduce_mean(sq2, 1);
        let me2 = g.addition(m2, eps);
        let rms2 = g.power(me2, neg_half);
        let n2 = g.multiplication(h, rms2);
        let sg = g.sigmoid(n2);
        let sl = g.multiplication(n2, sg);
        h = g.addition(h, sl);
    }
    
    let exec = match g.compile(NSQualityOfService::Default) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("ane_elem_init: compile failed: {e}");
            return -1;
        }
    };
    
    let input = TensorData::with_f32(&vec![0.01; dim * seq],
        Shape { channels: dim, height: 1, width: seq, batch: 1 });
    let output = TensorData::new(Shape { channels: dim, height: 1, width: seq, batch: 1 });
    
    // Warmup
    for _ in 0..3 { let _ = exec.run_cached_direct(&[&input], &[&output]); }
    
    *ELEM_PIPELINE.lock().unwrap() = Some(ElemPipeline { exec, input, output });
    0
}

/// Run one elementwise pass (N fused layers of norm+silu+residual).
#[unsafe(no_mangle)]
pub extern "C" fn ane_elem_run() -> i64 {
    let guard = ELEM_PIPELINE.lock().unwrap();
    let pipe = match guard.as_ref() {
        Some(p) => p,
        None => return -1,
    };
    let start = Instant::now();
    match pipe.exec.run_cached_direct(&[&pipe.input], &[&pipe.output]) {
        Ok(()) => start.elapsed().as_micros() as i64,
        Err(_) => -1,
    }
}

/// Benchmark N elementwise runs.
#[unsafe(no_mangle)]
pub extern "C" fn ane_elem_bench(n: u32) -> i64 {
    let guard = ELEM_PIPELINE.lock().unwrap();
    let pipe = match guard.as_ref() {
        Some(p) => p,
        None => return -1,
    };
    let start = Instant::now();
    for _ in 0..n { let _ = pipe.exec.run_cached_direct(&[&pipe.input], &[&pipe.output]); }
    (start.elapsed().as_micros() / n as u128) as i64
}

// === Prefill pipeline ===
static PREFILL_MODEL: Mutex<Option<PrefillModel>> = Mutex::new(None);

struct PrefillModel {
    exec: crate::Executable,
    input: TensorData,
    output: TensorData,
    dim: usize,
    seq: usize,
}

/// Compile a prefill kernel: 1 layer at seq_len=64
#[unsafe(no_mangle)]
pub extern "C" fn ane_prefill_init(dim: u32, hidden: u32) -> i32 {
    let dim = dim as usize;
    let hidden = hidden as usize;
    let seq = 64usize;
    
    let mut g = Graph::new();
    let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
    let q = g.inner_product(x, &vec![0.001; dim*dim], dim, dim);
    let o = g.inner_product(q, &vec![0.001; dim*dim], dim, dim);
    let h = g.addition(x, o);
    let gate = g.inner_product(h, &vec![0.001; hidden*dim], dim, hidden);
    let up = g.inner_product(h, &vec![0.001; hidden*dim], dim, hidden);
    let gs = g.sigmoid(gate);
    let gl = g.multiplication(gate, gs);
    let mix = g.multiplication(gl, up);
    let down = g.inner_product(mix, &vec![0.001; dim*hidden], hidden, dim);
    let _out = g.addition(h, down);
    
    let exec = match g.compile(NSQualityOfService::Default) {
        Ok(e) => e,
        Err(e) => { eprintln!("ane_prefill_init: {e}"); return -1; }
    };
    let input = TensorData::with_f32(&vec![0.01; dim*seq],
        Shape { batch: 1, channels: dim, height: 1, width: seq });
    let output = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });
    for _ in 0..3 { let _ = exec.run_cached_direct(&[&input], &[&output]); }
    *PREFILL_MODEL.lock().unwrap() = Some(PrefillModel { exec, input, output, dim, seq });
    0
}

/// Run one prefill chunk (64 tokens through 1 layer). Call 32x for full model.
#[unsafe(no_mangle)]
pub extern "C" fn ane_prefill_run() -> i64 {
    let guard = PREFILL_MODEL.lock().unwrap();
    let m = match guard.as_ref() { Some(m) => m, None => return -1 };
    let start = Instant::now();
    let _ = m.exec.run_cached_direct(&[&m.input], &[&m.output]);
    start.elapsed().as_micros() as i64
}

/// Benchmark N prefill runs.
#[unsafe(no_mangle)]
pub extern "C" fn ane_prefill_bench(n: u32) -> i64 {
    let guard = PREFILL_MODEL.lock().unwrap();
    let m = match guard.as_ref() { Some(m) => m, None => return -1 };
    let start = Instant::now();
    for _ in 0..n { let _ = m.exec.run_cached_direct(&[&m.input], &[&m.output]); }
    (start.elapsed().as_micros() / n as u128) as i64
}

// === Real-weight benchmark for apples-to-apples comparison ===
// Accepts actual model weights via pointer for fair Private API vs CoreML testing.

static REAL_MODEL: Mutex<Option<RealModel>> = Mutex::new(None);

struct RealModel {
    exec: crate::Executable,
    input: TensorData,
    output: TensorData,
    dim: usize,
    seq: usize,
}

/// Compile an FFN layer with REAL weights passed via pointer.
/// Layout: weights_ptr points to contiguous f32 data:
///   [gate_w: hidden*dim, up_w: hidden*dim, down_w: dim*hidden]
/// Total: 2*hidden*dim + dim*hidden = 3*hidden*dim floats.
/// Returns 0 on success, -1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn ane_real_init(
    dim: u32,
    hidden: u32,
    seq: u32,
    weights_ptr: *const f32,
    weights_len: u32,
) -> i32 {
    let dim = dim as usize;
    let hidden = hidden as usize;
    let seq = seq as usize;
    let expected = 3 * hidden * dim;
    if weights_ptr.is_null() || (weights_len as usize) < expected {
        eprintln!("ane_real_init: weights_len={} expected={}", weights_len, expected);
        return -1;
    }
    let weights = unsafe { std::slice::from_raw_parts(weights_ptr, expected) };

    let mut off = 0;
    let gate_w = &weights[off..off + hidden * dim]; off += hidden * dim;
    let up_w = &weights[off..off + hidden * dim]; off += hidden * dim;
    let down_w = &weights[off..off + dim * hidden];

    let mut g = Graph::new();
    let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
    let gate = g.inner_product(x, gate_w, dim, hidden);
    let up = g.inner_product(x, up_w, dim, hidden);
    let gs = g.sigmoid(gate);
    let gl = g.multiplication(gate, gs);
    let mix = g.multiplication(gl, up);
    let down = g.inner_product(mix, down_w, hidden, dim);
    let _out = g.addition(x, down);

    let exec = match g.compile(NSQualityOfService::Default) {
        Ok(e) => e,
        Err(e) => { eprintln!("ane_real_init: {e}"); return -1; }
    };
    let input = TensorData::with_f32(
        &vec![0.01; dim * seq],
        Shape { batch: 1, channels: dim, height: 1, width: seq },
    );
    let output = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });
    for _ in 0..5 { let _ = exec.run_cached_direct(&[&input], &[&output]); }
    *REAL_MODEL.lock().unwrap() = Some(RealModel { exec, input, output, dim, seq });
    0
}

/// Run one forward pass with real weights. Returns latency in microseconds.
#[unsafe(no_mangle)]
pub extern "C" fn ane_real_run() -> i64 {
    let guard = REAL_MODEL.lock().unwrap();
    let m = match guard.as_ref() { Some(m) => m, None => return -1 };
    let start = Instant::now();
    let _ = m.exec.run_cached_direct(&[&m.input], &[&m.output]);
    start.elapsed().as_micros() as i64
}

/// Benchmark N runs with real weights, return average microseconds.
#[unsafe(no_mangle)]
pub extern "C" fn ane_real_bench(n: u32) -> i64 {
    let guard = REAL_MODEL.lock().unwrap();
    let m = match guard.as_ref() { Some(m) => m, None => return -1 };
    let start = Instant::now();
    for _ in 0..n { let _ = m.exec.run_cached_direct(&[&m.input], &[&m.output]); }
    (start.elapsed().as_micros() / n as u128) as i64
}

/// Free real-weight model.
#[unsafe(no_mangle)]
pub extern "C" fn ane_real_free() {
    *REAL_MODEL.lock().unwrap() = None;
}

// === CoreML model loading (native, no Python needed at runtime) ===

static COREML_MODEL: Mutex<Option<CoreMLState>> = Mutex::new(None);

struct CoreMLState {
    model: crate::coreml::CoreMLModel,
    input_name: String,
    shape: Vec<usize>,
    seq: usize,
}

/// Load a CoreML .mlpackage model. path is a null-terminated C string.
/// seq is the sequence length used for benchmarking.
/// Returns 0 on success, -1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn coreml_load(path: *const std::ffi::c_char, seq: u32) -> i32 {
    let path_str = unsafe {
        match std::ffi::CStr::from_ptr(path).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return -1,
        }
    };
    let seq = seq as usize;

    match crate::coreml::CoreMLModel::load(&path_str) {
        Ok(model) => {
            // Default input shape for FFN: [1, seq, dim]
            // We'll discover dim from the first predict call
            *COREML_MODEL.lock().unwrap() = Some(CoreMLState {
                model,
                input_name: "x".to_string(),
                shape: vec![1, seq, 1], // placeholder, set properly via coreml_set_shape
                seq,
            });
            0
        }
        Err(e) => {
            eprintln!("coreml_load: {e}");
            -1
        }
    }
}

/// Set the input shape for the CoreML model.
#[unsafe(no_mangle)]
pub extern "C" fn coreml_set_shape(batch: u32, seq: u32, dim: u32) {
    if let Some(state) = COREML_MODEL.lock().unwrap().as_mut() {
        state.shape = vec![batch as usize, seq as usize, dim as usize];
        state.seq = seq as usize;
    }
}

/// Run one CoreML prediction. Returns latency in microseconds.
#[unsafe(no_mangle)]
pub extern "C" fn coreml_run() -> i64 {
    let guard = COREML_MODEL.lock().unwrap();
    let state = match guard.as_ref() {
        Some(s) => s,
        None => return -1,
    };
    match state.model.predict_once(&state.input_name, &state.shape) {
        Ok(us) => us,
        Err(e) => {
            eprintln!("coreml_run: {e}");
            -1
        }
    }
}

/// Benchmark N CoreML predictions. Returns average microseconds.
#[unsafe(no_mangle)]
pub extern "C" fn coreml_bench(n: u32) -> i64 {
    let guard = COREML_MODEL.lock().unwrap();
    let state = match guard.as_ref() {
        Some(s) => s,
        None => return -1,
    };
    match state.model.bench(&state.input_name, &state.shape, n) {
        Ok(us) => us,
        Err(e) => {
            eprintln!("coreml_bench: {e}");
            -1
        }
    }
}

/// Free CoreML model.
#[unsafe(no_mangle)]
pub extern "C" fn coreml_free() {
    *COREML_MODEL.lock().unwrap() = None;
}

/// Ping ANE with a trivially small 1x1 sigmoid graph.
/// Returns latency in microseconds, or -1 on failure.
/// Use this to prevent ANE from entering deep sleep.
#[unsafe(no_mangle)]
pub extern "C" fn ane_ping() -> i64 {
    let mut g = Graph::new();
    let x = g.placeholder(Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let _y = g.sigmoid(x);
    let exec = match g.compile(NSQualityOfService::Default) {
        Ok(e) => e,
        Err(_) => return -1,
    };
    let input = TensorData::with_f32(&[0.5], Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let output = TensorData::new(Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let start = Instant::now();
    let _ = exec.run_cached_direct(&[&input], &[&output]);
    start.elapsed().as_micros() as i64
}
