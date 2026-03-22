---
name: autoresearch
description: "Beat Apple's CoreML DistilBERT benchmark using the ANE private API. Optimize inference latency."
argument-hint: "[focus]"
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep, Agent
triggers:
  - autoresearch
  - optimize
  - benchmark
  - distilbert
---

# autoresearch — Beat CoreML on DistilBERT

## Goal

Beat Apple's published DistilBERT ANE benchmark using the private `_ANEInMemoryModel` API instead of CoreML.

**The metric: inference latency in milliseconds (lower is better).**

| Baseline | Latency | Source |
|----------|---------|--------|
| Apple published (iPhone 13) | 3.47ms | [Apple ML Research](https://machinelearning.apple.com/research/neural-engine-transformers) |
| CoreML on M1 Max (measured) | 5.86ms | Our benchmark of Apple's `DistilBERT_fp16.mlpackage` |
| **Our private API (target)** | **< 5.86ms** | `_ANEInMemoryModel` direct ANE access |

## Model

**DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`) — sentiment classifier.

- 6 transformer layers, dim=768, 12 heads, head_dim=64
- Encoder-only (no autoregressive decode, no KV cache)
- Sequence length 128, batch size 1
- LayerNorm, GELU activation, absolute position embeddings
- 66M parameters, fp16
- Same architecture as GPT-2 (our proven working kernel)

Apple's pre-built CoreML model: [apple/ane-distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/apple/ane-distilbert-base-uncased-finetuned-sst-2-english)

## Why we can win

CoreML adds overhead that the private API skips:
- **Scheduler** — CoreML decides CPU/GPU/ANE per op. We force ANE directly.
- **Buffer management** — CoreML manages IOSurfaces through abstractions. We use raw IOSurfaces.
- **Model loading** — CoreML compiles `.mlpackage` at load time. We pre-compile MIL in memory.
- **Per-inference overhead** — CoreML's `predict()` has Python→ObjC→ANE indirection.

Estimated breakdown of CoreML's 5.86ms:
```
CoreML runtime overhead:    ~1-2ms (scheduling, buffer mgmt, Python bridge)
IOSurface conversion:       ~0.5ms
Actual ANE compute:         ~3-4ms
Result readback:            ~0.5ms
```

If we eliminate 1-2ms of CoreML overhead → **~4ms**, beating CoreML by 30%.

## Setup

```bash
cd "${CLAUDE_SKILL_DIR}/../.."
make build
```

CoreML baseline (requires Python with coremltools):
```bash
python3 benchmark_coreml.py  # measures CoreML latency
```

## What to build

A single Rust example: `ane_kernel/crates/ane/examples/distilbert_bench.rs`

This file should:
1. Load DistilBERT weights from safetensors (HuggingFace)
2. Compile the 6-layer transformer encoder as ANE graphs via `_ANEInMemoryModel`
3. Run 1000 inferences at sequence length 128
4. Report mean/median/p95 latency in milliseconds
5. Compare against CoreML baseline

### Architecture (what to implement)

```
Input: [1, 128] token IDs
  → Embedding lookup (CPU: token + position → [768, 1, 128])
  → 6x Transformer encoder layer (ANE):
      LayerNorm → Q/K/V projection → attention → output projection → residual
      → LayerNorm → FC up (768→3072) → GELU → FC down (3072→768) → residual
  → Pooling (take [CLS] token)
  → Classifier head (768 → 2)
Output: [positive, negative] logit scores
```

### Key optimizations to try

1. **Merge layers** — fuse attention+FFN into one ANE graph per layer (fewer dispatches)
2. **Fuse multiple layers** — if ANE compiler allows, 2-3 layers per graph
3. **Pre-allocate IOSurfaces** — reuse between inferences, avoid allocation
4. **Minimize f32↔fp16 conversion** — write/read fp16 directly where possible
5. **QoS tuning** — try `NSQualityOfService::UserInteractive` vs `Default`

### Available ANE ops (from the `ane` crate)

```
inner_product, matrix_multiplication, convolution_2d,
addition, subtraction, multiplication, division, power,
relu, tanh, sigmoid, gelu (via tanh approximation),
soft_max, reduce_sum, reduce_mean,
concat, slice, reshape, transpose, flatten_2d,
pad, max_pool, avg_pool, global_avg_pool
```

## Measurement

```rust
// Warmup
for _ in 0..50 { exe.run(&inputs, &outputs).unwrap(); }

// Benchmark
let mut times = Vec::new();
for _ in 0..1000 {
    let start = std::time::Instant::now();
    exe.run(&inputs, &outputs).unwrap();
    times.push(start.elapsed().as_micros());
}
times.sort();
let median = times[times.len() / 2];
println!("Median: {:.3}ms", median as f64 / 1000.0);
```

## The loop

```
1. HYPOTHESIS — what change will reduce latency?
2. IMPLEMENT — edit distilbert_bench.rs
3. BUILD — make build
4. MEASURE — cargo run --release --example distilbert_bench
5. COMPARE — is median latency < 5.86ms?
6. KEEP or REVERT
```
