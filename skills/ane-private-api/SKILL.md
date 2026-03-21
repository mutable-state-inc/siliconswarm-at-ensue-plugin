---
name: ane-private-api
description: "Reference for the Apple Neural Engine private API via Rust bindings. Covers the graph builder, available ops, compilation, IOSurface I/O, and hardware constraints."
argument-hint: "[topic]  e.g. ops, inner_product, compilation, constraints, ffi"
allowed-tools: Read, Grep, Glob, Bash(*)
triggers:
  - ane api
  - private api
  - ane kernel
  - ane ops
  - rust ane
  - neural engine
---

# ANE Private API — Rust Bindings Reference

The Apple Neural Engine is accessed through `_ANEInMemoryModel` in the private `AppleNeuralEngine.framework`. The Rust crate at `ane_kernel/crates/ane/` provides typed bindings via `objc2`.

## Crate location

```
ane_kernel/crates/ane/src/
├── ffi.rs                  # FFI entry points (Go calls these via purego)
├── graph/
│   ├── mod.rs              # Graph struct, compile(), placeholder()
│   └── ops.rs              # All graph operations
├── client.rs               # MIL text generation → _ANEInMemoryModel compilation
├── executable.rs           # Executable::run_cached_direct() — the hot dispatch path
├── tensor_data.rs          # IOSurface-backed I/O buffers
├── ops/
│   ├── inner_product.rs    # InnerProductOp (constant-weight linear)
│   ├── mil.rs              # MIL code generation from ops
│   └── weights.rs          # f32 → fp16 weight blob encoding
├── coreml.rs               # CoreML MLModel loading (for comparison)
├── ane_in_memory_model.rs  # _ANEInMemoryModel objc2 bindings
├── ane_client.rs           # _ANEClient shared connection
└── io_surface.rs           # IOSurface allocation
```

## Lifecycle

```rust
let mut g = Graph::new();
let x = g.placeholder(Shape { batch: 1, channels: 2048, height: 1, width: 64 });
let y = g.inner_product(x, &weights, 2048, 11008);
let exec = g.compile(NSQualityOfService::Default)?;

let input = TensorData::with_f32(&data, shape);
let output = TensorData::new(out_shape);
exec.run_cached_direct(&[&input], &[&output])?;
```

## Graph operations

All ops are methods on `Graph`. Tensors are 4D NCHW: `Shape { batch, channels, height, width }`.

### Linear projections

| Op | Signature | Weight source | Performance |
|---|---|---|---|
| `inner_product` | `(src, &[f32], ic, oc) → Tensor` | Constant (baked into compiled model) | Fastest. Weights <16MB cached in ANE SRAM at 15000+ GB/s. |
| `matrix_multiplication` | `(x, y, transpose_x, transpose_y) → Tensor` | Dynamic (from input tensor via slice) | Slower. Requires reshape+transpose chain. |

`inner_product` converts f32 weights to fp16 at compile time. No runtime quantization.

### Convolutions

| Op | Signature | Notes |
|---|---|---|
| `convolution_2d_1x1(src, weights, bias)` | 1×1 conv with constant weights | Broken on macOS 26 (`_ANECompiler` removed) |
| `convolution_2d_1x1_dynamic(src, weights)` | 1×1 conv with dynamic weights | |
| `convolution_2d(src, weights, bias, desc)` | General 2D conv | `Convolution2dDescriptor` for stride/pad/groups |
| `convolution_transpose_2d(src, weights, bias, desc)` | Transposed conv | `ConvolutionTranspose2dDescriptor` |

### Elementwise binary

| Op | Signature |
|---|---|
| `addition(x, y)` | x + y |
| `subtraction(x, y)` | x - y |
| `multiplication(x, y)` | x * y |
| `division(x, y)` | x / y |
| `power(x, exp)` | x^exp |
| `maximum(x, y)` | max(x, y) |
| `minimum(x, y)` | min(x, y) |

### Elementwise unary

| Op | Signature |
|---|---|
| `absolute(x)` | |x| |
| `square_root(x)` | √x |
| `reciprocal_square_root(x)` | 1/√x — broken on macOS 26, use `power(x, -0.5)` |
| `exponent(x)` | e^x |
| `logarithm(x)` | ln(x) |
| `reciprocal(x)` | 1/x |

### Activations

| Op | Notes |
|---|---|
| `sigmoid(x)` | |
| `relu(x)` | |
| `tanh(x)` | |
| `leaky_relu(x, slope)` | |
| `elu(x, alpha)` | |
| `hard_sigmoid(x, alpha, beta)` | |
| `linear(x, alpha, beta)` | alpha*x + beta |
| `softplus(x)` | |
| `softsign(x)` | |

SiLU = `multiplication(x, sigmoid(x))` — no fused op.

### Pooling

| Op | Notes |
|---|---|
| `max_pool(x, kH, kW, sH, sW, pT, pB, pL, pR)` | |
| `avg_pool(x, kH, kW, sH, sW, pT, pB, pL, pR, exclude_pad)` | |
| `global_avg_pool(x)` | |

### Padding

| Op | Notes |
|---|---|
| `pad(x, pad_amounts, mode)` | `PadMode::Constant(val)`, `PadMode::Edge`, `PadMode::Reflect` |

### Tensor manipulation

| Op | Signature | Notes |
|---|---|---|
| `reshape(x, Shape)` | Change shape, same data | |
| `transpose(x, [4]perm)` | Permute NCHW dims | |
| `slice(x, begin, size)` | Extract sub-tensor | begin/size are `[4]usize` |
| `concat(&[Tensor], axis)` | Join along axis | |
| `flatten_2d(x)` | Flatten to [1, N, 1, 1] | |

### Reduction

| Op | Signature |
|---|---|
| `reduce_sum(x, axis)` | |
| `reduce_mean(x, axis)` | |
| `reduce_min(x, axis)` | |
| `reduce_max(x, axis)` | |

### Normalization

| Op | Notes |
|---|---|
| `instance_norm(x, params, channels, epsilon)` | Learnable params |
| `soft_max(x, axis)` | Softmax |

### Constants

| Op | Notes |
|---|---|
| `placeholder(Shape)` | Input tensor. Width must be ≥ 64. |
| `constant(&[f32], Shape)` | Constant tensor |
| `constant_with_scalar(val, Shape)` | Broadcast scalar |
| `constant_with_f16_bytes(&[u8], Shape)` | Constant from raw fp16 bytes |

## Hardware constraints (M1 Max)

| Constraint | Value | Impact |
|---|---|---|
| Min spatial width | 64 | Can't do seq=1. Must pad to 64. |
| Max fusion | 12 layers (dim=2048) | Larger models = fewer layers per dispatch |
| Max fusion | 24 layers (dim=896) | Smaller models fit more |
| SRAM cache | ~32 MB | Weights <16MB: 15000+ GB/s. Larger: ~51 GB/s from DRAM. |
| Dispatch overhead | 0.186ms | Per call to `run_cached_direct()` |
| Weight precision | fp16 only | `inner_product` converts f32→fp16 at compile time |
| Spatial limit (matmul) | 16384 width | For dynamic weights: seq + weight_slots ≤ 16384 |


## FFI entry points (current)

In `ane_kernel/crates/ane/src/ffi.rs`, called from Go:

| Function | Purpose |
|---|---|
| `ane_prefill_init(dim, hidden)` | Compile 1-layer FFN kernel |
| `ane_prefill_run()` | Run one dispatch, return µs |
| `ane_prefill_bench(n)` | Benchmark n dispatches, return avg µs |
| `ane_real_init(dim, hidden, seq, weights, len)` | Compile FFN with real weights |
| `ane_real_run()` | Run, return µs |
| `ane_real_bench(n)` | Benchmark, return avg µs |
| `ane_draft_init(dim, hidden, seq, layers)` | Compile multi-layer draft model |
| `ane_draft_run()` | Run, return µs |
| `coreml_load(path)` | Load CoreML .mlmodelc |
| `coreml_run()` | Run CoreML prediction, return µs |

## Build

```bash
make ane
```
