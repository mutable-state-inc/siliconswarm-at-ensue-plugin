---
name: ane-private-api
description: "Complete reference for the Apple Neural Engine private API via Rust bindings."
argument-hint: "[topic]"
allowed-tools: ""
triggers:
  - ane api
  - private api
  - ane ops
  - neural engine
---

# ANE Private API

Rust bindings to Apple's private `AppleNeuralEngine.framework` via `_ANEInMemoryModel`. Build computation graphs, compile to ANE machine code, execute on dedicated neural engine hardware with IOSurface zero-copy I/O.

Everything is below. Do not read source files.

## Lifecycle

```rust
let mut g = Graph::new();
let x = g.placeholder(shape);
let y = g.inner_product(x, &weights, in_ch, out_ch);
let exe = g.compile(NSQualityOfService::UserInteractive)?;

let input = TensorData::new(shape);
let output = TensorData::new(out_shape);
exe.run(&[&input], &[&output])?;
let data = output.as_f32_slice();
```

## Types

**Shape** — 4D NCHW:
```rust
Shape { batch: usize, channels: usize, height: usize, width: usize }
Shape::spatial(channels, height, width)   // batch=1
Shape::channels(c)                        // [1, c, 1, 1]
```

**TensorData** — IOSurface buffer (fp16 hardware, f32 API):
```rust
TensorData::new(shape) → TensorData
TensorData::with_f32(&[f32], shape) → TensorData
.as_f32_slice() → LockedSlice            // RAII read lock, fp16→f32
.as_f32_slice_mut() → LockedSliceMut     // RAII write lock, f32→fp16 on drop
.copy_from_f32(&[f32])                    // bulk write
.write_f32_at(index: usize, value: f32)  // single indexed write
.write_f32_sparse(&[usize], &[f32])      // batch indexed write
.read_f32() → Box<[f32]>                // allocating copy
.shape() → Shape
```

**Executable** — compiled ANE program:
```rust
exe.run(&[&TensorData], &[&TensorData]) → Result<(), Error>
exe.run_cached(&[&TensorData], &[&TensorData]) → Result<(), Error>
exe.run_cached_with_stats(&[&TensorData], &[&TensorData]) → Result<u64, Error>
exe.run_cached_direct(&[&TensorData], &[&TensorData]) → Result<(), Error>
```

- `run` — standard execution. Creates a new `_ANERequest` each call.
- `run_cached` — caches the ANE request object after first call. Saves ~0.095ms per dispatch. **Must pass the same TensorData objects every call** (contents can change, objects must be the same).
- `run_cached_with_stats` — same as `run_cached` but returns `hw_execution_time_ns`: actual nanoseconds spent on ANE hardware, excluding XPC/dispatch overhead. Use this to understand where time is really going.
- `run_cached_direct` — XPC bypass via `_ANEClient.doEvaluateDirectWithModel`. Skips the ANE daemon entirely. Same caching semantics as `run_cached`.

**Tensor** — graph node handle returned by all ops. Not data.

**PadMode** — `Valid`, `Same`

**PadFillMode** — `Constant`, `Reflect`, `Replicate`

## Graph Operations

All methods on `&mut Graph`. All return `Tensor`.

### Inputs & constants

| Op | Signature |
|---|---|
| `placeholder` | `(Shape) → Tensor` — runtime input. **Width ≥ 64.** |
| `constant` | `(&[f32], Shape) → Tensor` — compile-time, stored fp16 |
| `constant_with_scalar` | `(f32, Shape) → Tensor` — broadcast scalar |
| `constant_with_f16_bytes` | `(&[u8], Shape) → Tensor` — raw fp16 |

### Linear projections

| Op | Signature |
|---|---|
| `inner_product` | `(source, &[f32] weights, input_channels, output_channels) → Tensor` — constant-weight linear. Weights `[out, in]` row-major, baked as fp16. |
| `matrix_multiplication` | `(x, y, transpose_x: bool, transpose_y: bool) → Tensor` — dynamic matmul between runtime tensors. |

### Elementwise binary

| Op | Signature |
|---|---|
| `addition` | `(Tensor, Tensor) → Tensor` |
| `subtraction` | `(Tensor, Tensor) → Tensor` |
| `multiplication` | `(Tensor, Tensor) → Tensor` |
| `division` | `(Tensor, Tensor) → Tensor` |
| `power` | `(Tensor, Tensor) → Tensor` |
| `maximum` | `(Tensor, Tensor) → Tensor` |
| `minimum` | `(Tensor, Tensor) → Tensor` |

All broadcast: output shape is `max(left, right)` per dimension.

### Elementwise unary

| Op | Signature |
|---|---|
| `absolute` | `(Tensor) → Tensor` |
| `square_root` | `(Tensor) → Tensor` |
| `reciprocal_square_root` | `(Tensor) → Tensor` |
| `exponent` | `(Tensor) → Tensor` |
| `logarithm` | `(Tensor) → Tensor` |
| `reciprocal` | `(Tensor) → Tensor` |

### Activations

| Op | Signature |
|---|---|
| `relu` | `(Tensor) → Tensor` |
| `sigmoid` | `(Tensor) → Tensor` |
| `tanh` | `(Tensor) → Tensor` |
| `leaky_relu` | `(Tensor, negative_slope: f64) → Tensor` |
| `elu` | `(Tensor, alpha: f64) → Tensor` |
| `hard_sigmoid` | `(Tensor, alpha: f64, beta: f64) → Tensor` |
| `linear` | `(Tensor, alpha: f64, beta: f64) → Tensor` |
| `softplus` | `(Tensor) → Tensor` |
| `softsign` | `(Tensor) → Tensor` |

No fused GELU or SiLU. Compose from primitives.

### Tensor manipulation

| Op | Signature |
|---|---|
| `reshape` | `(Tensor, Shape) → Tensor` — same element count |
| `transpose` | `(Tensor, [usize; 4]) → Tensor` — permute NCHW |
| `slice` | `(Tensor, begin: [usize; 4], size: [usize; 4]) → Tensor` |
| `concat` | `(&[Tensor], axis: usize) → Tensor` — 0=N 1=C 2=H 3=W |
| `flatten_2d` | `(Tensor) → Tensor` — collapse to `[1, total, 1, 1]` |

### Reduction

| Op | Signature |
|---|---|
| `reduce_sum` | `(Tensor, axis: i64) → Tensor` |
| `reduce_mean` | `(Tensor, axis: i64) → Tensor` |
| `reduce_min` | `(Tensor, axis: i64) → Tensor` |
| `reduce_max` | `(Tensor, axis: i64) → Tensor` |

### Normalization

| Op | Signature |
|---|---|
| `soft_max` | `(Tensor, axis: i64) → Tensor` — use -1 for last dim |
| `instance_norm` | `(source, params: Tensor, epsilon: f64) → Tensor` |

### Convolution

| Op | Signature |
|---|---|
| `convolution_2d` | `(source, weights: Tensor, bias: Option<Tensor>, &Convolution2dDescriptor) → Tensor` |
| `convolution_2d_1x1` | `(source, weights: Tensor, bias: Option<Tensor>) → Tensor` |
| `convolution_2d_1x1_dynamic` | `(source, weights: Tensor) → Tensor` — dynamic-weight |
| `convolution_transpose_2d` | `(source, weights: Tensor, bias: Option<Tensor>, &ConvolutionTranspose2dDescriptor) → Tensor` |

```rust
Convolution2dDescriptor { groups: usize, pad_mode: PadMode }
ConvolutionTranspose2dDescriptor { groups: usize, stride_height: usize, stride_width: usize, pad_mode: PadMode }
```

### Pooling

| Op | Signature |
|---|---|
| `max_pool` | `(Tensor, kH, kW, stride_h, stride_w, PadMode) → Tensor` |
| `avg_pool` | `(Tensor, kH, kW, stride_h, stride_w, PadMode) → Tensor` |
| `global_avg_pool` | `(Tensor) → Tensor` — output `[1, C, 1, 1]` |

### Padding

| Op | Signature |
|---|---|
| `pad` | `(Tensor, top, bottom, left, right, PadFillMode, value: f64) → Tensor` |

## Hardware

| Property | Value |
|---|---|
| Compute precision | fp16 only |
| Dispatch overhead | ~0.095ms per `run()` (XPC/IOKit) |
| SRAM cache | ~32MB. Weights <16MB: ~15000 GB/s. Larger: ~51 GB/s DRAM. |
| Placeholder width | ≥ 64. Pad shorter sequences. |
| Graph depth limit | ~60 ops compiles. 2 fused transformer layers work. 3 compiles but crashes at runtime. |
| Weight layout | `inner_product`: `[out_channels, in_channels]` row-major (PyTorch `nn.Linear` convention) |
| QoS | `UserInteractive` = lowest latency. `Default` slightly slower. |
| Data layout | NCHW: `data[b*C*H*W + c*H*W + h*W + w]` |
