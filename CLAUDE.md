# autoresearch-ane-at-home

Rust bindings for Apple Neural Engine via reverse-engineered private API (`_ANEInMemoryModel`). Benchmarking direct ANE access vs Apple's CoreML on the same hardware.

## Quick start

```bash
make setup   # install Rust (if needed), compile, detect chip+RAM
make bench   # run synthetic benchmark
```

## Architecture

The `ane` crate (`ane_kernel/crates/ane/`) wraps Apple's private `AppleNeuralEngine.framework`:
- **Graph API** — build computation graphs with `inner_product`, `matrix_multiplication`, activations, reductions, etc.
- **Compile** — `graph.compile()` produces an `Executable` via `_ANEInMemoryModel`
- **Run** — `executable.run()` executes on ANE with IOSurface zero-copy I/O

### Key files

| File | What |
|------|------|
| `ane_kernel/crates/ane/src/graph/ops.rs` | All ANE graph operations |
| `ane_kernel/crates/ane/src/executable.rs` | Compile & run lifecycle |
| `ane_kernel/crates/ane/src/graph/compile.rs` | Graph → MIL → ANE machine code |
| `ane_kernel/crates/ane/src/coreml.rs` | CoreML bindings for comparison |
| `ane_kernel/crates/ane/examples/bench.rs` | Synthetic benchmark |

## ANE private API ops

`inner_product`, `matrix_multiplication`, `convolution_2d`, `convolution_transpose_2d`, `relu`, `tanh`, `sigmoid`, `leaky_relu`, `elu`, `hard_sigmoid`, `softplus`, `softsign`, `addition`, `subtraction`, `multiplication`, `division`, `power`, `maximum`, `minimum`, `absolute`, `square_root`, `reciprocal_square_root`, `exponent`, `logarithm`, `reciprocal`, `soft_max`, `instance_norm`, `reduce_sum`, `reduce_mean`, `reduce_min`, `reduce_max`, `concat`, `slice`, `reshape`, `transpose`, `flatten_2d`, `pad`, `max_pool`, `avg_pool`, `global_avg_pool`

## Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4)
- Rust (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)

## Authors
- [svv232](https://github.com/svv232)

## License

MIT
