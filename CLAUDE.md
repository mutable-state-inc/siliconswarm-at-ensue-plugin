# autoresearch-ane-at-home

LLM inference on Apple Neural Engine via reverse-engineered private API (`_ANEInMemoryModel`). Rust bindings to ANE hardware running GPT-2 Large (774M params, f32) at ~9.4 tok/s — faster than equivalent Metal GPU shaders for models up to ~1B params.

## Quick start

```bash
make chat    # interactive GPT-2 chat on ANE
make bench   # synthetic GPU+ANE pipeline benchmark
make build   # just compile
```

First run downloads GPT-2 Large (~3GB) from HuggingFace and compiles 72 ANE executables (~30s).

## Architecture

The `ane` crate (`ane_kernel/crates/ane/`) wraps Apple's private `AppleNeuralEngine.framework`:
- **Graph API** — build computation graphs with `inner_product`, `matrix_multiplication`, activations, reductions, etc.
- **Compile** — `graph.compile()` produces an `Executable` via `_ANEInMemoryModel`
- **Run** — `executable.run()` executes on ANE with IOSurface zero-copy I/O

### Key files

| File | What |
|------|------|
| `ane_kernel/crates/ane/examples/gpt2_ane.rs` | Single-file GPT-2 Large: download, compile, chat loop |
| `ane_kernel/crates/ane/src/graph/ops.rs` | All ANE graph operations |
| `ane_kernel/crates/ane/src/executable.rs` | Compile & run lifecycle |
| `ane_kernel/crates/ane/src/graph/compile.rs` | Graph → MIL → ANE machine code |
| `ane_kernel/crates/ane/examples/bench.rs` | Synthetic GPU+ANE concurrent dispatch benchmark |

## ANE private API ops

`inner_product`, `matrix_multiplication`, `convolution_2d`, `convolution_transpose_2d`, `relu`, `tanh`, `sigmoid`, `leaky_relu`, `elu`, `hard_sigmoid`, `softplus`, `softsign`, `addition`, `subtraction`, `multiplication`, `division`, `power`, `maximum`, `minimum`, `absolute`, `square_root`, `reciprocal_square_root`, `exponent`, `logarithm`, `reciprocal`, `soft_max`, `instance_norm`, `reduce_sum`, `reduce_mean`, `reduce_min`, `reduce_max`, `concat`, `slice`, `reshape`, `transpose`, `flatten_2d`, `pad`, `max_pool`, `avg_pool`, `global_avg_pool`

## Plugin skills

| Skill | Invocation | What it does |
|-------|-----------|-------------|
| Setup | `/ane-setup` | Clone mlx-go repos, build tools, detect chip |
| Autoresearch | `/autoresearch [focus]` | Autonomous inference optimization loop |

## Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4)
- Rust (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)

## Authors
- [svv232](https://github.com/svv232)

## License

MIT
