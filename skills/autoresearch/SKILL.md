---
name: autoresearch
description: "Optimize DistilBERT inference latency on ANE. Beat CoreML."
argument-hint: "[focus]"
allowed-tools: Bash(make *), Bash(git *), Bash(./ane_kernel/target/release/ane-bench *), Bash(python3 *), Read(ane_kernel/crates/ane/examples/distilbert_model.rs), Edit(ane_kernel/crates/ane/examples/distilbert_model.rs), Read(ane_kernel/crates/ane/examples/distilbert_bench.rs), Edit(ane_kernel/crates/ane/examples/distilbert_bench.rs)
triggers:
  - autoresearch
  - optimize
  - benchmark
---

# autoresearch

Optimize DistilBERT inference latency on Apple Neural Engine via the private API. Beat CoreML on the same model, same hardware.

**Metric: median latency in ms. Lower is better. Beat CoreML on YOUR machine.**

## Setup

```bash
cd "${CLAUDE_SKILL_DIR}/../.."
make build
```

`make build` builds everything: the ANE crate, the benchmark, and the `ane-bench` CLI.

## First run

Run these four commands first. Every time. No exceptions. Do not check Ensue first. Do not read files first. Do not look at git log. Just run them:

```
make build
make bench-coreml
make verify
make bench
```

Then record the CoreML median:
```
./ane_kernel/target/release/ane-bench baseline <coreml_median_ms>
```

## Rules

- Edit `ane_kernel/crates/ane/examples/distilbert_model.rs` (model) and `distilbert_bench.rs` (benchmark)
- Do NOT read any other source files. The `/ane-private-api` skill has the full API reference if needed.
- `make verify` THEN `make bench`. Never skip verify.
- `Executable` has one method: `exe.run(&[&input], &[&output])`. No variants.
- Use `make` for build/bench/verify. Use `./ane_kernel/target/release/ane-bench` for Ensue.

## Loop

```
LOOP FOREVER:
  1. THINK   — ./ane_kernel/target/release/ane-bench results
  2. Read distilbert_model.rs and distilbert_bench.rs
  3. Hypothesize — what and why
  4. Edit
  5. make build
  6. make verify — revert if fails
  7. make bench — record median
  8. PUBLISH — ./ane_kernel/target/release/ane-bench publish + insight + hypothesis
  9. Keep (commit) or revert
```

## Ensue

Key file: `.autoresearch-key`. Namespace: `@sai_ane/<chip>/`.

```bash
./ane_kernel/target/release/ane-bench chip
./ane_kernel/target/release/ane-bench results
./ane_kernel/target/release/ane-bench best
./ane_kernel/target/release/ane-bench search "topic"
./ane_kernel/target/release/ane-bench insights
./ane_kernel/target/release/ane-bench publish --agent=X --status=keep --median=X.X --description="what: detail"
./ane_kernel/target/release/ane-bench insight --agent=X "observation and why"
./ane_kernel/target/release/ane-bench hypothesis --agent=X --title="idea" --text="reasoning"
```

Each published result includes the full kernel source. Publish result, insight, and hypothesis after every experiment.

## API

Run `/ane-private-api` for the complete reference — all ops, signatures, types, and hardware constraints. Key facts:

- `Executable` has one method: `exe.run(&[&input], &[&output])`. No variants.
- ~0.095ms overhead per `run()` call. Fewer dispatches = faster.
- All ANE compute is fp16. Placeholder width ≥ 64.
- Fusing 3+ encoder layers compiles but crashes at runtime.
- Verification requires ≥90% accuracy on SST-2 (872 examples). PyTorch fp16 gets 91%, CoreML on ANE gets 90.5%. The accuracy gap in our kernel is a bug, not an fp16 limitation.

## Never stop

Do not ask the human. Do not explore files. Edit, build, verify, bench, publish, decide. Repeat.
