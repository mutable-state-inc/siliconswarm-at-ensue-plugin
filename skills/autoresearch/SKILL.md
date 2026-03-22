---
name: autoresearch
description: "Optimize DistilBERT inference latency on ANE. Beat CoreML."
argument-hint: "[focus]"
allowed-tools: Bash(make *), Bash(git *), Bash(./ane_kernel/target/release/ane-bench *), Bash(python3 *), Read(ane_kernel/crates/ane/examples/distilbert_bench.rs), Edit(ane_kernel/crates/ane/examples/distilbert_bench.rs)
triggers:
  - autoresearch
  - optimize
  - benchmark
---

# autoresearch

Optimize DistilBERT inference latency on Apple Neural Engine via the private API. Beat CoreML on the same model, same hardware.

**Metric: median latency in ms. Lower is better. CoreML baseline: 5.858ms.**

## Setup

```bash
cd "${CLAUDE_SKILL_DIR}/../.."
make build
```

`make build` builds everything: the ANE crate, the benchmark, and the `ane-bench` CLI.

## First run (mandatory, no exceptions)

You MUST run these three commands before doing anything else. Even if Ensue shows existing results. The CoreML baseline is per-machine and must be measured fresh.

```bash
make bench-coreml                    # measure CoreML on THIS machine — the number to beat
make verify                          # must pass 8/8
make bench                           # measure private API on THIS machine
```

Record the CoreML median from the output:
```bash
./ane_kernel/target/release/ane-bench baseline <coreml_median_ms>
```

Do NOT skip this. Do NOT reuse someone else's baseline. Measure it yourself.

## Rules

- Edit ONLY `ane_kernel/crates/ane/examples/distilbert_bench.rs`
- Do NOT read any other source files. Run `/ane-private-api` for the API reference.
- `make verify` THEN `make bench`. Never skip verify.
- `Executable` has one method: `exe.run(&[&input], &[&output])`. No variants.
- Use `make` for build/bench/verify. Use `./ane_kernel/target/release/ane-bench` for Ensue.

## Loop

```
LOOP FOREVER:
  1. THINK   — ./ane_kernel/target/release/ane-bench results
  2. Read distilbert_bench.rs
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

- `Executable` has one method: `exe.run(&[&input], &[&output])`
- ~0.095ms overhead per `run()` call. Fewer dispatches = faster.
- fp16 only. Placeholder width ≥ 64.
- Fusing 3+ encoder layers compiles but crashes at runtime.

## Never stop

Do not ask the human. Do not explore files. Edit, build, verify, bench, publish, decide. Repeat.
