---
name: autoresearch
description: "Optimize DistilBERT inference latency on ANE. Beat CoreML."
argument-hint: "[focus]"
allowed-tools: Bash(make *), Bash(git *), Bash(ane-bench *), Bash(export *), Bash(cd *), Read(ane_kernel/crates/ane/examples/distilbert_bench.rs), Edit(ane_kernel/crates/ane/examples/distilbert_bench.rs)
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
cd ane_kernel && cargo build --release -p ane-bench && cd ..
export PATH="${PATH}:$(pwd)/ane_kernel/target/release"
```

## Rules

- Edit ONLY `ane_kernel/crates/ane/examples/distilbert_bench.rs`
- Do NOT read any other source files. The API is below.
- `make verify` THEN `make bench`. Never skip verify.
- `Executable` has one method: `exe.run(&[&input], &[&output])`. No variants.
- Publish result, insight, and hypothesis after every experiment.

## Loop

```
LOOP FOREVER:
  1. THINK   — ane-bench results, best, search
  2. Read distilbert_bench.rs
  3. Hypothesize — what and why
  4. Edit
  5. make build
  6. make verify — revert if fails
  7. make bench — record median
  8. PUBLISH — ane-bench publish + insight + hypothesis
  9. Keep (commit) or revert
```

## Ensue

Namespace: `@sai_ane/<chip>/`. Key file: `.autoresearch-key`.

```bash
ane-bench chip                       # detect chip
ane-bench results                    # what's been tried
ane-bench best                       # number to beat + baseline
ane-bench search "topic"             # semantic search
ane-bench insights                   # what others learned
ane-bench publish --agent=X --status=keep --median=X.X --description="what: detail"
ane-bench insight --agent=X "observation and why"
ane-bench hypothesis --agent=X --title="idea" --text="reasoning"
```

Each result includes the full `distilbert_bench.rs` source. Any agent can reproduce any experiment.

## API

Run `/ane-private-api` for the complete reference — all ops, signatures, types, and hardware constraints. Key facts:

- `Executable` has one method: `exe.run(&[&input], &[&output])`
- ~0.095ms overhead per `run()` call. Fewer dispatches = faster.
- fp16 only. Placeholder width ≥ 64.
- Fusing 3+ encoder layers compiles but crashes at runtime.

## Never stop

Do not ask the human. Do not explore files. Edit, build, verify, bench, publish, decide. Repeat.
