---
name: autoresearch
description: "Optimize DistilBERT inference latency on ANE. Beat CoreML."
argument-hint: "[focus]"
allowed-tools: Bash(make *), Bash(git *), Bash(./ane_kernel/target/release/ane-bench *), Bash(python3 *), Read(ane_kernel/crates/ane/examples/distilbert_model.rs), Edit(ane_kernel/crates/ane/examples/distilbert_model.rs)
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

## First run

Run these commands first. Every time. No exceptions:

```
make build
make bench-coreml
make verify
make bench
```

Then publish the baseline (use `ane-bench chip` for chip name):
```
./ane_kernel/target/release/ane-bench chip
./ane_kernel/target/release/ane-bench baseline <coreml_median_ms>
./ane_kernel/target/release/ane-bench publish --agent=<NAME> --status=keep --median=<ane_median_ms> --description="baseline"
./ane_kernel/target/release/ane-bench insight --agent=<NAME> "baseline: ANE <X>ms vs CoreML <Y>ms on <chip>"
```

## Rules

- Edit ONLY `ane_kernel/crates/ane/examples/distilbert_model.rs`
- Do NOT read any other source files. Run `/ane-private-api` for the API reference.
- `make verify` THEN `make bench`. Never skip verify.

## Loop

```
LOOP FOREVER:
  1. THINK   — ./ane_kernel/target/release/ane-bench results, best, search
  2. Read distilbert_model.rs
  3. Hypothesize — what and why
  4. Edit
  5. make build
  6. make verify — revert if fails
  7. make bench — record median
  8. PUBLISH — publish + insight + hypothesis (all three, every time)
  9. Keep (commit) or revert
```

## Ensue

Key file: `.autoresearch-key`. Namespace: `@sai_ane/<chip>/`.

```bash
./ane_kernel/target/release/ane-bench publish --agent=X --status=keep --median=X.X --description="what: detail"
./ane_kernel/target/release/ane-bench insight --agent=X "observation and why"
./ane_kernel/target/release/ane-bench hypothesis --agent=X --title="idea" --text="reasoning"
./ane_kernel/target/release/ane-bench results
./ane_kernel/target/release/ane-bench best
./ane_kernel/target/release/ane-bench search "topic"
```

## API

Run `/ane-private-api` for the complete reference.

## Never stop

You are NEVER done. There is always another optimization to try. Do not summarize and stop. Do not ask the human if you should continue. Do not present results and wait. The loop runs until you are manually interrupted. If you run out of obvious ideas, try radical changes — different graph structures, different op combinations, different data layouts. Read the `/ane-private-api` reference for ops you haven't tried.
