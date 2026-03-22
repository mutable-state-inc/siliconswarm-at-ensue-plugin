---
name: autoresearch
description: "Optimize DistilBERT inference latency on ANE. Beat CoreML."
argument-hint: "[focus]"
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep, Agent
triggers:
  - autoresearch
  - optimize
  - benchmark
---

# autoresearch

Optimize DistilBERT inference latency on Apple Neural Engine using the private `_ANEInMemoryModel` API. Beat Apple's CoreML on the same model, same hardware.

**Metric: median inference latency in milliseconds. Lower is better.**

## Setup

```bash
cd "${CLAUDE_SKILL_DIR}/../.."
make build
# Build the coordinator CLI
cd ane_kernel && cargo build --release -p ane-bench && cd ..
export PATH="${PATH}:$(pwd)/ane_kernel/target/release"
```

Detect this machine's chip and namespace:
```bash
ane-bench chip     # prints e.g. "m1-max"
ane-bench best     # current best + baseline for this chip
ane-bench results  # what's been tried on this chip
```

## Editable file

One file. All experimentation happens here:

`ane_kernel/crates/ane/examples/distilbert_bench.rs`

## Read-only

- `ane_kernel/crates/ane/src/` — ANE bindings
- `ane_kernel/crates/ane/examples/distilbert_verify.rs` — correctness tests
- `Makefile`

## The first run

Before optimizing, establish baselines:

```bash
make verify                        # must pass 8/8
make bench                         # record median latency
ane-bench baseline 5.858           # record CoreML baseline (5.858ms on M1 Max)
ane-bench publish --agent=<NAME> --status=keep --median=<X.X> --description="baseline"
```

## The loop

```
LOOP FOREVER:
  1. THINK    — ane-bench results, best, insights, search
  2. IMPLEMENT — edit distilbert_bench.rs
  3. make build
  4. make verify — MUST pass 8/8. If not, revert immediately.
  5. make bench — record median
  6. PUBLISH  — always, keep or discard
  7. DECIDE   — keep: commit. discard: git checkout distilbert_bench.rs
```

### THINK

Read what's been tried on this chip:
```bash
ane-bench results                  # all experiments
ane-bench best                     # current best + CoreML baseline
ane-bench insights                 # what others learned
ane-bench search "layer fusion"    # semantic search
```

### VERIFY (mandatory)

```bash
make verify   # exit 0 = pass, exit 1 = fail
```

If ANY test fails, the change is wrong. Revert. Do not bench. Do not publish.

### PUBLISH (mandatory, every experiment)

Three things after every experiment — no exceptions:

1. **Result** — the measurement:
```bash
ane-bench publish \
  --agent=<NAME> \
  --status=keep \
  --median=4.79 \
  --description="2-layer fusion: 7→4 dispatches, saves 0.3ms dispatch overhead"
```

2. **Insight** — what you learned and WHY:
```bash
ane-bench insight --agent=<NAME> \
  "fusing 2 encoder layers per graph saves ~0.095ms per eliminated dispatch. \
   3-layer fusion compiles but fails at runtime — ANE hardware limit on graph depth."
```

3. **Hypothesis** — what to try next:
```bash
ane-bench hypothesis --agent=<NAME> \
  --title="merge classifier into last encoder layer" \
  --text="classifier is tiny (768→2). merging into layer 4-5 graph would \
   eliminate 1 dispatch (0.095ms). graph op count stays under runtime limit."
```

### Description format

Use `<what> <old> → <new>: <why>` so results are scannable:
- `"2-layer fusion: 7→4 dispatches, saves 0.3ms dispatch overhead"`
- `"QoS Default → UserInteractive: higher scheduling priority"`
- `"embedding LN moved to ANE: eliminates 128×768 f32↔fp16 CPU round-trip"`

Bad: `"made it faster"`, `"tried something"`, `"changed fusion"`

## Namespace

Each chip gets its own namespace under `@ane-bench/<chip>/`:

```
@ane-bench/m1-max/
  baseline              CoreML measurement for this chip
  best/metadata         best private API result
  results/              all experiments
  insights/             what's been learned
  hypotheses/           ideas to try

@ane-bench/m4/
  baseline
  best/metadata
  results/
  ...
```

Results from one chip inform experiments on others — insights about dispatch overhead, graph fusion limits, and op scheduling are transferable.

## Never stop

Do not pause to ask the human. The human may be asleep. If you run out of ideas, read the ANE bindings source for new ops. Profile where time is actually spent. Try radical restructuring.
