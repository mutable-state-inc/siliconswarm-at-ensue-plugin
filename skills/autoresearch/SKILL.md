---
name: autoresearch
description: "Optimize ANE inference latency for DistilBERT. Beat CoreML."
argument-hint: "[focus]"
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep, Agent
triggers:
  - autoresearch
  - optimize
  - benchmark
  - distilbert
---

# autoresearch

Optimize DistilBERT inference latency on Apple Neural Engine via the private `_ANEInMemoryModel` API. The target: beat Apple's CoreML on the same model, same hardware.

**The metric: median inference latency in milliseconds (lower is better).**

The baseline to beat is CoreML at **5.86ms** on M1 Max, running Apple's own ANE-optimized DistilBERT.

## Setup

```bash
cd "${CLAUDE_SKILL_DIR}/../.."
make build
```

## Editable files

**One file.** All experimentation happens here:

- `ane_kernel/crates/ane/examples/distilbert_bench.rs` — the kernel. Graph construction, layer fusion, IOSurface management, benchmark loop.

## Read-only files

Do not modify these:

- `ane_kernel/crates/ane/src/` — ANE bindings (the private API wrapper)
- `ane_kernel/crates/ane/examples/distilbert_verify.rs` — correctness tests
- `Makefile` — build/bench/verify targets

## Verification

**Every change must pass verification. No exceptions.**

```bash
make verify    # must exit 0 with "ALL TESTS PASSED"
```

If verify fails, the change is wrong. Revert. Do not benchmark. Do not commit.

## Benchmarking

```bash
make bench     # 1000 iterations, reports median/mean/p5/p95/min
```

Only run after `make verify` passes.

## The loop

```
1. Read distilbert_bench.rs — understand current state
2. Hypothesize — what change will reduce latency and why?
3. Implement — edit distilbert_bench.rs
4. make build — must compile
5. make verify — must pass (8/8 tests)
6. make bench — measure median latency
7. If improved: git add -A && git commit -m "<description>"
8. If not: git checkout -- ane_kernel/crates/ane/examples/distilbert_bench.rs
9. Goto 1
```

**The goal is simple: get the lowest median latency.** Everything in `distilbert_bench.rs` is fair game.

**Simplicity criterion**: all else being equal, simpler is better. Removing something and getting equal or better results is a win.

**The first run**: establish the baseline — run `make verify` then `make bench` without changing anything.

## Never stop

Once the loop begins, do not pause to ask the human. The human may be asleep. You are autonomous.

If you run out of ideas:
1. Re-read the kernel code and the ANE bindings for new levers
2. Profile — where is time actually spent?
3. Try radical restructuring, not just parameter tweaking
4. Read `ane_kernel/crates/ane/src/graph/ops.rs` for ops you haven't tried
