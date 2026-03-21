---
name: autoresearch
description: "Optimize local inference tok/s by building ANE kernels with the private API. Benchmark, keep or revert."
argument-hint: "[focus]"
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep, Agent
triggers:
  - autoresearch
  - optimize
  - tok/s
---

# autoresearch

Prove that the ANE private API improves inference tok/s on a coding agent workload (Qwen3.5-4B-4bit, ~2000 token prompt).

Before your first hypothesis, run `/ane-private-api` to understand the ANE private API — what ops exist, how the graph builder works, hardware constraints.

## Setup

All commands run from the plugin repo root (where the Makefile is).

```bash
cd "${CLAUDE_SKILL_DIR}/../.."
make build
```

## You may ONLY edit these files

1. `ane_kernel/crates/ane/src/ffi.rs` — Rust ANE kernel. Every hypothesis starts here.
2. `ane_draft.go` — Register and call new FFI functions from the Rust kernel.
3. `harness.go` — ONLY to wire ANE calls into the inference pipeline. Do not change cache type, warmup, env vars, or anything unrelated to calling the ANE kernel.

Do not edit `experiment.go`, `bench_test.go`, `bench_ane_test.go`, anything in `decode/`, anything in the `mlx-go` repo.

## Loop

1. Hypothesize — run `/ane-private-api` if you need the API reference. Only read files you can edit.
2. Implement — no debug prints, no diagnostic tests
3. Build — `make build`. One fix attempt if it fails, then revert.
4. Commit — `git add -A && git commit -m "<description>"`
5. Run — `make bench`
6. Publish — `./autoresearch-cli publish --agent=<NAME> --status=<keep|discard|crash> --description="<what>"`
7. tok/s improved → keep. Else → `git reset --hard HEAD~1`
8. Goto 1

Never stop. Never ask the human.
