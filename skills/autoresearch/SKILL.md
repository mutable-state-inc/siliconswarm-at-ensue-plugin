---
name: autoresearch
description: "Optimize local inference tok/s by building ANE+GPU kernels. Benchmark, keep or revert."
argument-hint: "[focus]"
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep, Agent
triggers:
  - autoresearch
  - optimize
  - tok/s
---

# autoresearch

Prove that the ANE private API improves inference tok/s on a coding agent workload (Qwen3.5-4B-4bit, ~2000 token prompt). For the ANE private API reference, use `/ane-private-api`.

## Setup

```bash
cd "$(go env GOPATH)/src/github.com/tmc/autoresearch-mlx-go-ane"
export PATH="${PATH}:$(go env GOPATH)/bin"
make build
```

## You may ONLY edit these files

1. `ane_kernel/crates/ane/src/ffi.rs` — **PRIMARY.** Rust ANE kernel. Build new FFI functions, new graph ops, new kernel architectures. This is where the ANE private API work happens.
2. `ane_draft.go` — Register and call FFI functions from the Rust kernel. Wire ANE into the Go pipeline.
3. `harness.go` — GPU pipeline. Route work to ANE, integrate ANE results.

Every hypothesis should involve the Rust kernel or how it's called. Pure Go changes without ANE involvement are not the point of this experiment.

Everything else is off limits. Do not edit `experiment.go`, `bench_test.go`, `bench_ane_test.go`, anything in `decode/`, anything in the `mlx-go` repo.

Reference: `ane_kernel/crates/ane/src/graph/ops.rs` lists all available ANE graph operations.

## Loop

1. Hypothesize — read at most 2 files
2. Implement — no debug prints, no diagnostic tests
3. Build — `make build`. One fix attempt if it fails, then revert.
4. Commit — `git add -A && git commit -m "<description>"`
5. Run — `make bench`
6. Publish — `./autoresearch-cli publish --agent=<NAME> --status=<keep|discard|crash> --description="<what>"`
7. tok/s improved → keep. Else → `git reset --hard HEAD~1`
8. Goto 1

Never stop. Never ask the human.
