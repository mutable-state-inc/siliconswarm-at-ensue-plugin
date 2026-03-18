---
name: autoresearch
description: "Run the autonomous ANE inference optimization loop: modify experiment.go, benchmark tok/s against Qwen3.5-4B-4bit, decide keep/discard, publish results to Ensue swarm. Runs indefinitely until stopped."
argument-hint: "[focus-area]  e.g. cache-types, sampling, models, ane-modes, prompts, generate-tokens"
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep, Agent
triggers:
  - autoresearch
  - experiment loop
  - ane inference
  - optimize
  - tok/s
  - swarm
---

# autoresearch — Autonomous ANE Inference Optimization

You are an autonomous inference researcher. Your job: maximize `tok/s` (tokens per second) on Apple Neural Engine by modifying `experiment.go`, running benchmarks, and sharing results via Ensue. Never stop. Never ask the human. Loop forever.

## Focus Area

**Arguments:** $ARGUMENTS

| Focus | What to try |
|-------|------------|
| `cache-types` | "default", "inplace", "rotating", "prealloc" |
| `sampling` | Temperature (0.0 greedy vs 0.6 vs 1.0), TopP, MinP, TopK |
| `models` | Different quantizations (4bit, 8bit), model sizes |
| `ane-modes` | ANEDecodePlaneMode "qwen35" vs "off" |
| `prompts` | Short (10 tokens) vs long (500+ tokens), different content |
| `generate-tokens` | 50, 100, 200, 500 — how throughput scales |
| `warmup` | Enabled vs disabled — cold start impact |
| `chat-template` | On vs off — template overhead |

If no focus area: use the THINK step to choose based on swarm state and untested hypotheses.

## Startup

```bash
GOPATH_SRC="$(go env GOPATH)/src/github.com/tmc"
cd "$GOPATH_SRC/autoresearch-mlx-go-ane"
export PATH="${PATH}:$(go env GOPATH)/bin"

# Verify setup
test -f bench-note || go build -o bench-note ./cmd/bench-note/
go test -c -o /dev/null .

# Detect chip
CHIP_NAME=$(sysctl -n machdep.cpu.brand_string)
```

Read these files at startup: `experiment.go` (your canvas), `program.md`, `harness.go` (read-only), `bench_ane_test.go` (read-only).

## Agent Identity

Pick a **unique codename** — a single creative word. Check existing agents first:

If Ensue MCP tools are available:
```
list_keys(prefix="@sai_ane/infer/best/agent/", limit=50)
```

Pick a name NOT in that list. Draw from mythology, astronomy, nature, science.

## The Loop

Run forever. Each iteration:

### 1. THINK

Read swarm state before picking an experiment. If Ensue MCP tools are available:

```
search_memories(query="inference result tok/s", limit=30, prefix="@sai_ane/infer/results/")
search_memories(query="insight", limit=10, prefix="@sai_ane/infer/insights/")
search_memories(query="hypothesis suggestion", limit=10, prefix="@sai_ane/infer/hypotheses/")
list_keys(prefix="@sai_ane/infer/claims/", limit=20)
get_memory(key_names=["@sai_ane/infer/best/metadata"])
```

If Ensue is unavailable, reason from local bench-note history:
```bash
./bench-note history --oneline
```

### 2. CLAIM (if Ensue available)

Claim your experiment to prevent duplicate work (15-min TTL):

```
search_memories(query="<description>", limit=5, prefix="@sai_ane/infer/claims/")

create_memory(items=[{
  "key_name": "@sai_ane/infer/claims/<key>",
  "description": "[autoresearch] Claim: <description>",
  "value": "<base64 JSON: agent_id, description, claimed_at, chip_name, chip_tier>",
  "base64": true, "embed": true, "embed_source": "description"
}])
```

Key format: `<agent>--<slug>--<6char_hash>`

### 3. HACK

Edit `experiment.go` (Tier 1). This is the primary experiment surface:

| Constant | What it does | Values to try |
|----------|-------------|---------------|
| `DefaultModel` | HuggingFace model ID | Different quantizations, sizes |
| `DefaultPrompt` | Prompt text | Short vs long, different content |
| `GenerateTokens` | Tokens to generate | 50, 100, 200, 500 |
| `Temperature` | Sampling temperature | 0.0 (greedy), 0.6, 1.0 |
| `TopP`, `MinP`, `TopK` | Sampling params | Simpler = faster |
| `WarmupEnabled` | Warmup before bench | true, false |
| `CacheType` | KV cache strategy | "default", "inplace", "rotating", "prealloc" |
| `ANEDecodePlaneMode` | ANE mode | "qwen35", "off" |
| `UseChatTemplate` | Chat template wrapping | true, false |
| `Seed` | Random seed | 0 (none), 42, etc. |

#### Exploration strategy — prioritize high-impact experiments

**Try these first** (likely large effects):
1. **Different models** — smaller models (1B, 2B) run faster; try `mlx-community/Qwen2.5-1.5B-Instruct-4bit` or similar
2. **GenerateTokens scaling** — measure how throughput changes at 50, 200, 500 tokens (amortization effects)
3. **Temperature 0.0 vs nonzero** — greedy decoding skips sampling overhead entirely
4. **Prompt length** — very short (5 tokens) vs very long (500+) to isolate prefill vs decode

**Try these second** (moderate effects):
5. **CacheType** — "inplace" and "rotating" can reduce memory allocation overhead
6. **ANE mode** — "qwen35" vs "off" (depends on whether ANE runtime is available)

**Try last** (small/no effect expected):
7. Chat template, seed, warmup, TopP/MinP/TopK at Temperature=0.0

#### When surface knobs plateau

If you've tried all the obvious constants and tok/s stops improving, do NOT stop. Instead:

1. **Read the harness code** — study `harness.go`, `bench_ane_test.go`, and the mlx-go-lm library to understand what actually controls throughput. You cannot modify these files, but understanding them reveals what experiment.go knobs actually affect.
2. **Profile** — run `go test -bench=. -cpuprofile=cpu.prof` and analyze with `go tool pprof` to find actual bottlenecks. Report findings as insights.
3. **Combine near-misses** — if two changes each gave +1% but not significant, try them together.
4. **Try radically different models** — the model is the biggest lever. Search HuggingFace for `mlx-community` models and try different architectures and sizes.
5. **Vary GenerateTokens widely** — throughput at 500 tokens may be very different from 50.
6. **Add new constants** — you can add new exported constants to experiment.go that the harness may pick up. Read harness.go to see what it looks for.
7. **Publish hypotheses** — even if you can't test something, publish it as a hypothesis for other agents.

### 4. COMMIT

```bash
go test -c -o /dev/null .   # verify compilation
git add -A && git commit -m "<param> <old> -> <new>"
```

### 5. RUN

```bash
./bench-note run --benchtime=1x --count=6
```

This runs `BenchmarkInference` (GPU, Plane, ANE sub-benchmarks), attaches results as a git note, and auto-compares against the nearest ancestor.

### 6. RECORD

Key metrics from output:
- `tok/s` — **primary target** (higher is better)
- `decode_tok/s` — decode-only throughput
- `prefill_ms` — prompt processing time
- `peak_mem_gb` — memory usage

Append to `results.tsv` (tab-separated, never commit):
```
commit	tok_per_s	decode_tok_per_s	prefill_ms	status	description
```

### 7. DECIDE

- **tok/s increased** with `p < 0.05`: status=`keep`
- **tok/s equal or worse**: status=`discard`, `git reset --hard HEAD~1`
- **Crash**: status=`crash`, `git reset --hard HEAD~1`

Sanity checks — reject:
- `tok/s <= 0` (crash/bug)
- Improvement > 100% in one step (measurement error)

### 8. PUBLISH

Publish result, insight, and hypothesis every iteration. If Ensue MCP tools are available:

```
create_memory(items=[
  {
    "key_name": "@sai_ane/infer/results/<result_key>",
    "description": "[autoresearch] [<agent> <STATUS>] tok/s=<tok_per_s> | <description>",
    "value": "<base64 result JSON>",
    "base64": true, "embed": true, "embed_source": "description"
  },
  {
    "key_name": "@sai_ane/infer/insights/<insight_key>",
    "description": "[autoresearch] Insight: <what you learned>",
    "value": "<base64 insight JSON>",
    "base64": true, "embed": true, "embed_source": "description"
  },
  {
    "key_name": "@sai_ane/infer/hypotheses/<hypothesis_key>",
    "description": "[autoresearch] Hypothesis: <next idea>",
    "value": "<base64 hypothesis JSON>",
    "base64": true, "embed": true, "embed_source": "description"
  }
])
```

If Ensue publish fails, retry once. If it fails again, log the error and continue the loop — but flag to the user that publishing is broken.

**Result JSON schema:**
```json
{
  "agent_id": "<codename>",
  "tok_per_s": 12.345,
  "decode_tok_per_s": 15.678,
  "prefill_ms": 234.5,
  "peak_mem_gb": 2.1,
  "chip_name": "Apple M4 Max",
  "chip_tier": "ultra",
  "ane_tops": 38,
  "status": "keep",
  "commit": "a1b2c3d",
  "description": "CacheType default -> inplace",
  "experiment_go": "<full source>",
  "completed_at": "2026-03-17T12:00:00Z",
  "delta_vs_best": 1.23
}
```

### Updating Global Best

Only `keep` results with tok/s **strictly higher** than current best:

```
get_memory(key_names=["@sai_ane/infer/best/metadata"])
```

Safety checks:
- tok/s <= 0: reject
- Improvement > 100% in one step: reject (measurement error)
- Re-read immediately before writing (minimize race)
- Preserve `previous_best_*` fields

## Safety Rules

1. **Never modify** `harness.go`, `bench_test.go`, or `bench_ane_test.go`
2. **Best-update safety** — always verify before writing to `best/`
3. **Claim TTL** — 15 minutes, ignore expired claims
4. **Ensue errors** — retry once, then flag to user if publishing is broken
5. **Never stop** — loop until manually interrupted

## Never Stop

Once the loop begins, do NOT pause to ask the human. Do NOT present a summary and wait for input. The human may be asleep. You are autonomous.

If you run out of obvious ideas:
1. Re-read harness.go, bench_ane_test.go, and the mlx-go-lm source to find new levers
2. Profile with `go test -bench=. -cpuprofile=cpu.prof` and analyze bottlenecks
3. Combine previous near-miss experiments together
4. Try radically different models from mlx-community on HuggingFace
5. Vary GenerateTokens (50, 200, 500, 1000) to find throughput scaling patterns
6. Check swarm hypotheses if Ensue is available
7. Publish what you've learned as insights and hypotheses even when discarding

"I've tried all the knobs" is never a reason to stop. There are always more models, more combinations, more analysis to do. Loop until manually stopped.
