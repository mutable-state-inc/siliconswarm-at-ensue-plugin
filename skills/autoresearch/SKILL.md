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

**Do NOT change DefaultModel.** The model is fixed at whatever is currently set in experiment.go. Optimize everything else — cache type, sampling, token count, ANE mode, prompts, chat template, warmup. Do not swap models to inflate tok/s.

For full Ensue protocol details (namespaces, key format, result/insight/hypothesis schemas, claim protocol, best-update rules), read [`coordination.md`](${CLAUDE_SKILL_DIR}/../../coordination.md) at startup.

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

### Branch discipline

**Always create a fresh branch from main before experimenting:**

```bash
git checkout main
git checkout main -- experiment.go   # reset to defaults
DATE=$(date +%Y%m%d)
git checkout -b "autoresearch/${DATE}-<YOUR_CODENAME>"
```

This ensures you start from a clean state. Each agent gets its own branch. When you discard an experiment (`git reset --hard HEAD~1`), you only affect your branch. Never push experiment branches — results go to Ensue, not git remote.

**IMPORTANT: Every Bash call gets a fresh shell.** You must prepend `export PATH="${PATH}:$(go env GOPATH)/bin"` to any command that uses `bench-note`, `benchstat`, or `autoresearch-cli`. Without this, benchstat will not be found.

Read these files at startup: `experiment.go` (your canvas), `program.md`, `harness.go` (read-only), `bench_ane_test.go` (read-only).

## Agent Identity

Pick a **unique codename** — a single creative word. Check existing agents first:

If Ensue MCP tools are available:
```
list_keys(prefix="infer/best/agent/", limit=50)
```

Pick a name NOT in that list. Draw from mythology, astronomy, nature, science.

## The Loop

Run forever. **Every single iteration** follows this exact sequence. No skipping steps. No reordering.

```
LOOP:
  1. THINK   → autoresearch-cli results + search (EVERY iteration, not just first)
  2. HACK    → edit experiment.go
  3. COMMIT  → go test -c && git commit
  4. RUN     → bench-note run
  5. PUBLISH → autoresearch-cli publish (EVERY iteration, not just keeps)
  6. DECIDE  → keep or git reset --hard HEAD~1
  7. GOTO 1
```

**You MUST run THINK before every experiment, not just the first one.** The swarm changes between iterations — other agents may have published new results, insights, or hypotheses while you were benchmarking. If you skip THINK, you will duplicate work or miss better approaches.

### 1. THINK

**FIRST, run this command. This is not optional:**

```bash
./autoresearch-cli results
```

This queries the Ensue shared memory and shows all experiments from all agents. **If this returns results, you are NOT the first agent.** Study what others have tried before picking your experiment.

Then gather more context:

```bash
# Search insights from other agents
./autoresearch-cli search --query="model size throughput" --prefix=infer/insights/

# Check current global best
./autoresearch-cli best

# List hypotheses to try
./autoresearch-cli list --prefix=infer/hypotheses/

# Local history (secondary — swarm data is primary)
./bench-note history --oneline
```

**Do NOT skip `./autoresearch-cli results`. Do NOT assume an empty swarm based only on `bench-note history`.**

### 2. CLAIM (if Ensue available)

Claim your experiment to prevent duplicate work (15-min TTL):

```
search_memories(query="<description>", limit=5, prefix="infer/claims/")

create_memory(items=[{
  "key_name": "infer/claims/<key>",
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

1. **Combine near-misses** — if two changes each gave +1% but not significant, try them together.
2. **Try radically different models** — the model is the biggest lever. Search HuggingFace for `mlx-community` models and try different architectures and sizes.
3. **Vary GenerateTokens widely** — throughput at 500 tokens may be very different from 50.
4. **Add new constants** — you can add new exported constants to experiment.go that the harness may pick up. Read harness.go to see what it looks for.
5. **Profile** — run `go test -bench=. -cpuprofile=cpu.prof` and analyze with `go tool pprof` to find actual bottlenecks. Report findings as insights.
6. **Propose deeper changes** — see "Proposing Changes Beyond experiment.go" below.

#### Proposing changes beyond experiment.go

You cannot modify `harness.go`, `bench_ane_test.go`, or the mlx-go / mlx-go-lm libraries directly. But you CAN and SHOULD read them deeply, reason about what limits throughput, and propose concrete changes to the user.

**When to do this**: After 5+ experiment.go iterations with no significant improvement, or when profiling reveals a bottleneck outside experiment.go.

**How to propose changes**:

1. **Read the relevant code** — study `harness.go`, `bench_ane_test.go`, `decode/plane.go`, `decode/config.go`, and mlx-go-lm source (especially `decode/` and `engine.go` under `$GOPATH/src/github.com/tmc/mlx-go/examples/mlx-go-lm/`).

2. **Identify the bottleneck** — use profiling, code reading, and your experiment results to form a thesis about what limits tok/s. Common bottlenecks:
   - Sampling strategy (hardcoded `"lazy"` in harness.go — are there faster strategies?)
   - KV cache configuration (rotating max size hardcoded, no quantized cache support)
   - ANE decode plane tuning (env vars like `MLXGO_ANE_QWEN35_OUTPUT_MODE`, `MLXGO_ANE_QWEN35_WAIT_MODE` etc. are powerful but not exposed in experiment.go)
   - Memory allocation patterns (per-iteration cache creation vs cache reuse)
   - Synchronization overhead (`mlx.Synchronize` calls in the hot path)

3. **Write a concrete proposal** — tell the user exactly:
   - Which file and lines to change
   - What the change is (show a diff or before/after)
   - Why you think it will help (with evidence from profiling or experiments)
   - Expected impact (estimated tok/s improvement and reasoning)

4. **Present it clearly** — output a structured proposal like:

   ```
   PROPOSAL: Expose ANE output mode in experiment.go

   File: harness.go:220
   Current: SamplingStrategy: "lazy"  (hardcoded)
   Proposed: SamplingStrategy: SamplingStrategy  (read from experiment.go)

   New constant in experiment.go:
     SamplingStrategy = "lazy"

   Rationale: Profiling shows 15% of decode time in sampling.
   Greedy (Temperature=0.0) should bypass this entirely but
   the "lazy" strategy still runs the sampling pipeline.
   An "argmax" strategy (if available in mlx-go-lm) would
   eliminate this overhead.

   Evidence: Temperature 0.0 vs 0.6 showed no difference,
   suggesting the sampling code runs regardless.
   ```

5. **Keep looping** — don't wait for a response. Propose the change, then continue experimenting with what you CAN change. The user will implement your proposals when they're ready.

6. **Publish proposals to Ensue** — if available, publish proposals as hypotheses so other agents and humans can see them:

   ```
   create_memory(items=[{
     "key_name": "@sai_ane/infer/proposals/<agent>--<slug>",
     "description": "[autoresearch] Proposal: <one-line summary>",
     "value": "<base64 proposal JSON with file, lines, diff, rationale, evidence>",
     "base64": true, "embed": true, "embed_source": "description"
   }])
   ```

**Key areas to investigate for proposals**:

- `harness.go` — sampling strategy, cache creation, synchronization points
- `decode/plane.go` — ANE output materialization, GPU-ANE sync, buffer pooling
- `decode/config.go` — env var knobs (OutputMode, ConsumerMode, PoolDepth, WaitMode, CompiledPrepare, DirectBlock) that could be exposed as experiment.go constants
- `mlx-go-lm/decode/` — token iterator implementation, sampling pipeline
- `mlx-go/` — core MLX bindings, memory management, Metal shader dispatch

### 4. COMMIT

```bash
go test -c -o /dev/null .   # verify compilation
git add -A && git commit -m "<param> <old> -> <new>"
```

### 5. RUN

```bash
./bench-note run --benchtime=1x --count=6
```

This runs `BenchmarkGenerate` and `BenchmarkPrefill`, attaches results as a git note, and auto-compares against the nearest ancestor.

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

**CRITICAL: You MUST run this exact command after every benchmark. No exceptions. No manual JSON. No shortcuts.**

```bash
./autoresearch-cli publish --agent=<YOUR_CODENAME> --status=<keep|discard|crash> --description="<what you changed>"
```

This is the ONLY way to publish results. The tool automatically:
- Reads the full `experiment.go` source code
- Reads the raw benchmark output from `bench-note raw`
- Reads the benchstat delta from `bench-note show`
- Detects chip name/tier/TOPS
- Parses tok/s, decode_tok/s, prefill_ms, peak_mem_gb from bench output
- Publishes everything to the Ensue shared memory

**Do NOT manually construct result JSON. Do NOT use create_memory directly. Just run the command.**

Example:
```bash
./autoresearch-cli publish --agent=cygnus --status=keep --description="CacheType default -> inplace"
```

Use `--dry-run` to preview without publishing. If the tool is missing, rebuild it:
```bash
cd "${CLAUDE_SKILL_DIR}/../.." && go build -o "$GOPATH_SRC/autoresearch-mlx-go-ane/publish-result" ./cmd/publish-result/
```

If `publish-result` fails, retry once. If it fails again, log the error and continue the loop — but flag to the user that publishing is broken.

### Updating Global Best

Only `keep` results with tok/s **strictly higher** than current best:

```
# 1. Read current best
get_memory(key_names=["infer/best/metadata"])

# 2. Safety checks: tok/s <= 0 reject, >100% improvement reject
# 3. Re-read immediately before writing (minimize race)

# 4. Update experiment.go source (standalone key — other agents pull this)
update_memory(key_name="infer/best/experiment_go",
              description="[autoresearch] Current best experiment.go source",
              value="<base64 experiment.go source>",
              base64=true, embed=true)

# 5. Update metadata (preserve previous_best_* fields)
update_memory(key_name="infer/best/metadata",
              description="[autoresearch] Best result metadata",
              value="<base64 JSON with tok_per_s, agent_id, chip_name, previous_best_*>",
              base64=true, embed=true)

# 6. Update per-agent best
update_memory(key_name="infer/best/agent/<codename>",
              description="[autoresearch] Best result for <codename>",
              value="<base64 JSON with tok_per_s, experiment_go, bench_raw, benchstat_delta>",
              base64=true, embed=true)
```

Other agents can adopt the best config by pulling `infer/best/experiment_go` and writing it to their local `experiment.go`.

## Safety Rules

1. **Never modify** `harness.go`, `bench_test.go`, or `bench_ane_test.go` — but DO read them and propose changes to the user
2. **Best-update safety** — always verify before writing to `best/`
3. **Claim TTL** — 15 minutes, ignore expired claims
4. **Ensue errors** — retry once, then flag to user if publishing is broken
5. **Never stop** — loop until manually interrupted

## Never Stop

Once the loop begins, do NOT pause to ask the human. Do NOT present a summary and wait for input. The human may be asleep. You are autonomous.

If you run out of obvious ideas:
1. Combine previous near-miss experiments together
2. Try radically different models from mlx-community on HuggingFace
3. Vary GenerateTokens (50, 200, 500, 1000) to find throughput scaling patterns
4. Profile with `go test -bench=. -cpuprofile=cpu.prof` and analyze bottlenecks
5. Read harness.go, decode/, and mlx-go-lm source — propose concrete changes to the user
6. Check swarm hypotheses if Ensue is available
7. Publish what you've learned as insights, hypotheses, and proposals

"I've tried all the knobs" is never a reason to stop. Propose deeper changes, try new models, profile, and keep iterating. Loop until manually stopped.
