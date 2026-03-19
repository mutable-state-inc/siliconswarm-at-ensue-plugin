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

**IMPORTANT: Every Bash call gets a fresh shell.** You must prepend these exports to any command that uses `go test`, `bench-note`, `benchstat`, or `autoresearch-cli`:

```bash
export PATH="${PATH}:$(go env GOPATH)/bin"
export GOFLAGS="-tags=ane_appleneuralengine"
```

Without `PATH`, benchstat will not be found. Without `GOFLAGS`, the ANE runtime will be unavailable and all ANE benchmarks will run in degraded GPU-fallback mode.

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

**You MUST run ALL FOUR of these commands before EVERY iteration. No exceptions.**

**Command 1 — Read all results:**
```bash
./autoresearch-cli results
```
Print the output. Study what experiments have been tried. Do NOT repeat an experiment that already exists.

**Command 2 — Read insights:**
```bash
./autoresearch-cli search --query="optimization throughput improvement" --prefix=infer/insights/
```
Print the output. These are lessons learned by all agents. Use them to inform your next experiment.

**Command 3 — Check current best:**
```bash
./autoresearch-cli best
```
Print the output. This is the number to beat.

**Command 4 — Read hypotheses:**
```bash
./autoresearch-cli list --prefix=infer/hypotheses/
```
Print the output. These are ideas from other agents that haven't been tested yet. Prioritize testing these over your own ideas.

**After running all four commands, write a brief analysis:**
- What has been tried?
- What worked? What didn't?
- What hypotheses are untested?
- What will you try next, and WHY based on the collective intelligence?

**Only then proceed to step 2 (HACK).** If you skip any of these commands, the experiment is wasted — you'll duplicate work or miss insights that would have saved time.

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
