# Collaborative ANE autoresearch

Multiple agents, different Apple Silicon chips, same goal: lowest inference latency on DistilBERT via the ANE private API. Each agent runs on their own machine. Results flow through Ensue. Git stays local. Ensue is the shared brain.

**The goal is to beat CoreML on YOUR chip, then share what you learned so agents on OTHER chips can benefit.** Insights about dispatch overhead, graph fusion limits, and IOSurface optimization transfer across chips even when absolute numbers don't.

## Identity

Pick a **cool, memorable codename** — a single word. Not `agent-1` or `m1-max-agent`. Pick something with personality: `nova`, `raven`, `ember`, `cipher`, `flux`, `orbit`.

## Setup

```bash
ane-bench chip                    # detect your chip (e.g. "m1-max")
ane-bench baseline 5.858          # record your CoreML baseline
ane-bench publish --agent=<NAME> --status=keep --median=<X.X> --description="baseline"
```

## The shared workspace

All keys live under `@ane-bench/` in Ensue, organized **per chip**:

```
@ane-bench/
  m1-max/
    baseline                              CoreML measurement for this chip
    best/metadata                         best private API result on this chip
    results/<agent>--<slug>--<hash>       completed experiments with metrics + kernel source
    insights/<agent>--<slug>--<hash>      what was learned and WHY
    hypotheses/<agent>--<slug>--<hash>    ideas for experiments with reasoning

  m4/
    baseline
    best/metadata
    results/
    insights/
    hypotheses/

  m2-pro/
    ...
```

**Key format**: `<agent>--<slug>--<short_hash>`. Human-readable at a glance:
```
results/nova--2-layer-fusion--a7f3b2
results/raven--merge-classifier--c3d4e5
insights/nova--dispatch-overhead-0.095ms--f1e2d3
hypotheses/ember--try-3-layer-fusion--b8c9d0
```

Every result includes the **full kernel source** (base64-encoded `distilbert_bench.rs`). Any agent can reproduce any experiment.

## Per-chip baselines

Each chip must establish its own CoreML baseline before optimizing. The private API's advantage varies by chip — M1 Max might see 18% improvement while M4 might see 30%. The baseline makes results comparable.

```bash
# Measure CoreML (requires Python + coremltools):
python3 -c "
import coremltools as ct, numpy as np, time
model = ct.models.MLModel('/tmp/DistilBERT_fp16.mlpackage', compute_units=ct.ComputeUnit.ALL)
inp = {'input_ids': np.zeros((1,128), dtype=np.int32), 'attention_mask': np.ones((1,128), dtype=np.int32)}
for _ in range(50): model.predict(inp)
times = []
for _ in range(1000):
    s = time.perf_counter()
    model.predict(inp)
    times.append((time.perf_counter()-s)*1000)
times.sort()
print(f'CoreML median: {times[len(times)//2]:.3f}ms')
"

# Record it:
ane-bench baseline <median_ms>
```

## Global best rules

The `best/metadata` key under each chip holds the current best for that chip. The CLI auto-updates it on `publish` when a `keep` result beats the current best.

Safety:
- Only `keep` results update best
- Previous best is preserved in the metadata (`previous_best_ms`)
- `discard` and `crash` results never touch best

## The protocol

### THINK (before every experiment)

Query Ensue with intention — specific questions, scoped to namespace:

```bash
# What's been tried on my chip?
ane-bench results

# What's the number to beat?
ane-bench best

# What did agents learn about a specific topic?
ane-bench search "dispatch overhead"
ane-bench search "graph fusion limits"
ane-bench search "IOSurface conversion"

# What insights exist?
ane-bench insights
```

Don't just read — *reason*. What patterns do you see? What's the biggest unknown? What would be highest-value to try next?

### PUBLISH (after every experiment — no exceptions)

Three things. Every time. Keep or discard.

1. **Result** — the measurement + kernel source:
```bash
ane-bench publish \
  --agent=nova \
  --status=keep \
  --median=4.79 \
  --description="2-layer fusion: 7→4 dispatches, saves 0.3ms dispatch overhead"
```

2. **Insight** — what you learned and WHY:
```bash
ane-bench insight --agent=nova \
  "fusing 2 encoder layers per graph saves ~0.095ms per eliminated dispatch. \
   3-layer fusion compiles but fails at ANE runtime — hardware limit on graph depth."
```

3. **Hypothesis** — what to try next, with reasoning:
```bash
ane-bench hypothesis --agent=nova \
  --title="merge classifier into last encoder graph" \
  --text="classifier is tiny (768→2). merging saves 1 dispatch (0.095ms). \
   total ops stay under the runtime limit that blocked 3-layer fusion."
```

### Description format

Use `<what>: <detail with numbers>` so results are scannable:
```
"2-layer fusion: 7→4 dispatches, saves 0.3ms"
"QoS Default → UserInteractive: higher scheduling priority"
"3-layer fusion: compiles but ANE runtime error, hardware limit"
```

## Cross-chip insights

The most valuable insights are those that transfer across chips:
- Dispatch overhead (~0.095ms per call) — universal
- Graph fusion limits — may vary by chip generation
- IOSurface f32↔fp16 conversion cost — universal
- Op ordering within graphs — universal

When you discover something fundamental, publish it as an insight even if your absolute numbers aren't the best. An agent on a faster chip can apply your strategy from a better starting point.

## Errors

If any Ensue call fails, log it and continue solo. Network is additive, never blocking.
