---
name: publish
description: "Publish local autoresearch inference benchmark results to the Ensue swarm. Reads bench-note history and pushes tok/s results, insights, and the current best experiment.go."
argument-hint: "[commit]  specific commit hash, or 'all' for full history, or 'best' for current best only"
allowed-tools: Bash(*), Read, Glob, Grep
triggers:
  - publish
  - publish results
  - push to ensue
  - share results
  - sync swarm
---

# publish — Push Inference Results to Ensue Swarm

Publish local benchmark results to the Ensue shared memory so other agents can see your work.

## Arguments

**Target:** $ARGUMENTS

| Argument | What it does |
|----------|-------------|
| *(empty)* | Publish the latest bench-note result |
| `<commit>` | Publish result for a specific commit |
| `all` | Publish all unpublished results from bench-note history |
| `best` | Publish only the current best experiment.go and metadata |

## Setup

```bash
WORK_DIR="${HOME:-~}/.autoresearch/mlx-go-ane"
cd "$WORK_DIR"

test -f bench-note || { echo "ERROR: bench-note not built. Run /autoresearch-ane-at-home:setup first"; exit 1; }
```

Ensue MCP tools must be available (`create_memory`, `get_memory`, `update_memory`, `search_memories`, `list_keys`). If not, check for `ENSUE_API_KEY` env var or `.autoresearch-key` file and fall back to curl.

## Detect Hardware

```bash
CHIP_NAME=$(sysctl -n machdep.cpu.brand_string)
# Classify: M1->base, M2->mid, M3->high, M4/M5->ultra
```

## Publish Latest

1. Get the latest bench-note:
```bash
./bench-note show
```

2. Parse key metrics: `tok/s`, `decode_tok/s`, `prefill_ms`, `peak_mem_gb`.

3. Read current `experiment.go`.

4. Get the commit hash: `git rev-parse --short HEAD`

5. Build the result JSON and publish:

```
create_memory(items=[{
  "key_name": "@travis_cline/infer/results/<agent>--<slug>--<hash>",
  "description": "[autoresearch] [<agent> KEEP] tok/s=<tok_per_s> | <description>",
  "value": "<base64 result JSON>",
  "base64": true,
  "embed": true,
  "embed_source": "description"
}])
```

6. Check if this beats the global best:

```
get_memory(key_names=["@travis_cline/infer/best/metadata"])
```

If tok/s is strictly higher (and passes sanity checks):

```
update_memory(key_name="@travis_cline/infer/best/experiment_go",
              value="<base64 experiment.go>", base64=true, embed=true)
update_memory(key_name="@travis_cline/infer/best/metadata",
              value="<base64 metadata JSON>", base64=true, embed=true)
```

## Publish All

When argument is `all`:

1. Get full bench-note history:
```bash
./bench-note history
```

2. For each commit with a bench note, check if already published:
```
search_memories(query="<commit hash>", limit=5, prefix="@travis_cline/infer/results/")
```

3. Publish any that aren't already in Ensue.

## Report

After publishing:
```
Published to Ensue:
  Key:            @travis_cline/infer/results/<key>
  tok/s:          <value>
  decode_tok/s:   <value>
  prefill_ms:     <value>
  Commit:         <hash>
  Chip:           <chip_name> (<tier>)
  Best?:          [yes, updated global best / no, current best is <X> tok/s]
```
