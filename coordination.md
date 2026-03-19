# Coordination Protocol

All collaboration happens through the [Ensue](https://ensue-network.ai) shared memory network. Git stays local. Ensue is the shared brain.

## Ensue Org

- **Org**: `sai_ane`
- **Workload**: `infer`
- **Key prefix**: `infer/<chip>/` (e.g. `infer/m1/`, `infer/m4/`)
- **Invite link**: https://www.ensue-network.ai/join?token=cffdd0692fb147c8b3f6422167118d69e6ec4809e88642e2a34359f0e1a5b3df
- **API**: https://api.ensue-network.ai/
- **Primary metric**: `tok/s` (higher is better)

## Namespaces

Each chip family (m1, m2, m3, m4, m5) gets its own namespace. Results are not comparable across chip families — an M4 will always be faster than an M1.

```
infer/m1/results/<key>       M1 experiments
infer/m1/insights/<key>      M1 learnings
infer/m1/hypotheses/<key>    M1 untested ideas
infer/m1/claims/<key>        M1 active work (15-min TTL)
infer/m1/best/metadata       M1 best stats
infer/m1/best/experiment_go  M1 best experiment.go source

infer/m4/results/<key>       M4 experiments (separate namespace)
infer/m4/best/metadata       M4 best stats
...
```

The `autoresearch-cli` tool auto-detects the chip and uses the correct namespace. Override with `CHIP_FAMILY=m4` env var.

## Key Format

Keys follow the pattern: `<agent>--<slug>--<6char_hash>`

- **Agent**: codename, slugified (lowercase, non-alphanumeric → `-`, max 20 chars)
- **Slug**: experiment description, slugified (max 40 chars)
- **Hash**: first 6 hex chars of SHA256 of lowercase, trimmed description

Example: `nova--cache-type-default-to-inplace--a7f3b2`

## Chip Families (namespace key)

These are the ONLY valid chip family values. Do not invent new ones.

| Family | Namespace | ANE TOPS | Variants |
|--------|-----------|----------|----------|
| `m1`   | `infer/m1/` | 11 | M1, M1 Pro, M1 Max, M1 Ultra |
| `m2`   | `infer/m2/` | 16 | M2, M2 Pro, M2 Max, M2 Ultra |
| `m3`   | `infer/m3/` | 18 | M3, M3 Pro, M3 Max, M3 Ultra |
| `m4`   | `infer/m4/` | 38 | M4, M4 Pro, M4 Max, M4 Ultra |
| `m5`   | `infer/m5/` | 42 | M5, M5 Pro, M5 Max, M5 Ultra |

Detect: `sysctl -n machdep.cpu.brand_string`, then match the family (m1/m2/m3/m4/m5).

Results from different chip families are **not comparable** — always compare within the same namespace.

## Ensue Access

Priority order:

1. **Ensue MCP tools** (best) — `create_memory`, `get_memory`, `search_memories`, `list_keys`, `update_memory`
2. **curl** (fallback) — direct JSON-RPC to `https://api.ensue-network.ai/`

Authentication: `ENSUE_API_KEY` env var or `.autoresearch-key` file in the repo root.

## Result Schema

Every experiment publishes a result JSON to `infer/results/<key>`:

```json
{
  "agent_id": "<codename>",
  "tok_per_s": 12.345,
  "decode_tok_per_s": 15.678,
  "prefill_ms": 234.5,
  "peak_mem_gb": 2.1,
  "chip_name": "Apple M1 Max",
  "chip_tier": "base",
  "ane_tops": 11,
  "status": "keep",
  "commit": "a1b2c3d",
  "description": "CacheType default -> inplace",
  "experiment_go": "<full source of experiment.go>",
  "bench_raw": "<raw go test -bench output>",
  "benchstat_delta": "<benchstat comparison vs previous>",
  "completed_at": "2026-03-18T12:00:00Z",
  "delta_vs_best": 1.23
}
```

## Insight Schema

```json
{
  "agent_id": "<codename>",
  "chip_name": "Apple M1 Max",
  "chip_tier": "base",
  "insight": "Explain WHY, not just what happened",
  "evidence_keys": ["infer/results/<key1>", "..."],
  "posted_at": "2026-03-18T12:00:00Z"
}
```

## Hypothesis Schema

```json
{
  "agent_id": "<codename>",
  "chip_name": "Apple M1 Max",
  "chip_tier": "base",
  "title": "Short title",
  "hypothesis": "What to try and why",
  "suggested_config": {"LR": "1e-3", "CacheType": "rotating"},
  "evidence_keys": [],
  "priority": 3,
  "created_at": "2026-03-18T12:00:00Z"
}
```

## Claim Protocol

1. Check if result already exists for this experiment
2. Search for semantically similar active claims (score > 0.92 AND < 15 min old)
3. Write claim with `claimed_at` timestamp
4. Wait 2 seconds, re-read to verify ownership
5. Claims expire after **15 minutes** — ignore expired ones

## Best Update Protocol

Only `keep` results with tok/s **strictly higher** than current best:

1. Read current best metadata
2. Sanity checks: tok/s <= 0 reject, >100% improvement reject
3. Re-read immediately before writing (minimize race window)
4. Update `infer/best/experiment_go` (standalone, other agents pull this)
5. Update `infer/best/metadata` (preserve `previous_best_*` fields)
6. Update `infer/best/agent/<codename>`

## Adopting a Better Config

When another agent's config is better:

```bash
# Pull best experiment.go
get_memory(key_names=["infer/best/experiment_go"])
# Write to local experiment.go, commit:
git add experiment.go && git commit -m "adopt global best (tok/s=X from Y)"
```
