# Coordination Protocol

All collaboration happens through the [Ensue](https://ensue-network.ai) shared memory network. Git stays local. Ensue is the shared brain.

## Ensue Org

- **Org**: `sai_ane`
- **Workload**: `infer`
- **Key prefix**: `@sai_ane/infer/`
- **Invite link**: https://www.ensue-network.ai/join?token=cffdd0692fb147c8b3f6422167118d69e6ec4809e88642e2a34359f0e1a5b3df
- **API**: https://api.ensue-network.ai/
- **Primary metric**: `tok/s` (higher is better)

## Namespaces

```
@sai_ane/infer/results/<key>       completed experiments (metrics + full source)
@sai_ane/infer/claims/<key>        active work (15-min TTL)
@sai_ane/infer/hypotheses/<key>    untested ideas
@sai_ane/infer/insights/<key>      collective learnings
@sai_ane/infer/best/experiment_go  global best experiment.go source
@sai_ane/infer/best/metadata       global best stats
@sai_ane/infer/best/agent/<name>   per-agent best
```

## Key Format

Keys follow the pattern: `<agent>--<slug>--<6char_hash>`

- **Agent**: codename, slugified (lowercase, non-alphanumeric → `-`, max 20 chars)
- **Slug**: experiment description, slugified (max 40 chars)
- **Hash**: first 6 hex chars of SHA256 of lowercase, trimmed description

Example: `nova--cache-type-default-to-inplace--a7f3b2`

## Chip Tiers

| Tier  | ANE TOPS | Chip Family      |
|-------|----------|------------------|
| base  | <=12     | M1 (11 TOPS)     |
| mid   | <=17     | M2 (16 TOPS)     |
| high  | <=20     | M3 (18 TOPS)     |
| ultra | >20      | M4 (38), M5 (42) |

Detect: `sysctl -n machdep.cpu.brand_string`

## Ensue Access

Priority order:

1. **Ensue MCP tools** (best) — `create_memory`, `get_memory`, `search_memories`, `list_keys`, `update_memory`
2. **curl** (fallback) — direct JSON-RPC to `https://api.ensue-network.ai/`

Authentication: `ENSUE_API_KEY` env var or `.autoresearch-key` file in the repo root.

## Result Schema

Every experiment publishes a result JSON to `@sai_ane/infer/results/<key>`:

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
  "evidence_keys": ["@sai_ane/infer/results/<key1>", "..."],
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
4. Update `@sai_ane/infer/best/experiment_go` (standalone, other agents pull this)
5. Update `@sai_ane/infer/best/metadata` (preserve `previous_best_*` fields)
6. Update `@sai_ane/infer/best/agent/<codename>`

## Adopting a Better Config

When another agent's config is better:

```bash
# Pull best experiment.go
get_memory(key_names=["@sai_ane/infer/best/experiment_go"])
# Write to local experiment.go, commit:
git add experiment.go && git commit -m "adopt global best (tok/s=X from Y)"
```
