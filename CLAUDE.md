# autoresearch-skill

Claude Code plugin for autonomous inference optimization on Apple Neural Engine. Runs real LLM benchmarks (Qwen3.5-4B-4bit via mlx-go) and coordinates results across machines via Ensue.

## Skills

| Skill | Invocation | What it does |
|-------|-----------|-------------|
| Setup | `/autoresearch:setup` | Clone mlx-go-ane + deps, build tools, detect chip, smoke test |
| Autoresearch | `/autoresearch:autoresearch [focus]` | Run the autonomous inference optimization loop (never stops) |
| Publish | `/autoresearch:publish [target]` | Push results to Ensue swarm |

## How it works

1. **Setup** clones `tmc/autoresearch-mlx-go-ane`, `tmc/mlx-go`, and `tmc/apple` into `~/.autoresearch/`, fixes go.mod replace directives, builds `bench-note`, and runs a smoke test. The model (Qwen3.5-4B-4bit) auto-downloads from HuggingFace on first run.

2. **Autoresearch** runs the autonomous experiment loop: modify `experiment.go` → benchmark (GPU/Plane/ANE modes) → keep/discard → publish to Ensue. Targets `tok/s` (tokens per second, higher is better).

3. **Publish** pushes local benchmark results to the Ensue shared memory network (`@travis_cline/infer/`) so other agents on other machines can see what you've tried.

## Ensue Swarm

Results are coordinated via Ensue shared memory under `@travis_cline/infer/`. Agents claim experiments (15-min TTL), publish results + insights + hypotheses, and update the global best. All Ensue features are optional — the harness works standalone.

## Testing locally

```bash
claude --plugin-dir /path/to/autoresearch-skill
# Then: /autoresearch:setup
# Then: /autoresearch:autoresearch cache-types
```
