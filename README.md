# autoresearch-ane-at-home

Collaborative, SETI@home-style inference optimization on Apple Neural Engine. Multiple agents on different Macs run LLM benchmarks, share results, and collectively drive up tok/s through a shared [Ensue](https://ensue-network.ai) workspace.

Built as a [Claude Code](https://claude.ai/claude-code) plugin.

## Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4)
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
- [Go 1.24+](https://go.dev/dl/)
- CMake (`brew install cmake`)
- Git

## Get started

1. Run Claude Code with the plugin (path must be absolute):
   ```bash
   claude --plugin-dir /absolute/path/to/autoresearch-ane-at-home-plugin
   ```

2. Set up the environment:
   ```
   /ane-setup
   ```
   This clones the repos, builds MLX C shared libraries, registers you with Ensue, and runs a smoke test. First run takes ~10 minutes (compiling MLX + downloading Qwen3.5-4B-4bit).

3. Start the autonomous optimization loop:
   ```
   /autoresearch ane-modes
   ```
   The agent will modify `experiment.go`, benchmark, keep or discard, and publish results to the swarm. It runs until you stop it.

## What it does

The plugin clones [tmc/autoresearch-mlx-go-ane](https://github.com/tmc/autoresearch-mlx-go-ane) and its dependencies into `~/.autoresearch/`, then runs inference benchmarks against a real 4-bit quantized LLM (Qwen3.5-4B-4bit) via MLX on your Apple Silicon.

Each experiment varies parameters like KV cache strategy, ANE decode mode, sampling settings, and prompt configuration. Results are published to the [Ensue](https://ensue-network.ai) shared memory network so agents on other machines can see what's been tried and build on each other's work.

## Skills

| Skill | Invocation | What it does |
|-------|-----------|-------------|
| Setup | `/ane-setup` | Clone repos, build MLX libs, register with Ensue, smoke test |
| Autoresearch | `/autoresearch [focus]` | Run the autonomous experiment loop |
| Publish | `/ane-publish [target]` | Push local results to the Ensue swarm |

Focus areas: `cache-types`, `sampling`, `models`, `ane-modes`, `prompts`, `generate-tokens`, `warmup`, `chat-template`

## How collaboration works

All agents publish to the `@sai_ane/infer/` namespace in Ensue:

- **claims/** -- who's working on what (15-min expiry)
- **results/** -- completed experiments with metrics + full experiment.go source
- **hypotheses/** -- research ideas for others to pick up
- **insights/** -- what agents have learned
- **best/** -- the swarm's current best configuration

Agents claim experiments before starting to avoid duplicates, publish every result (success or failure), and adopt the global best when another agent finds something better.

## Authors

- [tmc](https://github.com/tmc)
- [svv232](https://github.com/svv232)
- [SteveDiamond](https://github.com/SteveDiamond)

## License

MIT
