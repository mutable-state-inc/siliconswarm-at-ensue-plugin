---
name: setup
description: "Set up the autoresearch ANE inference environment: clone mlx-go-ane + mlx-go + apple repos, fix go.mod, build bench-note, detect Apple Silicon chip, and run a smoke test with Qwen3.5-4B-4bit."
argument-hint: "[action]  check | install | status"
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep, Agent, AskUserQuestion
triggers:
  - setup
  - setup autoresearch
  - install
  - get started
---

# setup — Autoresearch ANE Inference Environment

Set up everything needed to run autonomous ANE inference experiments. Walk through each phase, verify success, report status.

## Arguments

**Action:** $ARGUMENTS

| Action | What it does |
|--------|-------------|
| `check` | Verify prerequisites without changing anything |
| `install` | Full setup: prereqs → clone → build → smoke test |
| `status` | Show current state of all components |
| *(empty)* | Same as `install` |

## Working Directory

```
~/.autoresearch/
├── mlx-go-ane/                    # Inference benchmarking framework (experiments here)
├── mlx-go/                        # MLX Go bindings (dependency, private repo)
│   └── examples/mlx-go-lm/       # LM inference library
└── apple/                         # Apple framework bindings (dependency)
```

The model (Qwen3.5-4B-4bit) is auto-downloaded from HuggingFace on first benchmark run via mlx-go's HF cache. No manual download needed.

---

## Phase 1: Prerequisites

```bash
# 1. macOS Apple Silicon
uname -m                           # must be arm64
sysctl -n machdep.cpu.brand_string # must contain "Apple"

# 2. Go 1.24+
go version

# 3. Git
git --version

# 4. GitHub CLI (for private repo access to mlx-go)
gh --version
gh auth status

# 5. benchstat
which benchstat || go install golang.org/x/perf/cmd/benchstat@latest

# 6. Chip detection — report chip name, tier, and ANE TOPS
# M1: 11 TOPS (base), M2: 16 TOPS (mid), M3: 18 TOPS (high), M4: 38 TOPS (ultra)
```

If Go is missing: `brew install go`
If gh is missing: `brew install gh`
If gh auth fails: prompt user to run `gh auth login`

---

## Phase 2: Clone Repositories

The mlx-go-ane framework depends on sibling repos via `replace` directives in go.mod.

```bash
WORK_DIR="$HOME/.autoresearch"
mkdir -p "$WORK_DIR"

# 1. Clone mlx-go (private — needs gh auth)
if [ ! -d "$WORK_DIR/mlx-go/.git" ]; then
    gh repo clone tmc/mlx-go "$WORK_DIR/mlx-go"
else
    echo "mlx-go: already cloned"
fi

# 2. Clone apple framework bindings
if [ ! -d "$WORK_DIR/apple/.git" ]; then
    gh repo clone tmc/apple "$WORK_DIR/apple"
else
    echo "apple: already cloned"
fi

# 3. Clone mlx-go-ane (the main benchmarking repo)
if [ ! -d "$WORK_DIR/mlx-go-ane/.git" ]; then
    gh repo clone tmc/mlx-go-ane "$WORK_DIR/mlx-go-ane"
else
    echo "mlx-go-ane: already cloned"
fi
```

---

## Phase 3: Fix go.mod Replace Directives

The upstream go.mod may have `replace` directives pointing to the author's local paths. Update them to point to our `~/.autoresearch/` layout:

```bash
cd "$WORK_DIR/mlx-go-ane"
grep -n 'replace' go.mod
```

The replace directives should point to:
- `github.com/tmc/mlx-go => ../mlx-go`
- `github.com/tmc/mlx-go-lm => ../mlx-go/examples/mlx-go-lm`
- `github.com/tmc/apple => ../apple`

Fix any that point elsewhere:

```bash
go mod edit -replace github.com/tmc/mlx-go=../mlx-go
go mod edit -replace github.com/tmc/mlx-go-lm=../mlx-go/examples/mlx-go-lm
go mod edit -replace github.com/tmc/apple=../apple
go mod tidy
```

---

## Phase 4: Build Tools

```bash
cd "$WORK_DIR/mlx-go-ane"

# Build bench-note
go build -o bench-note ./cmd/bench-note/

# Verify test compilation
go test -c -o /dev/null .
```

---

## Phase 5: Smoke Test

Run a quick benchmark (1 iteration, 1 count) to verify everything works. This auto-downloads Qwen3.5-4B-4bit from HuggingFace on first run.

```bash
cd "$WORK_DIR/mlx-go-ane"
go test -bench=BenchmarkInference/mode=GPU/Generate -benchtime=1x -count=1 -run='^$' -timeout=10m -v
```

Report:
- Build success/failure
- Model download status (auto-cached by mlx-go HF cache)
- tok/s, decode_tok/s, prefill_ms
- Whether ANE mode was active or unavailable
- Peak memory usage

---

## Phase 6: Ensue Connection (Optional)

```bash
# Check for Ensue API key
if [ -n "${ENSUE_API_KEY:-}" ] || [ -f "$WORK_DIR/mlx-go-ane/.autoresearch-key" ]; then
    echo "Ensue: API key found"
else
    echo "Ensue: No API key (standalone mode — results stay local)"
    echo "  To enable swarm mode: export ENSUE_API_KEY=<your-key>"
fi
```

If Ensue MCP tools are available, test connectivity:
```
list_keys(prefix="@travis_cline/infer/best/", limit=5)
```

---

## Phase 7: Prepare Experiment Branch

```bash
cd "$WORK_DIR/mlx-go-ane"
DATE=$(date +%Y%m%d)

BRANCH=$(git branch --show-current)
if [[ ! "$BRANCH" == autoresearch/* ]]; then
    git checkout -b "autoresearch/${DATE}-setup"
fi
```

---

## Summary

Report the full setup state:

```
Setup complete!

  Chip:           [chip name] ([tier] tier, [TOPS] TOPS)
  Repos:          ~/.autoresearch/{mlx-go-ane, mlx-go, apple}
  Model:          Qwen3.5-4B-4bit (auto-downloaded via HF cache)
  Bench-note:     built
  ANE mode:       [active / unavailable]
  Ensue:          [connected / standalone]
  Branch:         autoresearch/[date]-setup

  Smoke test:
    tok/s:          [value]
    decode_tok/s:   [value]
    prefill_ms:     [value]

To start the autonomous inference optimization loop:
  /autoresearch:autoresearch [focus-area]

Focus areas: cache-types, sampling, models, ane-modes, prompts,
             generate-tokens, warmup, chat-template
```

---

## Status Command

When action is `status`, check and report without modifying:

```bash
WORK_DIR="$HOME/.autoresearch"

echo "=== Prerequisites ==="
uname -m
sysctl -n machdep.cpu.brand_string
go version 2>/dev/null || echo "Go: NOT INSTALLED"
gh --version 2>/dev/null || echo "gh: NOT INSTALLED"
which benchstat 2>/dev/null && echo "benchstat: OK" || echo "benchstat: MISSING"

echo ""
echo "=== Repositories ==="
for repo in mlx-go apple mlx-go-ane; do
    if [ -d "$WORK_DIR/$repo/.git" ]; then
        echo "$repo: CLONED ($(cd $WORK_DIR/$repo && git branch --show-current))"
    else
        echo "$repo: NOT CLONED"
    fi
done

echo ""
echo "=== Build ==="
test -f "$WORK_DIR/mlx-go-ane/bench-note" && echo "bench-note: BUILT" || echo "bench-note: NOT BUILT"

echo ""
echo "=== Ensue ==="
if [ -n "${ENSUE_API_KEY:-}" ] || [ -f "$WORK_DIR/mlx-go-ane/.autoresearch-key" ]; then
    echo "API key: FOUND"
else
    echo "API key: NONE (standalone mode)"
fi

echo ""
echo "=== Model Cache ==="
ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-4bit 2>/dev/null && echo "Qwen3.5-4B-4bit: CACHED" || echo "Qwen3.5-4B-4bit: NOT CACHED (will download on first run)"
```

---

## Error Recovery

- **gh auth fails**: Run `gh auth login` interactively
- **mlx-go clone fails (private repo)**: User needs invite — check https://github.com/tmc/mlx-go/invitations
- **go.mod replace paths wrong**: Fix with `go mod edit -replace`
- **Go build fails**: Run `go mod tidy`, ensure Go 1.24+
- **Model download slow/fails**: Check HF cache at `~/.cache/huggingface/`
- **ANE not available**: Falls back to GPU-only mode (set `ANEDecodePlaneMode = "off"`)
- **Benchmark timeout**: Increase timeout (`-timeout=15m`), first run is slower due to model download + compilation
