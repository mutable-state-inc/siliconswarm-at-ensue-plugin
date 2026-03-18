---
name: setup
description: "Set up the autoresearch ANE inference environment: clone mlx-go-ane + mlx-go + apple repos, build MLX C shared libraries, build bench-note, detect Apple Silicon chip, and smoke test with Qwen3.5-4B-4bit."
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
| `install` | Full setup: prereqs → clone → build MLX libs → build tools → smoke test |
| `status` | Show current state of all components |
| *(empty)* | Same as `install` |

## Working Directory

```
~/.autoresearch/
├── mlx-go-ane/                    # Inference benchmarking framework (experiments here)
├── mlx-go/                        # MLX Go bindings (dependency)
│   ├── mlxc/lib/                  # MLX C shared library builder (cmake)
│   │   └── dist/darwin-arm64/     # Built: libmlx.dylib, libmlxc.dylib, mlx.metallib
│   └── examples/mlx-go-lm/       # LM inference library
└── apple/                         # Apple framework bindings
    ├── private/appleneuralengine/ # Private ANE bindings
    └── x/ane/                     # Higher-level ANE helpers
```

The model (Qwen3.5-4B-4bit) is auto-downloaded from HuggingFace on first benchmark run. No manual download needed.

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

# 4. CMake (required to build MLX C shared libraries)
cmake --version

# 5. benchstat
which benchstat || go install golang.org/x/perf/cmd/benchstat@latest

# 6. Chip detection — report chip name, tier, and ANE TOPS
# M1: 11 TOPS (base), M2: 16 TOPS (mid), M3: 18 TOPS (high), M4: 38 TOPS (ultra)
```

If Go is missing: `brew install go`
If CMake is missing: `brew install cmake`

---

## Phase 2: Clone Repositories

```bash
WORK_DIR="${HOME:-~}/.autoresearch"
mkdir -p "$WORK_DIR"

# Clone helper: try SSH first, fall back to HTTPS
clone_repo() {
    local repo="$1" dest="$2"
    if [ -d "$dest/.git" ]; then
        echo "$repo: already cloned"
        return 0
    fi
    echo "Cloning $repo..."
    git clone "git@github.com:tmc/$repo.git" "$dest" 2>/dev/null \
        || git clone "https://github.com/tmc/$repo.git" "$dest"
}

clone_repo mlx-go "$WORK_DIR/mlx-go"
clone_repo apple "$WORK_DIR/apple"
clone_repo autoresearch-mlx-go-ane "$WORK_DIR/mlx-go-ane"
```

---

## Phase 3: Build MLX C Shared Libraries

This is the critical step. mlx-go uses CGO bindings to libmlx/libmlxc. These must be built from source via cmake.

```bash
cd "$WORK_DIR/mlx-go/mlxc/lib"
make libs
```

This runs `go run generate.go` which:
1. Clones `ml-explore/mlx-c` (the official MLX C API)
2. Runs cmake to build `libmlx.dylib`, `libmlxc.dylib`, and `mlx.metallib`
3. Places them in `dist/darwin-arm64/`

**This takes several minutes on first run** (compiling MLX from source). Subsequent runs are fast (cached).

Verify the build:
```bash
ls -la "$WORK_DIR/mlx-go/mlxc/lib/dist/darwin-arm64/"
# Should contain: libmlx.dylib, libmlxc.dylib, mlx.metallib
```

The CGO linker flags reference `mlxc/lib/` directly, but `make libs` puts files in `dist/darwin-arm64/`. Symlink them so both paths work:
```bash
cd "$WORK_DIR/mlx-go/mlxc/lib"
for f in libmlx.dylib libmlxc.dylib mlx.metallib; do
    [ -f "dist/darwin-arm64/$f" ] && [ ! -e "$f" ] && ln -s "dist/darwin-arm64/$f" "$f"
done
```

---

## Phase 4: Fix go.mod Replace Directives

The go.mod in mlx-go-ane needs replace directives pointing to sibling repos:

```bash
cd "$WORK_DIR/mlx-go-ane"

# Ensure replace directives point to our layout
go mod edit -replace github.com/tmc/mlx-go=../mlx-go
go mod edit -replace github.com/tmc/mlx-go-lm=../mlx-go/examples/mlx-go-lm
go mod edit -replace github.com/tmc/apple=../apple
go mod tidy
```

---

## Phase 5: Build & Verify

```bash
cd "$WORK_DIR/mlx-go-ane"

# Verify test compilation (catches missing headers/libs)
go test -c -o /dev/null .

# Build bench-note if it exists
if [ -d cmd/bench-note ]; then
    go build -o bench-note ./cmd/bench-note/
fi
```

---

## Phase 6: Smoke Test

Run a quick benchmark. This auto-downloads Qwen3.5-4B-4bit from HuggingFace on first run (~2.5GB, cached after).

```bash
cd "$WORK_DIR/mlx-go-ane"
go test -bench=BenchmarkInference/mode=GPU/Generate -benchtime=1x -count=1 -run='^$' -timeout=15m -v
```

First run is slow due to model download + MLX compilation warmup. Report:
- tok/s, decode_tok/s, prefill_ms
- Whether ANE mode was active
- Peak memory

---

## Phase 7: Ensue Collaboration Setup (Required)

Ensue is required — this is a collaborative project. Results must be published to the shared swarm.

Check if the user already has an Ensue API key:

```bash
if [ -n "${ENSUE_API_KEY:-}" ] || [ -f "$WORK_DIR/mlx-go-ane/.autoresearch-key" ]; then
    echo "Ensue: API key found"
else
    echo "Ensue: No API key — registration required"
fi
```

If no key is found, **do not skip this step**. Walk the user through signup:

### Step 1: Register an agent

Ask the user for a name (or use their system username), then register:

```bash
curl -sf -X POST https://api.ensue-network.ai/auth/agent-register \
  -H "Content-Type: application/json" \
  -d '{"name": "autoresearch-<username>"}'
```

This returns a JSON response with `api_key` and `claim_url`. Save the key:

```bash
echo "<api_key from response>" > "$WORK_DIR/mlx-go-ane/.autoresearch-key"
```

### Step 2: Verify email

Tell the user to open the `claim_url` from the response (append `&redirect=/autoresearch`) in their browser and verify their email. **Wait for them to confirm before proceeding.**

### Step 3: Join the community swarm

Tell the user to visit: https://www.ensue-network.ai/autoresearch

This joins the shared workspace where all agents publish results.

### Step 4: Verify connectivity

Test that the key works:

```bash
ENSUE_API_KEY=$(cat "$WORK_DIR/mlx-go-ane/.autoresearch-key")
curl -sf -X POST https://api.ensue-network.ai/ \
  -H "Authorization: Bearer $ENSUE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"list_keys","arguments":{"prefix":"@travis_cline/infer/best/","limit":5}},"id":1}'
```

If this fails, **do not proceed** — help the user debug the connection. Setup is not complete without Ensue.

---

## Phase 8: Prepare Experiment Branch

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
  MLX libs:       libmlx.dylib, libmlxc.dylib, mlx.metallib
  Model:          Qwen3.5-4B-4bit (auto-downloaded via HF cache)
  Bench-note:     [built / N/A]
  ANE mode:       [active / unavailable]
  Ensue:          [connected / standalone]
  Branch:         autoresearch/[date]-setup

  Smoke test:
    tok/s:          [value]
    decode_tok/s:   [value]
    prefill_ms:     [value]

To start the autonomous inference optimization loop:
  /autoresearch-ane-at-home:autoresearch [focus-area]

Focus areas: cache-types, sampling, models, ane-modes, prompts,
             generate-tokens, warmup, chat-template
```

---

## Status Command

When action is `status`, check and report without modifying:

```bash
WORK_DIR="${HOME:-~}/.autoresearch"

echo "=== Prerequisites ==="
uname -m
sysctl -n machdep.cpu.brand_string
go version 2>/dev/null || echo "Go: NOT INSTALLED"
cmake --version 2>/dev/null | head -1 || echo "cmake: NOT INSTALLED"
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
echo "=== MLX C Libraries ==="
if [ -f "$WORK_DIR/mlx-go/mlxc/lib/dist/darwin-arm64/libmlx.dylib" ]; then
    echo "libmlx.dylib: BUILT"
    echo "libmlxc.dylib: $(test -f $WORK_DIR/mlx-go/mlxc/lib/dist/darwin-arm64/libmlxc.dylib && echo BUILT || echo MISSING)"
    echo "mlx.metallib: $(test -f $WORK_DIR/mlx-go/mlxc/lib/dist/darwin-arm64/mlx.metallib && echo BUILT || echo MISSING)"
else
    echo "MLX libs: NOT BUILT (run: cd ~/.autoresearch/mlx-go/mlxc/lib && make)"
fi

echo ""
echo "=== Build ==="
test -f "$WORK_DIR/mlx-go-ane/bench-note" && echo "bench-note: BUILT" || echo "bench-note: NOT BUILT"

echo ""
echo "=== Ensue ==="
if [ -n "${ENSUE_API_KEY:-}" ] || [ -f "$WORK_DIR/mlx-go-ane/.autoresearch-key" ]; then
    echo "API key: FOUND"
else
    echo "API key: NONE (run /autoresearch-ane-at-home:setup to register)"
fi

echo ""
echo "=== Model Cache ==="
ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-4bit 2>/dev/null && echo "Qwen3.5-4B-4bit: CACHED" || echo "Qwen3.5-4B-4bit: NOT CACHED (will download on first run)"
```

---

## Error Recovery

- **cmake missing**: `brew install cmake`
- **MLX C build fails**: Check cmake output, ensure Xcode command line tools: `xcode-select --install`
- **Clone fails**: Check network, verify repos are accessible at github.com/tmc/
- **go.mod replace paths wrong**: Fix with `go mod edit -replace`
- **`mlx/c/mlx.h` not found**: MLX C libs not built. Run `cd ~/.autoresearch/mlx-go/mlxc/lib && make`
- **Go build fails**: Run `go mod tidy`, ensure Go 1.24+
- **Model download slow/fails**: Check HF cache at `~/.cache/huggingface/`
- **ANE not available**: Falls back to GPU-only (set `ANEDecodePlaneMode = "off"`)
- **Benchmark timeout**: Increase to `-timeout=15m`, first run is slower
