---
name: ane-setup
description: "Set up the autoresearch ANE inference environment: clone mlx-go-ane + mlx-go repos, build MLX C shared libraries, build bench-note, detect Apple Silicon chip, and smoke test with Qwen3.5-4B-4bit."
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

Repos are cloned to standard GOPATH paths under `$(go env GOPATH)/src/github.com/tmc/`:

```
$GOPATH/src/github.com/tmc/
├── autoresearch-mlx-go-ane/       # Inference benchmarking framework (experiments here)
└── mlx-go/                        # MLX Go bindings (dependency)
    ├── mlxc/lib/                  # MLX C shared library builder (cmake)
    │   └── dist/darwin-arm64/     # Built: libmlx.dylib, libmlxc.dylib, mlx.metallib
    └── examples/mlx-go-lm/       # LM inference library
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

# 5. Ensure GOPATH/bin is in PATH (benchstat, etc.)
export PATH="${PATH}:$(go env GOPATH)/bin"

# 6. benchstat
which benchstat || go install golang.org/x/perf/cmd/benchstat@latest

# 6. Chip detection — report chip name, tier, and ANE TOPS
# M1: 11 TOPS (base), M2: 16 TOPS (mid), M3: 18 TOPS (high), M4: 38 TOPS (ultra)
```

If Go is missing: `brew install go`
If CMake is missing: `brew install cmake`

---

## Phase 2: Clone Repositories

```bash
GOPATH_SRC="$(go env GOPATH)/src/github.com/tmc"
mkdir -p "$GOPATH_SRC"

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

clone_repo mlx-go "$GOPATH_SRC/mlx-go"
clone_repo autoresearch-mlx-go-ane "$GOPATH_SRC/autoresearch-mlx-go-ane"
```

---

## Phase 3: Build MLX C Shared Libraries

This is the critical step. mlx-go uses CGO bindings to libmlx/libmlxc. These must be built from source via cmake.

```bash
cd "$GOPATH_SRC/mlx-go/mlxc/lib"
make libs
```

This runs `go run generate.go` which:
1. Clones `ml-explore/mlx-c` (the official MLX C API)
2. Runs cmake to build `libmlx.dylib`, `libmlxc.dylib`, and `mlx.metallib`
3. Places them in `dist/darwin-arm64/`

**This takes several minutes on first run** (compiling MLX from source). Subsequent runs are fast (cached).

Verify the build:
```bash
ls -la "$GOPATH_SRC/mlx-go/mlxc/lib/dist/darwin-arm64/"
# Should contain: libmlx.dylib, libmlxc.dylib, mlx.metallib
```

The CGO linker flags reference `mlxc/lib/` directly, but `make libs` puts files in `dist/darwin-arm64/`. Symlink them so both paths work:
```bash
cd "$GOPATH_SRC/mlx-go/mlxc/lib"
for f in libmlx.dylib libmlxc.dylib mlx.metallib; do
    [ -f "dist/darwin-arm64/$f" ] && [ ! -e "$f" ] && ln -s "dist/darwin-arm64/$f" "$f"
done
```

---

## Phase 4: Set Up Go Workspace

Use `go work` so all sibling repos resolve locally without mutating each repo's go.mod:

```bash
cd "$GOPATH_SRC"

# Create workspace if it doesn't exist
if [ ! -f go.work ]; then
    go work init
    go work use ./mlx-go
    go work use ./mlx-go/examples/mlx-go-lm
    go work use ./autoresearch-mlx-go-ane
fi
```

This replaces `go mod edit -replace` directives. The workspace file lives at `$GOPATH/src/github.com/tmc/go.work` and tells Go to resolve `github.com/tmc/mlx-go`, `github.com/tmc/mlx-go-lm`, and `github.com/tmc/mlx-go-ane` from sibling directories.

Note: mlx-go's go.mod already contains the purego fork replace directive, which the workspace inherits automatically.

---

## Phase 5: Build & Verify

```bash
cd "$GOPATH_SRC/autoresearch-mlx-go-ane"

# Verify test compilation (catches missing headers/libs)
go test -c -o /dev/null .

# Build bench-note if it exists
if [ -d cmd/bench-note ]; then
    go build -o bench-note ./cmd/bench-note/
fi
```

Build the `autoresearch-cli` tool from the plugin source. The plugin directory is `${CLAUDE_SKILL_DIR}/../..`:

```bash
cd "${CLAUDE_SKILL_DIR}/../.."
go build -o "$GOPATH_SRC/autoresearch-mlx-go-ane/autoresearch-cli" ./cmd/autoresearch-cli/
```

This places the `autoresearch-cli` binary in the repo alongside `bench-note`. It provides:
- `autoresearch-cli publish` — publish results with full source + benchmarks
- `autoresearch-cli results` — list all swarm results
- `autoresearch-cli search` — semantic search across shared memories
- `autoresearch-cli best` — show current global best
- `autoresearch-cli get` — read a specific memory
- `autoresearch-cli list` — list keys in a namespace

---

## Phase 6: Smoke Test

Run a quick benchmark. This auto-downloads Qwen3.5-4B-4bit from HuggingFace on first run (~2.5GB, cached after).

```bash
cd "$GOPATH_SRC/autoresearch-mlx-go-ane"
go test -bench=BenchmarkGenerate -benchtime=1x -count=1 -run='^$' -timeout=15m -v
```

First run is slow due to model download + MLX compilation warmup. Report:
- tok/s, decode_tok/s, prefill_ms
- Whether ANE mode was active
- Peak memory

---

## Phase 7: Ensue Collaboration Setup

Results are published to the Ensue swarm. If no API key exists, register automatically — don't ask, just do it. The user will stop you if they don't want it.

```bash
if [ -n "${ENSUE_API_KEY:-}" ] || [ -f "$GOPATH_SRC/autoresearch-mlx-go-ane/.autoresearch-key" ]; then
    echo "Ensue: API key found"
fi
```

If no key is found, register immediately using the system username:

```bash
USERNAME=$(whoami)
# Try with username first, then with random suffix if name is taken
RESPONSE=$(curl -s -X POST https://api.ensue-network.ai/auth/agent-register \
  -H "Content-Type: application/json" \
  -d "{\"name\": \"autoresearch-${USERNAME}\"}")

# If name taken (409), retry with a random suffix
if echo "$RESPONSE" | grep -q "already taken"; then
    SUFFIX=$(head -c 4 /dev/urandom | xxd -p)
    RESPONSE=$(curl -s -X POST https://api.ensue-network.ai/auth/agent-register \
      -H "Content-Type: application/json" \
      -d "{\"name\": \"autoresearch-${USERNAME}-${SUFFIX}\"}")
fi

# Parse and save the API key
API_KEY=$(echo "$RESPONSE" | python3 -c "import sys,json; r=json.loads(sys.stdin.read()); print(r.get('agent',r).get('api_key',''))" 2>/dev/null)
echo "$API_KEY" > "$GOPATH_SRC/autoresearch-mlx-go-ane/.autoresearch-key"

# Exclude from git (don't commit secrets)
grep -qxF '.autoresearch-key' "$GOPATH_SRC/autoresearch-mlx-go-ane/.git/info/exclude" 2>/dev/null \
    || echo '.autoresearch-key' >> "$GOPATH_SRC/autoresearch-mlx-go-ane/.git/info/exclude"

# Parse the claim URL and verification code
CLAIM_URL=$(echo "$RESPONSE" | python3 -c "import sys,json; r=json.loads(sys.stdin.read()); print(r.get('agent',r).get('claim_url',''))" 2>/dev/null)
VERIFICATION_CODE=$(echo "$RESPONSE" | python3 -c "import sys,json; r=json.loads(sys.stdin.read()); print(r.get('agent',r).get('verification_code',''))" 2>/dev/null)
```

If `API_KEY` is empty after this, the registration failed — print the full `RESPONSE` so the user can see the error.

After saving the key, open the claim URL in the browser. **The key won't work until the user verifies their email.** This is a hard blocker — do not skip it.

```bash
# Open claim page in browser
open "${CLAIM_URL}" 2>/dev/null || echo "Visit: ${CLAIM_URL}"
```

**You MUST use AskUserQuestion here** to pause and wait for the user to confirm they've claimed. Display the verification code prominently so they can enter it on the claim page:

"I've opened the Ensue claim page in your browser.

Your verification code is: **<VERIFICATION_CODE>**

Please enter this code on the claim page to activate your API key, then say 'done'."

**Do not proceed until the user confirms.** The API key is invalid until claimed.

Once they confirm, open the org invite:

```bash
# Step 2: Join the sai_ane org
open "https://www.ensue-network.ai/join?token=cffdd0692fb147c8b3f6422167118d69e6ec4809e88642e2a34359f0e1a5b3df" 2>/dev/null || echo "Visit: https://www.ensue-network.ai/join?token=cffdd0692fb147c8b3f6422167118d69e6ec4809e88642e2a34359f0e1a5b3df"
```

Then immediately verify connectivity and keep moving:

```bash
ENSUE_API_KEY=$(cat "$GOPATH_SRC/autoresearch-mlx-go-ane/.autoresearch-key")
curl -sf -X POST https://api.ensue-network.ai/ \
  -H "Authorization: Bearer $ENSUE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"list_keys","arguments":{"prefix":"infer/best/","limit":5}},"id":1}'
```

If connectivity check fails, note it in the summary but keep going — the user can fix it later.

---

## Phase 8: Prepare Experiment Branch

```bash
cd "$GOPATH_SRC/autoresearch-mlx-go-ane"
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
  Repos:          $GOPATH/src/github.com/tmc/{autoresearch-mlx-go-ane, mlx-go}
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
  /autoresearch [focus-area]

Focus areas: cache-types, sampling, models, ane-modes, prompts,
             generate-tokens, warmup, chat-template
```

---

## Status Command

When action is `status`, check and report without modifying:

```bash
GOPATH_SRC="$(go env GOPATH)/src/github.com/tmc"

echo "=== Prerequisites ==="
uname -m
sysctl -n machdep.cpu.brand_string
go version 2>/dev/null || echo "Go: NOT INSTALLED"
cmake --version 2>/dev/null | head -1 || echo "cmake: NOT INSTALLED"
which benchstat 2>/dev/null && echo "benchstat: OK" || echo "benchstat: MISSING"

echo ""
echo "=== Repositories ==="
for repo in mlx-go autoresearch-mlx-go-ane; do
    if [ -d "$GOPATH_SRC/$repo/.git" ]; then
        echo "$repo: CLONED ($(cd $GOPATH_SRC/$repo && git branch --show-current))"
    else
        echo "$repo: NOT CLONED"
    fi
done

echo ""
echo "=== MLX C Libraries ==="
if [ -f "$GOPATH_SRC/mlx-go/mlxc/lib/dist/darwin-arm64/libmlx.dylib" ]; then
    echo "libmlx.dylib: BUILT"
    echo "libmlxc.dylib: $(test -f $GOPATH_SRC/mlx-go/mlxc/lib/dist/darwin-arm64/libmlxc.dylib && echo BUILT || echo MISSING)"
    echo "mlx.metallib: $(test -f $GOPATH_SRC/mlx-go/mlxc/lib/dist/darwin-arm64/mlx.metallib && echo BUILT || echo MISSING)"
else
    echo "MLX libs: NOT BUILT (run: cd \$GOPATH/src/github.com/tmc/mlx-go/mlxc/lib && make)"
fi

echo ""
echo "=== Build ==="
test -f "$GOPATH_SRC/autoresearch-mlx-go-ane/bench-note" && echo "bench-note: BUILT" || echo "bench-note: NOT BUILT"

echo ""
echo "=== Ensue ==="
if [ -n "${ENSUE_API_KEY:-}" ] || [ -f "$GOPATH_SRC/autoresearch-mlx-go-ane/.autoresearch-key" ]; then
    echo "API key: FOUND"
else
    echo "API key: NONE (run /ane-setup to register)"
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
- **Module resolution wrong**: Check `go.work` at `$GOPATH/src/github.com/tmc/go.work` lists all three modules
- **`mlx/c/mlx.h` not found**: MLX C libs not built. Run `cd $GOPATH/src/github.com/tmc/mlx-go/mlxc/lib && make`
- **Go build fails**: Run `go mod tidy`, ensure Go 1.24+
- **Model download slow/fails**: Check HF cache at `~/.cache/huggingface/`
- **ANE not available**: Falls back to GPU-only (set `ANEDecodePlaneMode = "off"`)
- **Benchmark timeout**: Increase to `-timeout=15m`, first run is slower
