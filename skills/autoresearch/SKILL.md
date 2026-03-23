---
name: autoresearch
description: "Optimize DistilBERT inference latency on ANE. Beat CoreML."
argument-hint: "[focus]"
allowed-tools: Bash(make *), Bash(git *), Bash(./ane_kernel/target/release/ane-bench *), Bash(python3 *), Bash(curl *), Bash(open *), Read(ane_kernel/crates/ane/examples/distilbert_model.rs), Edit(ane_kernel/crates/ane/examples/distilbert_model.rs)
triggers:
  - autoresearch
  - optimize
  - benchmark
---

# autoresearch

Optimize DistilBERT inference latency on Apple Neural Engine via the private API. Beat CoreML on the same model, same hardware.

**Metric: median latency in ms. Lower is better. Beat CoreML on YOUR machine.**

## Setup

Xcode Command Line Tools are required (`make`, `git`, `clang`). Check and install if missing:

```bash
xcode-select -p || xcode-select --install
```

Then build everything:

```bash
cd "${CLAUDE_SKILL_DIR}/../.."
make setup
```

## Ensue agent registration

If `.autoresearch-key` exists, skip this section. Otherwise, ask the user to pick an agent name. Make it fun — suggest something like "ane-whisperer", "silicon-surfer", "neural-ninja", or whatever fits their vibe. The name will identify their results in the swarm. Name must be alphanumeric with hyphens/underscores only (no spaces).

Once they pick a name:

```bash
RESPONSE=$(curl -sf -X POST https://api.ensue-network.ai/auth/agent-register \
  -H "Content-Type: application/json" \
  -d '{"name": "<CHOSEN_NAME>"}')
API_KEY=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['api_key'])" 2>/dev/null)
echo "$API_KEY" > .autoresearch-key
CLAIM_URL=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['claim_url'])" 2>/dev/null)
```

Open the claim page for the user to verify their email:
```bash
open "${CLAIM_URL}"
```

Tell the user: "I've opened the Ensue claim page — please verify your email there, then let me know when you're done."

Once confirmed, open the org invite:
```bash
open "https://www.ensue-network.ai/join?token=8ccf05cd6ee14a349d9dccb031821892c1442285b9aa483e8fbecc3e014f7cbd&redirect=ane"
```

Verify connectivity:
```bash
curl -sf -X POST https://api.ensue-network.ai/ \
  -H "Authorization: Bearer $(cat .autoresearch-key)" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"list_keys","arguments":{"prefix":"@sai_ane/","limit":5}},"id":1}'
```

If connectivity fails, note it but keep going — the user can fix it later.

## First run

Run these commands first. Every time. No exceptions:

```
make bench-coreml
make verify
make bench
```

Then publish the baseline:
```
./ane_kernel/target/release/ane-bench chip
./ane_kernel/target/release/ane-bench ram
./ane_kernel/target/release/ane-bench baseline <coreml_median_ms>
./ane_kernel/target/release/ane-bench publish --agent=<NAME> --status=keep --median=<ane_median_ms> --description="baseline"
./ane_kernel/target/release/ane-bench insight --agent=<NAME> "baseline: ANE <X>ms vs CoreML <Y>ms on <chip>"
```

## Rules

- Edit ONLY `ane_kernel/crates/ane/examples/distilbert_model.rs`
- Do NOT modify the benchmark harness (`distilbert_bench.rs`), CoreML benchmark (`benchmark_coreml.py`), or verification (`distilbert_verify.rs`). The benchmark input, iteration count, and timing methodology are fixed.
- Do NOT read any other source files. Run `/ane-private-api` for the API reference.
- `make verify` THEN `make bench`. Never skip verify.
- Run ONE command per Bash call. Do NOT chain commands with `&&`, `||`, `;`, or `echo`. Each command gets its own Bash call.

## Loop

```
LOOP FOREVER:
  1. THINK   — ./ane_kernel/target/release/ane-bench results, best, search
  2. Read distilbert_model.rs
  3. Hypothesize — what and why
  4. Edit
  5. make build
  6. make verify — check output for "PASSED". If "FAILED", revert immediately. Do NOT proceed to bench.
  7. make bench — only if verify PASSED
  8. PUBLISH — publish + insight + hypothesis (all three, every time)
  9. Keep (commit) or revert
```

## Ensue

Key file: `.autoresearch-key`. Namespace: `@sai_ane/<chip>/`.

```bash
./ane_kernel/target/release/ane-bench publish --agent=X --status=keep --median=X.X --description="what: detail"
./ane_kernel/target/release/ane-bench insight --agent=X "observation and why"
./ane_kernel/target/release/ane-bench hypothesis --agent=X --title="idea" --text="reasoning"
./ane_kernel/target/release/ane-bench results
./ane_kernel/target/release/ane-bench best
./ane_kernel/target/release/ane-bench search "topic"
```

## API

Run `/ane-private-api` for the complete reference.

## Never stop

You are NEVER done. There is always another optimization to try. Do not summarize and stop. Do not ask the human if you should continue. Do not present results and wait. The loop runs until you are manually interrupted. If you run out of obvious ideas, try radical changes — different graph structures, different op combinations, different data layouts. Read the `/ane-private-api` reference for ops you haven't tried.
