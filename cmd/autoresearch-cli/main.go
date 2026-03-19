// Command autoresearch-cli is the deterministic interface to the Ensue
// shared memory for autoresearch agents. It handles publishing results
// (with full source + benchmarks), searching the collective intelligence,
// and reading specific memories.
//
// Usage:
//
//	autoresearch-cli publish  --agent=X --status=keep --description="..."
//	autoresearch-cli search   --query="cache type" [--prefix=infer/results/] [--limit=20]
//	autoresearch-cli list     [--prefix=infer/results/] [--limit=20]
//	autoresearch-cli get      --key=infer/best/metadata
//	autoresearch-cli best     # show current global best
//	autoresearch-cli results  # all results sorted by description
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const (
	ensueAPI    = "https://api.ensue-network.ai/"
	keyFile     = ".autoresearch-key"
	defaultOrg  = "sai_ane"
	defaultWork = "infer"
)

// chipFamily returns the Apple Silicon family name: m1, m2, m3, m4, m5.
func chipFamily() string {
	if f := os.Getenv("CHIP_FAMILY"); f != "" {
		return f
	}
	out, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output()
	if err != nil {
		return "unknown"
	}
	chip := strings.ToLower(string(out))
	for _, f := range []string{"m5", "m4", "m3", "m2", "m1"} {
		if strings.Contains(chip, f) {
			return f
		}
	}
	return "unknown"
}

// keyPrefix returns the org-qualified prefix for shared memory keys.
// Each chip family gets its own namespace: @org/infer/m1/, @org/infer/m4/, etc.
func keyPrefix() string {
	org := os.Getenv("ENSUE_ORG")
	if org == "" {
		org = defaultOrg
	}
	return "@" + org + "/" + defaultWork + "/" + chipFamily()
}

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	cmd := os.Args[1]
	args := os.Args[2:]
	var err error
	switch cmd {
	case "publish":
		err = cmdPublish(args)
	case "search":
		err = cmdSearch(args)
	case "list":
		err = cmdList(args)
	case "get":
		err = cmdGet(args)
	case "best":
		err = cmdBest(args)
	case "results":
		err = cmdResults(args)
	case "insight":
		err = cmdInsight(args)
	case "hypothesis":
		err = cmdHypothesis(args)
	default:
		usage()
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "autoresearch-cli %s: %v\n", cmd, err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `Usage: autoresearch-cli <command> [flags]

Commands:
  publish   --agent=NAME --status=keep|discard|crash --description="..."
            Publish experiment result with full source + benchmarks to Ensue.
            Run from the repo directory (reads experiment.go, bench-note).
            Use --dry-run to preview without publishing.

  search    --query="cache type experiments" [--prefix=infer/results/] [--limit=20]
            Semantic search across shared memories.

  list      [--prefix=infer/results/] [--limit=20]
            List keys in a namespace.

  get       --key=infer/best/metadata
            Read a specific memory. Decodes base64 values automatically.

  best      Show the current global best result.

  results   List all experiment results with metrics.

  insight   --agent=NAME --text="what you learned and WHY"
            Publish an insight to the shared memory.

  hypothesis --agent=NAME --title="short title" --text="what to try and why"
            Publish a hypothesis for other agents to test.
`)
	os.Exit(2)
}

// --- publish ---

type Result struct {
	AgentID        string  `json:"agent_id"`
	Model          string  `json:"model"`
	TokPerS        float64 `json:"tok_per_s"`
	DecodeTokPerS  float64 `json:"decode_tok_per_s"`
	PrefillMs      float64 `json:"prefill_ms"`
	PeakMemGB      float64 `json:"peak_mem_gb"`
	ChipName       string  `json:"chip_name"`
	ChipTier       string  `json:"chip_tier"`
	AneTops        int     `json:"ane_tops"`
	Status         string  `json:"status"`
	Commit         string  `json:"commit"`
	Description    string  `json:"description"`
	ExperimentGo   string  `json:"experiment_go"`
	HarnessGo      string  `json:"harness_go"`
	BenchRaw       string  `json:"bench_raw"`
	BenchstatDelta string  `json:"benchstat_delta"`
	CompletedAt    string  `json:"completed_at"`
	DeltaVsBest    float64 `json:"delta_vs_best"`
}

func cmdPublish(args []string) error {
	var agent, status, description, commit string
	var dryRun bool
	for _, a := range args {
		switch {
		case strings.HasPrefix(a, "--agent="):
			agent = strings.TrimPrefix(a, "--agent=")
		case strings.HasPrefix(a, "--status="):
			status = strings.TrimPrefix(a, "--status=")
		case strings.HasPrefix(a, "--description="):
			description = strings.TrimPrefix(a, "--description=")
		case strings.HasPrefix(a, "--commit="):
			commit = strings.TrimPrefix(a, "--commit=")
		case a == "--dry-run":
			dryRun = true
		}
	}
	if agent == "" || description == "" {
		return fmt.Errorf("--agent and --description are required")
	}
	if status == "" {
		status = "keep"
	}
	if commit == "" {
		commit = "HEAD"
	}

	experimentGo, err := os.ReadFile("experiment.go")
	if err != nil {
		return fmt.Errorf("read experiment.go: %w (are you in the repo directory?)", err)
	}

	harnessGo, _ := os.ReadFile("harness.go")
	if harnessGo == nil {
		harnessGo = []byte("(harness.go not found)")
	}

	// bench-note show returns the full benchmark output (raw + benchstat).
	// There is no separate "raw" subcommand — show is the canonical output.
	benchOutput, _ := shellOutput("./bench-note", "show")
	if benchOutput == "" {
		benchOutput = "(no bench-note output available)"
	}
	benchRaw := benchOutput
	benchShow := benchOutput

	// Parse metrics from BenchmarkGenerate (the primary harness.go Engine benchmark).
	// Fall back to BenchmarkInference/mode=GPU/Generate if BenchmarkGenerate unavailable.
	bgLine := findBenchLine(benchRaw, "BenchmarkGenerate-")
	tokPerS := parseMetricFromLine(bgLine, `(\d+\.?\d*)\s+tok/s`)
	prefillMs := parseMetricFromLine(bgLine, `(\d+\.?\d*)\s+prefill_ms`)
	peakMemGB := parseMetricFromLine(bgLine, `(\d+\.?\d*)\s+peak_mem_gb`)
	decodeTokPerS := parseMetricFromLine(bgLine, `(\d+\.?\d*)\s+decode_tok/s`)

	// Fall back to BenchmarkInference/GPU if BenchmarkGenerate didn't produce tok/s
	if tokPerS == 0 {
		gpuGenLine := findBenchLine(benchRaw, "mode=GPU/Generate")
		tokPerS = parseMetricFromLine(gpuGenLine, `(\d+\.?\d*)\s+tok/s`)
		prefillMs = parseMetricFromLine(gpuGenLine, `(\d+\.?\d*)\s+prefill_ms`)
		peakMemGB = parseMetricFromLine(gpuGenLine, `(\d+\.?\d*)\s+peak_mem_gb`)
	}
	if decodeTokPerS == 0 {
		gpuDecodeLine := findBenchLine(benchRaw, "mode=GPU/Decode")
		decodeTokPerS = parseMetricFromLine(gpuDecodeLine, `(\d+\.?\d*)\s+decode_tok/s`)
	}

	// Also parse ANE metrics for reporting
	aneGenLine := findBenchLine(benchRaw, "mode=ANE/Generate")
	aneTokPerS := parseMetricFromLine(aneGenLine, `(\d+\.?\d*)\s+tok/s`)
	_ = aneTokPerS // included in bench_raw for swarm visibility

	commitHash, err := shellOutput("git", "rev-parse", "--short", commit)
	if err != nil {
		return fmt.Errorf("git rev-parse: %w", err)
	}
	commitHash = strings.TrimSpace(commitHash)

	chipName, chipTier, aneTops := detectChip()

	// Parse model from experiment.go
	model := parseStringConst(string(experimentGo), `DefaultModel\s*=\s*"([^"]+)"`)

	result := Result{
		AgentID:        agent,
		Model:          model,
		TokPerS:        tokPerS,
		DecodeTokPerS:  decodeTokPerS,
		PrefillMs:      prefillMs,
		PeakMemGB:      peakMemGB,
		ChipName:       chipName,
		ChipTier:       chipTier,
		AneTops:        aneTops,
		Status:         status,
		Commit:         commitHash,
		Description:    description,
		ExperimentGo:   string(experimentGo),
		HarnessGo:      string(harnessGo),
		BenchRaw:       benchRaw,
		BenchstatDelta: benchShow,
		CompletedAt:    time.Now().UTC().Format(time.RFC3339),
	}

	resultJSON, _ := json.MarshalIndent(result, "", "  ")

	if dryRun {
		os.Stdout.Write(resultJSON)
		fmt.Println()
		return nil
	}

	apiKey := getAPIKey()
	if apiKey == "" {
		os.Stdout.Write(resultJSON)
		fmt.Println()
		return fmt.Errorf("no Ensue API key found (output written to stdout)")
	}

	key := experimentKey(agent, description)
	resultKey := keyPrefix() + "/results/" + key

	statusTag := strings.ToUpper(status)
	desc := fmt.Sprintf("[autoresearch] [%s %s] tok/s=%.1f | %s", agent, statusTag, tokPerS, description)

	insight := map[string]interface{}{
		"agent_id":  agent,
		"chip_name": chipName,
		"chip_tier": chipTier,
		"insight":   fmt.Sprintf("Result of experiment: %s. tok/s=%.1f (status=%s)", description, tokPerS, status),
		"posted_at": result.CompletedAt,
	}
	insightJSON, _ := json.Marshal(insight)

	hypothesis := map[string]interface{}{
		"agent_id":   agent,
		"chip_name":  chipName,
		"chip_tier":  chipTier,
		"title":      "Next experiment after: " + description,
		"hypothesis": "Agent should determine next experiment based on these results",
		"priority":   3,
		"created_at": result.CompletedAt,
	}
	hypothesisJSON, _ := json.Marshal(hypothesis)

	items := []map[string]interface{}{
		{
			"key_name":     resultKey,
			"description":  desc,
			"value":        base64.StdEncoding.EncodeToString(resultJSON),
			"base64":       true,
			"embed":        true,
			"embed_source": "description",
		},
		{
			"key_name":     keyPrefix() + "/insights/" + key,
			"description":  fmt.Sprintf("[autoresearch] Insight from %s: %s", agent, description),
			"value":        base64.StdEncoding.EncodeToString(insightJSON),
			"base64":       true,
			"embed":        true,
			"embed_source": "description",
		},
		{
			"key_name":     keyPrefix() + "/hypotheses/" + key,
			"description":  fmt.Sprintf("[autoresearch] Hypothesis from %s after: %s", agent, description),
			"value":        base64.StdEncoding.EncodeToString(hypothesisJSON),
			"base64":       true,
			"embed":        true,
			"embed_source": "description",
		},
	}

	if err := ensueRPC(apiKey, "create_memory", map[string]interface{}{"items": items}); err != nil {
		return fmt.Errorf("publish failed: %w", err)
	}

	fmt.Printf("published %s\n", resultKey)
	fmt.Printf("  agent:   %s\n", agent)
	fmt.Printf("  model:   %s\n", model)
	fmt.Printf("  tok/s:   %.1f\n", tokPerS)
	fmt.Printf("  status:  %s\n", status)
	fmt.Printf("  commit:  %s\n", commitHash)

	// Auto-update best if this is a keep and beats the current best
	if status == "keep" && tokPerS > 0 {
		updateBestIfNeeded(apiKey, result, resultJSON)
	}

	return nil
}

// updateBestIfNeeded reads the current best, compares, and updates if we beat it.
func updateBestIfNeeded(apiKey string, result Result, resultJSON []byte) {
	bestKey := keyPrefix() + "/best/metadata"

	// Read current best by decoding the memory value
	currentBestTokPerS := readBestTokPerS(apiKey, bestKey)

	if currentBestTokPerS > 0 && result.TokPerS <= currentBestTokPerS {
		fmt.Printf("  not a new best (current: %.1f tok/s)\n", currentBestTokPerS)
		return
	}

	// Sanity: reject >100% improvement as measurement error
	if currentBestTokPerS > 0 && result.TokPerS > currentBestTokPerS*2 {
		fmt.Fprintf(os.Stderr, "  warning: >100%% improvement (%.1f -> %.1f), skipping best update\n",
			currentBestTokPerS, result.TokPerS)
		return
	}

	// Update best metadata
	bestDesc := fmt.Sprintf("[autoresearch] Best: tok/s=%.1f by %s on %s with %s",
		result.TokPerS, result.AgentID, result.ChipName, result.Model)

	bestItems := []map[string]interface{}{
		{
			"key_name":     bestKey,
			"description":  bestDesc,
			"value":        base64.StdEncoding.EncodeToString(resultJSON),
			"base64":       true,
			"embed":        true,
			"embed_source": "description",
		},
	}

	// Try update first (key exists), fall back to create (key doesn't exist)
	for _, item := range bestItems {
		keyName := item["key_name"].(string)
		err := ensueRPC(apiKey, "update_memory", map[string]interface{}{
			"key_name":    keyName,
			"value":       item["value"],
			"description": item["description"],
			"base64":      true,
		})
		if err != nil {
			fmt.Fprintf(os.Stderr, "  update %s failed: %v, trying create\n", keyName, err)
			err2 := ensueRPC(apiKey, "create_memory", map[string]interface{}{
				"items": []map[string]interface{}{item},
			})
			if err2 != nil {
				fmt.Fprintf(os.Stderr, "  create %s also failed: %v\n", keyName, err2)
			}
		}
	}

	fmt.Printf("  NEW BEST: %.1f tok/s (was %.1f)\n", result.TokPerS, currentBestTokPerS)
}

// readBestTokPerS reads the current best tok/s from Ensue.
// Returns 0 if the key doesn't exist or can't be parsed.
func readBestTokPerS(apiKey, bestKey string) float64 {
	raw, err := ensueRPCRaw(apiKey, "get_memory", map[string]interface{}{
		"key_names": []string{bestKey},
	})
	if err != nil {
		return 0
	}
	// The response may have SSE prefix and nested JSON.
	// Just search for "tok_per_s" in any decoded base64 values.
	s := string(raw)
	// Try to find base64 values and decode them
	for _, part := range strings.Split(s, "\"value\":\"") {
		if len(part) < 10 {
			continue
		}
		end := strings.Index(part, "\"")
		if end < 0 {
			continue
		}
		decoded, err := base64.StdEncoding.DecodeString(part[:end])
		if err != nil {
			continue
		}
		var obj map[string]interface{}
		if json.Unmarshal(decoded, &obj) == nil {
			if v, ok := obj["tok_per_s"].(float64); ok && v > 0 {
				return v
			}
		}
	}
	return 0
}

// --- search ---

func cmdSearch(args []string) error {
	query := "experiment result"
	prefix := keyPrefix() + "/results/"
	limit := 20
	for _, a := range args {
		switch {
		case strings.HasPrefix(a, "--query="):
			query = strings.TrimPrefix(a, "--query=")
		case strings.HasPrefix(a, "--prefix="):
			prefix = strings.TrimPrefix(a, "--prefix=")
		case strings.HasPrefix(a, "--limit="):
			v, _ := strconv.Atoi(strings.TrimPrefix(a, "--limit="))
			if v > 0 {
				limit = v
			}
		}
	}

	apiKey := requireKey()
	result, err := ensueRPCResult(apiKey, "search_memories", map[string]interface{}{
		"query":  query,
		"prefix": prefix,
		"limit":  limit,
	})
	if err != nil {
		return err
	}

	printEnsueResult(result)
	return nil
}

// --- list ---

func cmdList(args []string) error {
	prefix := keyPrefix() + "/"
	limit := 20
	for _, a := range args {
		switch {
		case strings.HasPrefix(a, "--prefix="):
			prefix = strings.TrimPrefix(a, "--prefix=")
		case strings.HasPrefix(a, "--limit="):
			v, _ := strconv.Atoi(strings.TrimPrefix(a, "--limit="))
			if v > 0 {
				limit = v
			}
		}
	}

	apiKey := requireKey()
	result, err := ensueRPCResult(apiKey, "list_keys", map[string]interface{}{
		"prefix": prefix,
		"limit":  limit,
	})
	if err != nil {
		return err
	}

	printEnsueResult(result)
	return nil
}

// --- get ---

func cmdGet(args []string) error {
	var key string
	for _, a := range args {
		if strings.HasPrefix(a, "--key=") {
			key = strings.TrimPrefix(a, "--key=")
		}
	}
	if key == "" {
		return fmt.Errorf("--key is required")
	}

	apiKey := requireKey()
	result, err := ensueRPCResult(apiKey, "get_memory", map[string]interface{}{
		"key_names": []string{key},
	})
	if err != nil {
		return err
	}

	// Try to decode base64 values in results
	printDecodedMemory(result)
	return nil
}

// --- best ---

func cmdBest(_ []string) error {
	apiKey := requireKey()
	result, err := ensueRPCResult(apiKey, "get_memory", map[string]interface{}{
		"key_names": []string{keyPrefix() + "/best/metadata"},
	})
	if err != nil {
		return err
	}

	printDecodedMemory(result)
	return nil
}

// --- results ---

func cmdResults(_ []string) error {
	apiKey := requireKey()
	result, err := ensueRPCResult(apiKey, "list_keys", map[string]interface{}{
		"prefix": keyPrefix() + "/results/",
		"limit":  50,
	})
	if err != nil {
		return err
	}

	printEnsueResult(result)
	return nil
}

// --- insight ---

func cmdInsight(args []string) error {
	var agent, text string
	for _, a := range args {
		switch {
		case strings.HasPrefix(a, "--agent="):
			agent = strings.TrimPrefix(a, "--agent=")
		case strings.HasPrefix(a, "--text="):
			text = strings.TrimPrefix(a, "--text=")
		}
	}
	if agent == "" || text == "" {
		return fmt.Errorf("--agent and --text are required")
	}

	apiKey := requireKey()
	chipName, chipTier, _ := detectChip()

	insight := map[string]interface{}{
		"agent_id":  agent,
		"chip_name": chipName,
		"chip_tier": chipTier,
		"insight":   text,
		"posted_at": time.Now().UTC().Format(time.RFC3339),
	}
	insightJSON, _ := json.Marshal(insight)

	key := experimentKey(agent, text)
	items := []map[string]interface{}{
		{
			"key_name":     keyPrefix() + "/insights/" + key,
			"description":  fmt.Sprintf("[autoresearch] Insight from %s: %s", agent, truncate(text, 80)),
			"value":        base64.StdEncoding.EncodeToString(insightJSON),
			"base64":       true,
			"embed":        true,
			"embed_source": "description",
		},
	}

	if err := ensueRPC(apiKey, "create_memory", map[string]interface{}{"items": items}); err != nil {
		return fmt.Errorf("publish insight failed: %w", err)
	}

	fmt.Printf("published insight: %s\n", truncate(text, 80))
	return nil
}

// --- hypothesis ---

func cmdHypothesis(args []string) error {
	var agent, title, text string
	priority := 3
	for _, a := range args {
		switch {
		case strings.HasPrefix(a, "--agent="):
			agent = strings.TrimPrefix(a, "--agent=")
		case strings.HasPrefix(a, "--title="):
			title = strings.TrimPrefix(a, "--title=")
		case strings.HasPrefix(a, "--text="):
			text = strings.TrimPrefix(a, "--text=")
		case strings.HasPrefix(a, "--priority="):
			v, _ := strconv.Atoi(strings.TrimPrefix(a, "--priority="))
			if v > 0 {
				priority = v
			}
		}
	}
	if agent == "" || title == "" || text == "" {
		return fmt.Errorf("--agent, --title, and --text are required")
	}

	apiKey := requireKey()
	chipName, chipTier, _ := detectChip()

	hypothesis := map[string]interface{}{
		"agent_id":   agent,
		"chip_name":  chipName,
		"chip_tier":  chipTier,
		"title":      title,
		"hypothesis": text,
		"priority":   priority,
		"created_at": time.Now().UTC().Format(time.RFC3339),
	}
	hypothesisJSON, _ := json.Marshal(hypothesis)

	key := experimentKey(agent, title)
	items := []map[string]interface{}{
		{
			"key_name":     keyPrefix() + "/hypotheses/" + key,
			"description":  fmt.Sprintf("[autoresearch] Hypothesis from %s: %s", agent, truncate(title, 80)),
			"value":        base64.StdEncoding.EncodeToString(hypothesisJSON),
			"base64":       true,
			"embed":        true,
			"embed_source": "description",
		},
	}

	if err := ensueRPC(apiKey, "create_memory", map[string]interface{}{"items": items}); err != nil {
		return fmt.Errorf("publish hypothesis failed: %w", err)
	}

	fmt.Printf("published hypothesis: %s\n", truncate(title, 80))
	return nil
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

// --- Ensue RPC helpers ---

func ensueRPC(apiKey, method string, arguments map[string]interface{}) error {
	_, err := ensueRPCRaw(apiKey, method, arguments)
	return err
}

func ensueRPCResult(apiKey, method string, arguments map[string]interface{}) (string, error) {
	respBody, err := ensueRPCRaw(apiKey, method, arguments)
	if err != nil {
		return "", err
	}

	// Strip SSE "data: " prefix if present
	s := string(respBody)
	if strings.HasPrefix(s, "data: ") {
		s = strings.TrimPrefix(s, "data: ")
	}

	// Extract the text content from JSON-RPC response
	var rpcResp map[string]interface{}
	if err := json.Unmarshal([]byte(s), &rpcResp); err != nil {
		return s, nil // return raw if not JSON
	}
	if errField, ok := rpcResp["error"]; ok {
		return "", fmt.Errorf("RPC error: %v", errField)
	}

	result, ok := rpcResp["result"]
	if !ok {
		return s, nil
	}

	resultMap, ok := result.(map[string]interface{})
	if !ok {
		return s, nil
	}

	content, ok := resultMap["content"]
	if !ok {
		return s, nil
	}

	contentArr, ok := content.([]interface{})
	if !ok || len(contentArr) == 0 {
		return s, nil
	}

	first, ok := contentArr[0].(map[string]interface{})
	if !ok {
		return s, nil
	}

	text, ok := first["text"].(string)
	if !ok {
		return s, nil
	}

	return text, nil
}

func ensueRPCRaw(apiKey, method string, arguments map[string]interface{}) ([]byte, error) {
	rpc := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  "tools/call",
		"params": map[string]interface{}{
			"name":      method,
			"arguments": arguments,
		},
		"id": 1,
	}

	body, err := json.Marshal(rpc)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", ensueAPI, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode == 401 {
		return nil, fmt.Errorf("invalid API key (HTTP 401)")
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

// --- output helpers ---

func printEnsueResult(text string) {
	// Try to pretty-print as JSON
	var data interface{}
	if err := json.Unmarshal([]byte(text), &data); err == nil {
		pretty, _ := json.MarshalIndent(data, "", "  ")
		fmt.Println(string(pretty))
	} else {
		fmt.Println(text)
	}
}

func printDecodedMemory(text string) {
	var data map[string]interface{}
	if err := json.Unmarshal([]byte(text), &data); err != nil {
		fmt.Println(text)
		return
	}

	results, ok := data["results"].([]interface{})
	if !ok {
		printEnsueResult(text)
		return
	}

	for _, r := range results {
		rm, ok := r.(map[string]interface{})
		if !ok {
			continue
		}
		val, ok := rm["value"].(string)
		if !ok {
			continue
		}
		// Try base64 decode
		decoded, err := base64.StdEncoding.DecodeString(val)
		if err != nil {
			fmt.Println(val)
			continue
		}
		// Try to parse as JSON and pretty-print
		var obj interface{}
		if json.Unmarshal(decoded, &obj) == nil {
			pretty, _ := json.MarshalIndent(obj, "", "  ")
			fmt.Println(string(pretty))
		} else {
			fmt.Println(string(decoded))
		}
	}
}

// --- utility ---

func requireKey() string {
	k := getAPIKey()
	if k == "" {
		fmt.Fprintln(os.Stderr, "error: no Ensue API key. Set ENSUE_API_KEY or create .autoresearch-key")
		os.Exit(1)
	}
	return k
}

func getAPIKey() string {
	if k := os.Getenv("ENSUE_API_KEY"); k != "" {
		return strings.TrimSpace(k)
	}
	if d, err := os.ReadFile(keyFile); err == nil {
		return strings.TrimSpace(string(d))
	}
	return ""
}

func parseStringConst(text, pattern string) string {
	re := regexp.MustCompile(pattern)
	m := re.FindStringSubmatch(text)
	if len(m) < 2 {
		return ""
	}
	return m[1]
}

// findBenchLine finds the last benchmark output line containing the given substring.
// Benchmark lines look like: BenchmarkInference/model=.../mode=GPU/Generate-10  1  ...
func findBenchLine(text, sub string) string {
	var last string
	for _, line := range strings.Split(text, "\n") {
		if strings.Contains(line, sub) && strings.Contains(line, "ns/op") {
			last = line
		}
	}
	return last
}

// parseMetricFromLine extracts a metric from a single benchmark line.
func parseMetricFromLine(line, pattern string) float64 {
	if line == "" {
		return 0
	}
	re := regexp.MustCompile(pattern)
	m := re.FindStringSubmatch(line)
	if len(m) < 2 {
		return 0
	}
	v, _ := strconv.ParseFloat(m[1], 64)
	return v
}

func detectChip() (name, tier string, tops int) {
	out, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output()
	if err != nil {
		return "unknown", "unknown", 0
	}
	name = strings.TrimSpace(string(out))
	switch {
	case strings.Contains(name, "M5"):
		tops = 42
	case strings.Contains(name, "M4"):
		tops = 38
	case strings.Contains(name, "M3"):
		tops = 18
	case strings.Contains(name, "M2"):
		tops = 16
	case strings.Contains(name, "M1"):
		tops = 11
	}
	switch {
	case tops <= 0:
		tier = "unknown"
	case tops <= 12:
		tier = "base"
	case tops <= 17:
		tier = "mid"
	case tops <= 20:
		tier = "high"
	default:
		tier = "ultra"
	}
	return
}

var reNonAlnum = regexp.MustCompile(`[^a-z0-9]+`)

func slugify(text string, maxLen int) string {
	s := reNonAlnum.ReplaceAllString(strings.ToLower(strings.TrimSpace(text)), "-")
	s = strings.Trim(s, "-")
	if len(s) > maxLen {
		s = strings.TrimRight(s[:maxLen], "-")
	}
	return s
}

func experimentKey(agent, desc string) string {
	a := slugify(agent, 20)
	if a == "" {
		a = "unknown"
	}
	h := sha256.Sum256([]byte(strings.ToLower(strings.TrimSpace(desc))))
	return a + "--" + slugify(desc, 40) + "--" + fmt.Sprintf("%x", h)[:6]
}

func shellOutput(name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	out, err := cmd.Output()
	return string(out), err
}
