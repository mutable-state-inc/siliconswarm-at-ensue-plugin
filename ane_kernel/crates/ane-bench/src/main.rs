/// ane-bench: Ensue coordinator CLI for ANE autoresearch.
///
/// Per-chip namespaces: @ane-bench/<chip>/results, insights, hypotheses, best
/// Each chip establishes its own CoreML baseline before optimizing.
use std::process::Command;

use serde::Deserialize;

const ORG: &str = "sai_ane";
const API_URL: &str = "https://api.ensue-network.ai/";

fn git_root() -> Option<std::path::PathBuf> {
    Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| std::path::PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

fn api_key() -> String {
    if let Ok(k) = std::env::var("ENSUE_API_KEY") {
        return k;
    }
    // Check git repo root for .autoresearch-key
    if let Some(root) = git_root() {
        let key_path = root.join(".autoresearch-key");
        if let Ok(k) = std::fs::read_to_string(&key_path) {
            let k = k.trim().to_string();
            if !k.is_empty() {
                return k;
            }
        }
    }
    // Fallback: check cwd (for running outside git)
    if let Ok(k) = std::fs::read_to_string(".autoresearch-key") {
        let k = k.trim().to_string();
        if !k.is_empty() {
            return k;
        }
    }
    eprintln!("No Ensue API key found.");
    eprintln!("  Checked: ENSUE_API_KEY env var");
    if let Some(root) = git_root() {
        eprintln!("  Checked: {}/.autoresearch-key", root.display());
    }
    eprintln!("  Fix: Set ENSUE_API_KEY or create .autoresearch-key in the project root");
    std::process::exit(1);
}

fn ram_gb() -> u64 {
    Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<u64>()
                .ok()
        })
        .map(|bytes| bytes / (1024 * 1024 * 1024))
        .unwrap_or(0)
}

fn chip_name() -> String {
    let out = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into());
    // Normalize: "Apple M1 Max" → "m1-max/64gb"
    // RAM is a sub-namespace so chip-level findings are browsable at m1-max/
    let base = out.to_lowercase().replace("apple ", "").replace(' ', "-");
    let gb = ram_gb();
    if gb > 0 {
        format!("{base}/{gb}gb")
    } else {
        base
    }
}

fn chip_slug(chip: &str) -> String {
    chip.chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '/')
        .collect()
}

// ─── Ensue JSON-RPC ────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct RpcResponse {
    result: Option<serde_json::Value>,
    error: Option<serde_json::Value>,
}

/// MCP-style RPC: method is always "tools/call", tool_name + arguments in params.
fn rpc(tool_name: &str, arguments: serde_json::Value) -> Option<serde_json::Value> {
    let key = api_key();
    let payload = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
        "id": 1
    });
    let mut response = match ureq::post(API_URL)
        .header("Authorization", &format!("Bearer {key}"))
        .header("Content-Type", "application/json")
        .send_json(&payload)
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Ensue HTTP error: {e}");
            if e.to_string().contains("401") {
                eprintln!("  Key may be invalid, expired, or org invite not accepted.");
                eprintln!("  Join org: https://www.ensue-network.ai/join?token=cffdd0692fb147c8b3f6422167118d69e6ec4809e88642e2a34359f0e1a5b3df");
            }
            return None;
        }
    };
    let body: String = response.body_mut().read_to_string().ok()?;
    // Handle SSE "data: " prefix
    let json_str = body.strip_prefix("data: ").unwrap_or(&body).trim();
    let resp: RpcResponse = serde_json::from_str(json_str).ok()?;
    if let Some(err) = resp.error {
        eprintln!("Ensue error: {err}");
        return None;
    }
    // MCP tool results: extract structuredContent or parse content[0].text
    let result = resp.result?;
    if let Some(sc) = result.get("structuredContent") {
        return Some(sc.clone());
    }
    if let Some(content) = result.get("content").and_then(|c| c.as_array())
        && let Some(text) = content
            .first()
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
    {
        return serde_json::from_str(text).ok();
    }
    Some(result)
}

fn write_memory(key: &str, description: &str, value: &serde_json::Value) -> bool {
    rpc(
        "create_memory",
        serde_json::json!({
            "items": [{
                "key_name": key,
                "description": description,
                "value": value.to_string(),
                "embed": true,
                "embed_source": "description"
            }]
        }),
    )
    .is_some()
}

fn read_memory(key: &str) -> Option<serde_json::Value> {
    let resp = rpc("get_memory", serde_json::json!({"key_names": [key]}))?;
    let results = resp.get("results").and_then(|r| r.as_array())?;
    let mem = results.first()?;
    let val_str = mem.get("value")?.as_str()?;
    serde_json::from_str(val_str).ok()
}

fn list_keys(prefix: &str) -> Vec<String> {
    let resp = rpc(
        "list_keys",
        serde_json::json!({"prefix": prefix, "limit": 50}),
    );
    let val = match resp {
        Some(v) => v,
        None => return vec![],
    };
    // Response: {"keys": [...], "count": N}
    val.get("keys")
        .and_then(|v| v.as_array())
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|v| {
            v.get("key_name")
                .and_then(|k| k.as_str())
                .map(|s| s.to_string())
                .or_else(|| v.get("key").and_then(|k| k.as_str()).map(|s| s.to_string()))
                .or_else(|| v.as_str().map(|s| s.to_string()))
        })
        .collect()
}

fn search_memories(query: &str, prefix: &str) -> Vec<(String, String)> {
    rpc(
        "search_memories",
        serde_json::json!({"query": query, "prefix": prefix, "limit": 10}),
    )
    .and_then(|v| v.as_array().cloned())
    .unwrap_or_default()
    .iter()
    .filter_map(|v| {
        let key = v
            .get("key")
            .and_then(|k| k.as_str())
            .or_else(|| v.as_str())?
            .to_string();
        let snippet = v
            .get("snippet")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        Some((key, snippet))
    })
    .collect()
}

// ─── Commands ──────────────────────────────────────────────────────────────

fn cmd_baseline(chip: &str, coreml_ms: f64) {
    let key = format!("@{ORG}/{chip}/baseline");
    let val = serde_json::json!({
        "chip": chip,
        "coreml_median_ms": coreml_ms,
        "timestamp": chrono_now(),
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "seq_len": 128,
    });
    if write_memory(
        &key,
        &format!("CoreML baseline for {chip}: {coreml_ms:.3}ms"),
        &val,
    ) {
        println!("Baseline recorded: {chip} → CoreML {coreml_ms:.3}ms");
    }
}

fn cmd_publish(chip: &str, agent: &str, status: &str, median_ms: f64, description: &str) {
    let source = std::fs::read_to_string("ane_kernel/crates/ane/examples/distilbert_model.rs")
        .unwrap_or_else(|_| "(source not found)".into());
    let slug = slugify(description);
    let hash = short_hash(&format!("{agent}{description}{}", chrono_now()));
    let key = format!("@{ORG}/{chip}/results/{agent}--{slug}--{hash}");

    let val = serde_json::json!({
        "agent": agent,
        "chip": chip,
        "status": status,
        "median_ms": median_ms,
        "description": description,
        "timestamp": chrono_now(),
        "kernel_source": source,
    });

    if write_memory(
        &key,
        &format!("[{chip}] {status}: {median_ms:.3}ms — {description}"),
        &val,
    ) {
        println!("Published: {key}");
        println!("  {status}: {median_ms:.3}ms — {description}");

        // Update best if this is a keep and beats current best
        if status == "keep" {
            if let Some(best) = read_memory(&format!("@{ORG}/{chip}/best/metadata")) {
                let best_ms = best
                    .get("median_ms")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(f64::MAX);
                if median_ms < best_ms {
                    let best_val = serde_json::json!({
                        "agent": agent,
                        "median_ms": median_ms,
                        "description": description,
                        "timestamp": chrono_now(),
                        "previous_best_ms": best_ms,
                    });
                    let best_key = format!("@{ORG}/{chip}/best/metadata");
                    let best_desc = format!("Best for {chip}: {median_ms:.3}ms — {description}");
                    rpc(
                        "delete_memory",
                        serde_json::json!({"key_names": [&best_key]}),
                    );
                    write_memory(&best_key, &best_desc, &best_val);
                    println!("  New best for {chip}! {best_ms:.3}ms → {median_ms:.3}ms");
                }
            } else {
                // No best yet, set it
                let best_val = serde_json::json!({
                    "agent": agent,
                    "median_ms": median_ms,
                    "description": description,
                    "timestamp": chrono_now(),
                });
                let best_key = format!("@{ORG}/{chip}/best/metadata");
                let best_desc = format!("Best for {chip}: {median_ms:.3}ms — {description}");
                rpc(
                    "delete_memory",
                    serde_json::json!({"key_names": [&best_key]}),
                );
                write_memory(&best_key, &best_desc, &best_val);
                println!("  First best for {chip}: {median_ms:.3}ms");
            }
        }
    }
}

fn cmd_insight(chip: &str, agent: &str, text: &str) {
    let slug = slugify(text);
    let hash = short_hash(&format!("{agent}{text}{}", chrono_now()));
    let key = format!("@{ORG}/{chip}/insights/{agent}--{slug}--{hash}");
    let val = serde_json::json!({
        "agent": agent,
        "chip": chip,
        "insight": text,
        "timestamp": chrono_now(),
    });
    if write_memory(&key, &format!("[{chip}] {text}"), &val) {
        println!("Insight: {key}");
    }
}

fn cmd_hypothesis(chip: &str, agent: &str, title: &str, text: &str) {
    let slug = slugify(title);
    let hash = short_hash(&format!("{agent}{title}{}", chrono_now()));
    let key = format!("@{ORG}/{chip}/hypotheses/{agent}--{slug}--{hash}");
    let val = serde_json::json!({
        "agent": agent,
        "chip": chip,
        "title": title,
        "hypothesis": text,
        "timestamp": chrono_now(),
    });
    if write_memory(&key, &format!("[{chip}] {title}: {text}"), &val) {
        println!("Hypothesis: {key}");
    }
}

fn cmd_results(chip: &str) {
    let keys = list_keys(&format!("@{ORG}/{chip}/results/"));
    if keys.is_empty() {
        println!("No results for {chip}");
        return;
    }
    println!("Results for {chip}:");
    for key in &keys {
        // list_keys returns relative key_name, get_memory needs full @org/ prefix
        let full_key = if key.starts_with('@') {
            key.clone()
        } else {
            format!("@{ORG}/{key}")
        };
        if let Some(val) = read_memory(&full_key) {
            let ms = val.get("median_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let status = val.get("status").and_then(|v| v.as_str()).unwrap_or("?");
            let desc = val
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            println!("  {ms:6.3}ms [{status:7}] {desc}");
        }
    }
}

fn cmd_best(chip: &str) {
    // Scan results to find actual best (more reliable than best/metadata key)
    let keys = list_keys(&format!("@{ORG}/{chip}/results/"));
    let mut best_ms = f64::MAX;
    let mut best_desc = String::new();
    let mut best_agent = String::new();
    for key in &keys {
        let full_key = if key.starts_with('@') {
            key.clone()
        } else {
            format!("@{ORG}/{key}")
        };
        if let Some(val) = read_memory(&full_key) {
            let status = val.get("status").and_then(|v| v.as_str()).unwrap_or("");
            if status != "keep" {
                continue;
            }
            let ms = val
                .get("median_ms")
                .and_then(|v| v.as_f64())
                .unwrap_or(f64::MAX);
            if ms < best_ms {
                best_ms = ms;
                best_desc = val
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_string();
                best_agent = val
                    .get("agent")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_string();
            }
        }
    }
    if best_ms < f64::MAX {
        println!("Best for {chip}: {best_ms:.3}ms by {best_agent} — {best_desc}");
        // Update best/metadata to match
        let best_val = serde_json::json!({
            "agent": best_agent,
            "median_ms": best_ms,
            "description": best_desc,
            "timestamp": chrono_now(),
        });
        let best_key = format!("@{ORG}/{chip}/best/metadata");
        let best_d = format!("Best for {chip}: {best_ms:.3}ms — {best_desc}");
        rpc(
            "delete_memory",
            serde_json::json!({"key_names": [&best_key]}),
        );
        write_memory(&best_key, &best_d, &best_val);
    } else {
        println!("No results for {chip} yet.");
    }
    match read_memory(&format!("@{ORG}/{chip}/baseline")) {
        Some(val) => {
            let ms = val
                .get("coreml_median_ms")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            println!("CoreML baseline: {ms:.3}ms");
        }
        None => println!("No CoreML baseline. Run `ane-bench baseline <ms>`."),
    }
}

fn cmd_insights(chip: &str) {
    let keys = list_keys(&format!("@{ORG}/{chip}/insights/"));
    if keys.is_empty() {
        println!("No insights for {chip}");
        return;
    }
    for key in &keys {
        if let Some(val) = read_memory(key) {
            let text = val.get("insight").and_then(|v| v.as_str()).unwrap_or("?");
            let agent = val.get("agent").and_then(|v| v.as_str()).unwrap_or("?");
            println!("  [{agent}] {text}");
        }
    }
}

fn cmd_search(chip: &str, query: &str) {
    let results = search_memories(query, &format!("@{ORG}/{chip}/"));
    if results.is_empty() {
        println!("No results for query: {query}");
        return;
    }
    for (key, snippet) in &results {
        println!("  {key}");
        if !snippet.is_empty() {
            println!("    {snippet}");
        }
    }
}

fn cmd_chip() {
    let chip = chip_slug(&chip_name());
    println!("{chip}");
}

fn cmd_ram() {
    let gb = ram_gb();
    println!("{gb}gb");
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn slugify(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
        .chars()
        .take(40)
        .collect()
}

fn short_hash(s: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    format!("{:06x}", h.finish() & 0xFFFFFF)
}

fn chrono_now() -> String {
    Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");
    let chip = chip_slug(&chip_name());

    match cmd {
        "chip" => cmd_chip(),
        "ram" => cmd_ram(),

        "baseline" => {
            let ms: f64 = args
                .get(2)
                .expect("Usage: ane-bench baseline <coreml_median_ms>")
                .parse()
                .expect("invalid number");
            cmd_baseline(&chip, ms);
        }

        "publish" => {
            let agent = arg_flag(&args, "--agent").expect("--agent required");
            let status =
                arg_flag(&args, "--status").expect("--status required (keep|discard|crash)");
            let median = arg_flag(&args, "--median").expect("--median required (ms)");
            let desc = arg_flag(&args, "--description").expect("--description required");
            let ms: f64 = median.parse().expect("invalid median");
            cmd_publish(&chip, &agent, &status, ms, &desc);
        }

        "insight" => {
            let agent = arg_flag(&args, "--agent").expect("--agent required");
            let text = args[3..].join(" ");
            cmd_insight(&chip, &agent, &text);
        }

        "hypothesis" => {
            let agent = arg_flag(&args, "--agent").expect("--agent required");
            let title = arg_flag(&args, "--title").expect("--title required");
            let text = arg_flag(&args, "--text").expect("--text required");
            cmd_hypothesis(&chip, &agent, &title, &text);
        }

        "results" => cmd_results(&chip),
        "best" => cmd_best(&chip),
        "insights" => cmd_insights(&chip),

        "search" => {
            let query = args.get(2).expect("Usage: ane-bench search <query>");
            cmd_search(&chip, query);
        }

        _ => {
            eprintln!("ane-bench — Ensue coordinator for ANE autoresearch");
            eprintln!();
            eprintln!("Chip: {chip}");
            eprintln!("Namespace: @{ORG}/{chip}/");
            eprintln!();
            eprintln!("Commands:");
            eprintln!("  chip                              Print detected chip name");
            eprintln!("  ram                               Print detected RAM in GB");
            eprintln!("  baseline <coreml_ms>              Record CoreML baseline for this chip");
            eprintln!(
                "  publish --agent=X --status=keep|discard --median=X.X --description=\"...\""
            );
            eprintln!("                                    Publish experiment result");
            eprintln!("  insight --agent=X <text>          Share an observation");
            eprintln!("  hypothesis --agent=X --title=\"...\" --text=\"...\"");
            eprintln!("                                    Propose an experiment idea");
            eprintln!("  results                           List all results for this chip");
            eprintln!(
                "  best                              Show best result + baseline for this chip"
            );
            eprintln!("  insights                          List insights for this chip");
            eprintln!(
                "  search <query>                    Semantic search within this chip's namespace"
            );
        }
    }
}

fn arg_flag(args: &[String], flag: &str) -> Option<String> {
    let prefix = format!("{flag}=");
    args.iter()
        .find(|a| a.starts_with(&prefix))
        .map(|a| a[prefix.len()..].to_string())
}
