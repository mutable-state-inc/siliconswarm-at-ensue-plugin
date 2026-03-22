/// DistilBERT verification: full SST-2 validation set (872 examples).
/// Exit 0 if accuracy ≥ 91%, exit 1 otherwise.
#[allow(dead_code)]
#[path = "distilbert_model.rs"]
mod distilbert_model;

use std::io::Write;
use ane::{Shape, TensorData};
use hf_hub::api::sync::ApiBuilder;
use safetensors::SafeTensors;
use distilbert_model::*;

const REPO_ID: &str = "distilbert-base-uncased-finetuned-sst-2-english";
const MIN_ACCURACY: f64 = 91.0; // PyTorch fp16 = 91.06%, CoreML = 90.48%

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  DistilBERT Verification (ANE private API)");
    eprintln!("═══════════════════════════════════════════════════════");

    let api = ApiBuilder::new().with_progress(false).build()?;
    let repo = api.model(REPO_ID.to_string());
    let sf_bytes = std::fs::read(repo.get("model.safetensors")?)?;
    let sf = SafeTensors::deserialize(&sf_bytes)?;
    let mw = load(&sf);
    let tok = tokenizers::Tokenizer::from_file(
        api.model("distilbert-base-uncased".to_string()).get("tokenizer.json")?
    ).map_err(|e| format!("{e}"))?;

    eprint!("  Compiling... ");
    let layer_exes: Vec<_> = mw.layers.iter().map(|lw| compile_layer(lw)).collect();
    let cls_exe = compile_classifier(&mw);
    eprintln!("ok");

    let hidden_a = TensorData::new(Shape::spatial(DIM, 1, SEQ));
    let hidden_b = TensorData::new(Shape::spatial(DIM, 1, SEQ));
    let mask_td = TensorData::new(Shape { batch: 1, channels: 1, height: SEQ, width: SEQ });
    let cls_out = TensorData::new(Shape::spatial(CLS, 1, SEQ));

    // Load SST-2 validation set
    let sst2_path = std::path::Path::new("sst2_validation.tsv");
    if !sst2_path.exists() {
        eprintln!("  ERROR: sst2_validation.tsv not found in ane_kernel/");
        std::process::exit(1);
    }

    let tsv = std::fs::read_to_string(sst2_path).expect("read sst2");
    let lines: Vec<&str> = tsv.lines().skip(1).collect();
    let total = lines.len();
    eprintln!("  Running SST-2 validation ({total} examples)...");

    let mut correct = 0;
    let mut errors: Vec<(String, &str, &str)> = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let parts: Vec<&str> = line.splitn(2, '\t').collect();
        if parts.len() != 2 { continue; }
        let label: usize = parts[0].parse().unwrap_or(0);
        let sentence = parts[1].trim();
        let expected = if label == 1 { "POSITIVE" } else { "NEGATIVE" };

        let slen = embed(&mw, &tok, sentence, &hidden_a);
        set_mask(&mask_td, slen);
        forward(&layer_exes, &cls_exe, &hidden_a, &hidden_b, &mask_td, &cls_out);
        let (_, _, got) = classify(&cls_out);

        if got == expected {
            correct += 1;
        } else if errors.len() < 5 {
            errors.push((sentence.to_string(), expected, got));
        }

        if (i + 1) % 100 == 0 {
            eprint!("\r\x1b[2K  {}/{total} ({:.1}%)...", i + 1, correct as f64 / (i + 1) as f64 * 100.0);
            let _ = std::io::stderr().flush();
        }
    }

    let accuracy = correct as f64 / total as f64 * 100.0;

    eprintln!("\r\x1b[2K");
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  SST-2 Validation Results");
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  Correct:    {correct}/{total}");
    eprintln!("  Accuracy:   {accuracy:.2}%");
    eprintln!("  Threshold:  {MIN_ACCURACY}%");

    if !errors.is_empty() {
        eprintln!();
        eprintln!("  Sample errors:");
        for (sent, expected, got) in &errors {
            let short = if sent.len() > 60 { &sent[..60] } else { sent.as_str() };
            eprintln!("    {short}... expected={expected} got={got}");
        }
    }

    eprintln!("═══════════════════════════════════════════════════════");
    if accuracy >= MIN_ACCURACY {
        eprintln!("  PASSED ({accuracy:.2}% ≥ {MIN_ACCURACY}%)");
    } else {
        eprintln!("  FAILED ({accuracy:.2}% < {MIN_ACCURACY}%)");
        std::process::exit(1);
    }

    Ok(())
}
