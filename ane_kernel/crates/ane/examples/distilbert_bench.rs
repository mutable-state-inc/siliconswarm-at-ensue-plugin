/// DistilBERT inference benchmark: private ANE API vs Apple's CoreML.
#[allow(dead_code)]
#[path = "distilbert_model.rs"]
mod distilbert_model;

use ane::{Shape, TensorData};
use distilbert_model::*;
use hf_hub::api::sync::ApiBuilder;
use safetensors::SafeTensors;
use std::time::Instant;

const REPO_ID: &str = "distilbert-base-uncased-finetuned-sst-2-english";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  DistilBERT Benchmark: Private ANE API vs CoreML");
    eprintln!("═══════════════════════════════════════════════════════");

    eprint!("  Downloading... ");
    let api = ApiBuilder::new().with_progress(false).build()?;
    let repo = api.model(REPO_ID.to_string());
    let sf_bytes = std::fs::read(repo.get("model.safetensors")?)?;
    let sf = SafeTensors::deserialize(&sf_bytes)?;
    let mw = load(&sf);
    let tok = tokenizers::Tokenizer::from_file(
        api.model("distilbert-base-uncased".to_string())
            .get("tokenizer.json")?,
    )
    .map_err(|e| format!("{e}"))?;
    eprintln!("ok");

    eprint!("  Compiling... ");
    let t0 = Instant::now();
    let layer_exes: Vec<_> = mw.layers.iter().map(compile_layer).collect();
    let cls_exe = compile_classifier(&mw);
    eprintln!(
        "ok ({:.1}s, {} dispatches)",
        t0.elapsed().as_secs_f64(),
        layer_exes.len() + 1
    );

    let hidden_a = TensorData::new(Shape::spatial(DIM, 1, SEQ));
    let hidden_b = TensorData::new(Shape::spatial(DIM, 1, SEQ));
    let mask_td = TensorData::new(Shape {
        batch: 1,
        channels: 1,
        height: SEQ,
        width: SEQ,
    });
    let cls_out = TensorData::new(Shape::spatial(CLS, 1, SEQ));

    // Sanity check
    let slen = embed(&mw, &tok, "I love this movie!", &hidden_a);
    set_mask(&mask_td, slen);
    forward(
        &layer_exes,
        &cls_exe,
        &hidden_a,
        &hidden_b,
        &mask_td,
        &cls_out,
    );
    let (_, _, label) = classify(&cls_out);
    eprintln!("  Sanity: \"I love this movie!\" → {label}");

    // Benchmark: embed + forward per iteration to match CoreML model.predict() scope
    // (CoreML's predict includes embedding layer; tokenization overhead is ~µs, negligible)
    eprintln!("\n  Benchmarking (1000 iterations, embed+forward)...");
    let text = "This is a test sentence for benchmarking.";
    let slen = embed(&mw, &tok, text, &hidden_a);
    set_mask(&mask_td, slen);

    for _ in 0..50 {
        embed(&mw, &tok, text, &hidden_a);
        forward(
            &layer_exes,
            &cls_exe,
            &hidden_a,
            &hidden_b,
            &mask_td,
            &cls_out,
        );
    }

    let mut times = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let start = Instant::now();
        embed(&mw, &tok, text, &hidden_a);
        forward(
            &layer_exes,
            &cls_exe,
            &hidden_a,
            &hidden_b,
            &mask_td,
            &cls_out,
        );
        times.push(start.elapsed().as_micros() as f64 / 1000.0);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let mean = times.iter().sum::<f64>() / times.len() as f64;

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!(
        "  Results (ANE private API, embed+{LAYERS} layers+cls, {} dispatches)",
        LAYERS + 1
    );
    eprintln!("═══════════════════════════════════════════════════════");
    eprintln!("  Mean:   {mean:.3} ms");
    eprintln!("  Median: {median:.3} ms");
    eprintln!("  P5:     {:.3} ms", times[times.len() / 20]);
    eprintln!("  P95:    {:.3} ms", times[times.len() * 19 / 20]);
    eprintln!("  Min:    {:.3} ms", times[0]);
    eprintln!("═══════════════════════════════════════════════════════");

    Ok(())
}
