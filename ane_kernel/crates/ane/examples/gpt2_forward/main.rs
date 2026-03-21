mod compiled_model;
mod config;
mod executables;
mod kv_cache;
mod sampling;
mod session;
mod spinner;
mod weights;

use std::io::{self, Write};
use std::iter::repeat_n;
use std::time::Instant;

use safetensors::SafeTensors;
use tokenizers::Tokenizer;

use compiled_model::CompiledModel;
use session::Session;
use spinner::Spinner;

const REPO_ID: &str = "openai-community/gpt2";
const PROMPT: &str = "The meaning of life is";
const MAX_NEW_TOKENS: usize = 60;
const MAX_SEQUENCE_LENGTH: usize = 128;
const MIN_SPATIAL_WIDTH: usize = 64;
const TEMPERATURE: f32 = 0.8;
const TOP_P: f32 = 0.95;
const REPETITION_PENALTY: f32 = 1.2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let model_files = weights::download_model(REPO_ID)?;
    let config = model_files.config;

    let tokenizer = Tokenizer::from_file(&model_files.tokenizer_path)
        .map_err(|error| format!("tokenizer: {error}"))?;

    let encoding = tokenizer
        .encode(PROMPT, false)
        .map_err(|error| format!("encode: {error}"))?;
    let prompt_token_ids = encoding.get_ids();
    let prompt_length = prompt_token_ids.len();

    let padding_token = config.vocab_size as u32 - 1;
    let padded_length = prompt_length.max(MIN_SPATIAL_WIDTH);
    let padded_token_ids: Box<[u32]> = prompt_token_ids
        .iter()
        .copied()
        .chain(repeat_n(padding_token, padded_length - prompt_length))
        .collect();

    let safetensors_data = SafeTensors::deserialize(&model_files.safetensors_bytes)?;
    let model = CompiledModel::from_safetensors(
        config, &safetensors_data, padded_length, MAX_SEQUENCE_LENGTH,
    )?;

    let mut session = Session::new(&model, padded_length);
    let mut rng = rand::rng();

    {
        let prefill_spinner = Spinner::new("Prefilling prompt");
        let logits = session.prefill(&padded_token_ids, prompt_length);
        let first_token = sampling::sample(&logits, TEMPERATURE, TOP_P, REPETITION_PENALTY, prompt_token_ids, &mut rng);
        prefill_spinner.finish("Prefill complete");

        let prompt_text = tokenizer.decode(prompt_token_ids, true)
            .map_err(|error| format!("decode: {error}"))?;
        print!("{prompt_text}");
        io::stdout().flush()?;

        let mut generated_tokens: Vec<u32> = prompt_token_ids.to_vec();
        let mut previous_text = prompt_text;

        generated_tokens.push(first_token);
        let current_text = tokenizer.decode(&generated_tokens, true)
            .map_err(|error| format!("decode: {error}"))?;
        if let Some(delta) = current_text.strip_prefix(&previous_text) {
            print!("{delta}");
        }
        io::stdout().flush()?;
        previous_text = current_text;

        let generation_start = Instant::now();
        let mut total_attn_us = 0u64;
        let mut total_ffn_us = 0u64;

        for _ in 0..MAX_NEW_TOKENS - 1 {
            let last_token = *generated_tokens.last().unwrap();
            let (logits, timing) = session.decode_step_timed(last_token);
            total_attn_us += timing.attn_us;
            total_ffn_us += timing.ffn_us;

            let next_token = sampling::sample(&logits, TEMPERATURE, TOP_P, REPETITION_PENALTY, &generated_tokens, &mut rng);
            generated_tokens.push(next_token);

            let current_text = tokenizer.decode(&generated_tokens, true)
                .map_err(|error| format!("decode: {error}"))?;
            if let Some(delta) = current_text.strip_prefix(&previous_text) {
                print!("{delta}");
            }
            io::stdout().flush()?;
            previous_text = current_text;
        }
        let generation_elapsed = generation_start.elapsed().as_secs_f64();
        let ane_tok_s = MAX_NEW_TOKENS as f64 / generation_elapsed;
        let attn_ms = total_attn_us as f64 / 1000.0;
        let ffn_ms = total_ffn_us as f64 / 1000.0;
        let total_compute_ms = attn_ms + ffn_ms;
        let pipeline_ms = attn_ms.max(ffn_ms); // if pipelined, total = max(attn, ffn)
        let pipeline_tok_s = MAX_NEW_TOKENS as f64 / (pipeline_ms / 1000.0);

        println!();
        eprintln!();
        eprintln!("═══════════════════════════════════════════════════════");
        eprintln!("  GPT-2 on Apple Neural Engine (Private API)");
        eprintln!("═══════════════════════════════════════════════════════");
        eprintln!("  ANE-only (current):  {ane_tok_s:6.1} tok/s  ({generation_elapsed:.2}s)");
        eprintln!("  ─── per-token breakdown ───");
        eprintln!("  Attention:           {attn_ms:6.1}ms total ({:.2}ms/tok)",
            attn_ms / (MAX_NEW_TOKENS - 1) as f64);
        eprintln!("  FFN:                 {ffn_ms:6.1}ms total ({:.2}ms/tok)",
            ffn_ms / (MAX_NEW_TOKENS - 1) as f64);
        eprintln!("  ─── projected throughput ───");
        eprintln!("  GPU-only (if attn on GPU): would need Metal implementation");
        eprintln!("  GPU+ANE pipeline:    {pipeline_tok_s:6.1} tok/s  (attn||FFN concurrent)");
        eprintln!("  Pipeline speedup:    {:+.0}% vs ANE-only",
            (pipeline_tok_s / ane_tok_s - 1.0) * 100.0);
        eprintln!("═══════════════════════════════════════════════════════");
        eprintln!("  Total time: {:.1}s (including model download + compile)", start.elapsed().as_secs_f64());
    }

    Ok(())
}
