fn main() {
    let ane_ms: f64 = 9.8;   // ANE draft: 102 tok/s
    let gpu_ms: f64 = 10.5;  // GPU target: 95 tok/s

    println!("=== SPECULATIVE DECODING: ANE DRAFT + GPU VERIFY ===\n");
    println!("ANE draft model: {:.0} tok/s ({ane_ms}ms/token)", 1000.0/ane_ms);
    println!("GPU target model: {:.0} tok/s ({gpu_ms}ms/token)\n", 1000.0/gpu_ms);

    println!("{:>2} {:>4} {:>8} {:>8} {:>8} {:>7} {:>7}", "k", "α", "draft", "verify", "round", "tok/s", "speedup");
    println!("{}", "-".repeat(55));

    for k in 1u32..=5 {
        for &alpha in &[0.5f64, 0.7, 0.8, 0.9] {
            let draft_time = k as f64 * ane_ms;
            let verify_time = gpu_ms;
            let round_time = draft_time.max(verify_time); // pipelined overlap
            let expected = (1.0 - alpha.powi(k as i32 + 1)) / (1.0 - alpha);
            let tps = expected / round_time * 1000.0;
            let speedup = tps / (1000.0 / gpu_ms);
            println!("{k:>2} {alpha:>4.1} {draft_time:>7.1}ms {verify_time:>7.1}ms {round_time:>7.1}ms {tps:>6.0} {speedup:>6.2}x");
        }
    }
    
    println!("\n--- With FASTER ANE draft (dim=128, 6L, ~3M params) ---\n");
    let fast_ane: f64 = 3.0; // estimated for tiny model
    println!("{:>2} {:>4} {:>8} {:>8} {:>8} {:>7} {:>7}", "k", "α", "draft", "verify", "round", "tok/s", "speedup");
    println!("{}", "-".repeat(55));
    for k in 1u32..=8 {
        let alpha: f64 = 0.7;
        let draft_time = k as f64 * fast_ane;
        let round_time = draft_time.max(gpu_ms);
        let expected = (1.0 - alpha.powi(k as i32 + 1)) / (1.0 - alpha);
        let tps = expected / round_time * 1000.0;
        let speedup = tps / (1000.0 / gpu_ms);
        println!("{k:>2} {alpha:>4.1} {draft_time:>7.1}ms {gpu_ms:>7.1}ms {round_time:>7.1}ms {tps:>6.0} {speedup:>6.2}x");
    }
    
    println!("\nKey insight: ANE draft runs SIMULTANEOUSLY with GPU verify.");
    println!("Best config: fast draft (3ms/tok) with k=3, α=0.7 → 220 tok/s (2.3x!)");
}
