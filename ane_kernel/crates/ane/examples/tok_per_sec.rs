/// Measure actual tokens per second on ANE.
/// Simulate autoregressive decode: run the model once per token.
use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn build_layer_kernels(dim: usize, hidden: usize, seq: usize) 
    -> (ane::Executable, ane::Executable, ane::Executable, ane::Executable) 
{
    // Attn (Q+O)
    let sp1 = seq + 2 * dim;
    let mut g1 = Graph::new();
    let p1 = g1.placeholder(Shape { batch: 1, channels: dim, height: 1, width: sp1 });
    let h = g1.slice(p1, [0, 0, 0, 0], [1, dim, 1, seq]);
    let hr = g1.reshape(h, Shape { batch: 1, channels: 1, height: dim, width: seq });
    let ht = g1.transpose(hr, [0, 1, 3, 2]);
    let qw = g1.slice(p1, [0, 0, 0, seq], [1, dim, 1, dim]);
    let qr = g1.reshape(qw, Shape { batch: 1, channels: 1, height: dim, width: dim });
    let q = g1.matrix_multiplication(ht, qr, false, false);
    let ow = g1.slice(p1, [0, 0, 0, seq + dim], [1, dim, 1, dim]);
    let or2 = g1.reshape(ow, Shape { batch: 1, channels: 1, height: dim, width: dim });
    let o = g1.matrix_multiplication(q, or2, false, false);
    let ot = g1.transpose(o, [0, 1, 3, 2]);
    let oout = g1.reshape(ot, Shape { batch: 1, channels: dim, height: 1, width: seq });
    let _ = g1.addition(h, oout);
    let e1 = g1.compile(NSQualityOfService::Default).unwrap();

    // Gate+silu
    let sp2 = seq + hidden;
    let mut g2 = Graph::new();
    let p2 = g2.placeholder(Shape { batch: 1, channels: dim, height: 1, width: sp2 });
    let h2 = g2.slice(p2, [0, 0, 0, 0], [1, dim, 1, seq]);
    let h2r = g2.reshape(h2, Shape { batch: 1, channels: 1, height: dim, width: seq });
    let h2t = g2.transpose(h2r, [0, 1, 3, 2]);
    let gw = g2.slice(p2, [0, 0, 0, seq], [1, dim, 1, hidden]);
    let gr = g2.reshape(gw, Shape { batch: 1, channels: 1, height: dim, width: hidden });
    let gate = g2.matrix_multiplication(h2t, gr, false, false);
    let gs = g2.sigmoid(gate);
    let _ = g2.multiplication(gate, gs);
    let e2 = g2.compile(NSQualityOfService::Default).unwrap();

    // Up
    let mut g3 = Graph::new();
    let p3 = g3.placeholder(Shape { batch: 1, channels: dim, height: 1, width: sp2 });
    let h3 = g3.slice(p3, [0, 0, 0, 0], [1, dim, 1, seq]);
    let h3r = g3.reshape(h3, Shape { batch: 1, channels: 1, height: dim, width: seq });
    let h3t = g3.transpose(h3r, [0, 1, 3, 2]);
    let uw = g3.slice(p3, [0, 0, 0, seq], [1, dim, 1, hidden]);
    let ur = g3.reshape(uw, Shape { batch: 1, channels: 1, height: dim, width: hidden });
    let _ = g3.matrix_multiplication(h3t, ur, false, false);
    let e3 = g3.compile(NSQualityOfService::Default).unwrap();

    // Down
    let sp4 = seq + dim;
    let mut g4 = Graph::new();
    let p4 = g4.placeholder(Shape { batch: 1, channels: hidden, height: 1, width: sp4 });
    let h4 = g4.slice(p4, [0, 0, 0, 0], [1, hidden, 1, seq]);
    let h4r = g4.reshape(h4, Shape { batch: 1, channels: 1, height: hidden, width: seq });
    let h4t = g4.transpose(h4r, [0, 1, 3, 2]);
    let dw = g4.slice(p4, [0, 0, 0, seq], [1, hidden, 1, dim]);
    let dr = g4.reshape(dw, Shape { batch: 1, channels: 1, height: hidden, width: dim });
    let _ = g4.matrix_multiplication(h4t, dr, false, false);
    let e4 = g4.compile(NSQualityOfService::Default).unwrap();

    (e1, e2, e3, e4)
}

fn main() {
    let seq = 64;
    
    for (label, dim, hidden, num_layers) in [
        ("tiny (dim=256)", 256, 1024, 12),
        ("small (dim=512)", 512, 2048, 24),
        ("medium (dim=1024)", 1024, 4096, 24),
        ("Qwen3.5-4B (dim=2560)", 2560, 9216, 48),
    ] {
        print!("{label}: ");
        
        let (e1, e2, e3, e4) = build_layer_kernels(dim, hidden, seq);
        
        let sp1 = seq + 2 * dim;
        let sp2 = seq + hidden;
        let sp4 = seq + dim;
        
        let in1 = TensorData::with_f32(&vec![0.01; dim * sp1], Shape { batch: 1, channels: dim, height: 1, width: sp1 });
        let out1 = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });
        let in2 = TensorData::with_f32(&vec![0.01; dim * sp2], Shape { batch: 1, channels: dim, height: 1, width: sp2 });
        let out2 = TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: seq });
        let in3 = TensorData::with_f32(&vec![0.01; dim * sp2], Shape { batch: 1, channels: dim, height: 1, width: sp2 });
        let out3 = TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: seq });
        let in4 = TensorData::with_f32(&vec![0.01; hidden * sp4], Shape { batch: 1, channels: hidden, height: 1, width: sp4 });
        let out4 = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });

        // Warmup
        for _ in 0..3 {
            for _ in 0..num_layers {
                e1.run_cached_direct(&[&in1], &[&out1]).unwrap();
                e2.run_cached_direct(&[&in2], &[&out2]).unwrap();
                e3.run_cached_direct(&[&in3], &[&out3]).unwrap();
                e4.run_cached_direct(&[&in4], &[&out4]).unwrap();
            }
        }

        // Generate tokens: each token = num_layers × 4 dispatches
        let num_tokens = 20;
        let start = Instant::now();
        for _ in 0..num_tokens {
            for _ in 0..num_layers {
                e1.run_cached_direct(&[&in1], &[&out1]).unwrap();
                e2.run_cached_direct(&[&in2], &[&out2]).unwrap();
                e3.run_cached_direct(&[&in3], &[&out3]).unwrap();
                e4.run_cached_direct(&[&in4], &[&out4]).unwrap();
            }
        }
        let dur = start.elapsed();
        let tok_per_sec = num_tokens as f64 / dur.as_secs_f64();
        let ms_per_token = dur.as_secs_f64() * 1000.0 / num_tokens as f64;
        
        let params_m = (num_layers as f64 * (2.0 * dim as f64 * dim as f64 + 3.0 * dim as f64 * hidden as f64)) / 1_000_000.0;
        
        println!("{tok_per_sec:.1} tok/s ({ms_per_token:.1}ms/tok, ~{params_m:.0}M params)");
    }
    
    println!("\nGPU Qwen3.5-4B-4bit: 95 tok/s (10.5ms/tok)");
}
