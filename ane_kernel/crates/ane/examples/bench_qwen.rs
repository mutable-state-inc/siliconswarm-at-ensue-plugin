use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

/// Build a single Qwen3.5-4B-scale layer as one ANE graph.
/// dim=2560, hidden=9216, but using dynamic-weight matmul pattern.
/// Since spatial width max is 16384, we can fit the weights for 1 layer.
fn main() {
    let dim = 2560;
    let hidden = 9216;
    let seq = 64;

    // Weights per layer for simplified model (Q + O + gate + up + down = 5 matmuls):
    //   Q:    dim*dim   = 2560 slots (IC=dim, OC=dim)  
    //   O:    dim*dim   = 2560 slots
    //   gate: dim*hidden= 9216 slots (IC=dim, OC=hidden)
    //   up:   dim*hidden= 9216 slots
    //   down: need IC=hidden → separate graph
    // Total from dim side: 2*dim + 2*hidden = 2*2560 + 2*9216 = 23552
    // sp = seq + 23552 = 23616 > 16384! Won't fit.
    
    // Strategy: split into 2 dispatches per layer:
    //   Dispatch 1: attention (Q+O) — sp = 64 + 2*2560 = 5184 ✓
    //   Dispatch 2: FFN gate+up — sp = 64 + 2*9216 = 18496 > 16384!
    //
    // FFN still won't fit. Split FFN into 3 separate dispatches:
    //   Dispatch 2: gate — sp = 64 + 9216 = 9280 ✓  
    //   Dispatch 3: up — sp = 64 + 9216 = 9280 ✓
    //   Dispatch 4: down (IC=hidden) — sp = 64 + 2560 = 2624 ✓
    //
    // Total: 4 dispatches per layer × 48 layers = 192 dispatches
    // At ~0.13ms overhead each... 192 × 0.13 = 25ms minimum overhead
    
    // Better: fuse Q+O into one graph, gate+up into one graph
    // Q+O: sp = 64 + 2*2560 = 5184 ✓
    // gate+up: sp = 64 + 2*9216 = 18496 > 16384 ✗
    // So gate and up must be separate.
    
    // Let's benchmark the 4 dispatches:
    
    // 1. Attention: Q + O projection (2 matmuls from dim)
    println!("Building Qwen3.5-4B layer kernels (dim={dim}, hidden={hidden})...\n");
    
    let attn_sp = seq + 2 * dim; // 5184
    let mut g1 = Graph::new();
    let p1 = g1.placeholder(Shape { batch: 1, channels: dim, height: 1, width: attn_sp });
    let h1 = g1.slice(p1, [0, 0, 0, 0], [1, dim, 1, seq]);
    let h1_r = g1.reshape(h1, Shape { batch: 1, channels: 1, height: dim, width: seq });
    let h1_t = g1.transpose(h1_r, [0, 1, 3, 2]);
    // Q
    let qw = g1.slice(p1, [0, 0, 0, seq], [1, dim, 1, dim]);
    let qr = g1.reshape(qw, Shape { batch: 1, channels: 1, height: dim, width: dim });
    let q = g1.matrix_multiplication(h1_t, qr, false, false);
    // O 
    let ow = g1.slice(p1, [0, 0, 0, seq + dim], [1, dim, 1, dim]);
    let or2 = g1.reshape(ow, Shape { batch: 1, channels: 1, height: dim, width: dim });
    let o = g1.matrix_multiplication(q, or2, false, false);
    let ot = g1.transpose(o, [0, 1, 3, 2]);
    let oout = g1.reshape(ot, Shape { batch: 1, channels: dim, height: 1, width: seq });
    let _r1 = g1.addition(h1, oout); // residual
    
    let start = Instant::now();
    let exec1 = g1.compile(NSQualityOfService::Default).unwrap();
    println!("  attn (Q+O): compiled in {:?}, sp={attn_sp}", start.elapsed());
    
    // 2. FFN gate projection (dim -> hidden)
    let gate_sp = seq + hidden; // 9280
    let mut g2 = Graph::new();
    let p2 = g2.placeholder(Shape { batch: 1, channels: dim, height: 1, width: gate_sp });
    let h2 = g2.slice(p2, [0, 0, 0, 0], [1, dim, 1, seq]);
    let h2_r = g2.reshape(h2, Shape { batch: 1, channels: 1, height: dim, width: seq });
    let h2_t = g2.transpose(h2_r, [0, 1, 3, 2]);
    let gw = g2.slice(p2, [0, 0, 0, seq], [1, dim, 1, hidden]);
    let gr = g2.reshape(gw, Shape { batch: 1, channels: 1, height: dim, width: hidden });
    let gate = g2.matrix_multiplication(h2_t, gr, false, false);
    // Apply sigmoid for SiLU (silu = x * sigmoid(x))
    let gsig = g2.sigmoid(gate);
    let _gsilu = g2.multiplication(gate, gsig);
    
    let start = Instant::now();
    let exec2 = g2.compile(NSQualityOfService::Default).unwrap();
    println!("  gate+silu: compiled in {:?}, sp={gate_sp}", start.elapsed());
    
    // 3. FFN up projection (dim -> hidden)  
    let up_sp = seq + hidden;
    let mut g3 = Graph::new();
    let p3 = g3.placeholder(Shape { batch: 1, channels: dim, height: 1, width: up_sp });
    let h3 = g3.slice(p3, [0, 0, 0, 0], [1, dim, 1, seq]);
    let h3_r = g3.reshape(h3, Shape { batch: 1, channels: 1, height: dim, width: seq });
    let h3_t = g3.transpose(h3_r, [0, 1, 3, 2]);
    let uw = g3.slice(p3, [0, 0, 0, seq], [1, dim, 1, hidden]);
    let ur = g3.reshape(uw, Shape { batch: 1, channels: 1, height: dim, width: hidden });
    let _up = g3.matrix_multiplication(h3_t, ur, false, false);
    
    let start = Instant::now();
    let exec3 = g3.compile(NSQualityOfService::Default).unwrap();
    println!("  up: compiled in {:?}, sp={up_sp}", start.elapsed());
    
    // 4. FFN down projection (hidden -> dim)
    let down_sp = seq + dim;
    let mut g4 = Graph::new();
    let p4 = g4.placeholder(Shape { batch: 1, channels: hidden, height: 1, width: down_sp });
    let h4 = g4.slice(p4, [0, 0, 0, 0], [1, hidden, 1, seq]);
    let h4_r = g4.reshape(h4, Shape { batch: 1, channels: 1, height: hidden, width: seq });
    let h4_t = g4.transpose(h4_r, [0, 1, 3, 2]);
    let dw = g4.slice(p4, [0, 0, 0, seq], [1, hidden, 1, dim]);
    let dr = g4.reshape(dw, Shape { batch: 1, channels: 1, height: hidden, width: dim });
    let _down = g4.matrix_multiplication(h4_t, dr, false, false);
    
    let start = Instant::now();
    let exec4 = g4.compile(NSQualityOfService::Default).unwrap();
    println!("  down: compiled in {:?}, sp={down_sp}", start.elapsed());
    
    // Create I/O tensors
    let in1 = TensorData::with_f32(&vec![0.01; dim * attn_sp], Shape { batch: 1, channels: dim, height: 1, width: attn_sp });
    let out1 = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });
    let in2 = TensorData::with_f32(&vec![0.01; dim * gate_sp], Shape { batch: 1, channels: dim, height: 1, width: gate_sp });
    let out2 = TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: seq });
    let in3 = TensorData::with_f32(&vec![0.01; dim * up_sp], Shape { batch: 1, channels: dim, height: 1, width: up_sp });
    let out3 = TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: seq });
    let in4 = TensorData::with_f32(&vec![0.01; hidden * down_sp], Shape { batch: 1, channels: hidden, height: 1, width: down_sp });
    let out4 = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });

    // Warmup
    println!("\nWarming up...");
    for _ in 0..3 {
        exec1.run_cached_direct(&[&in1], &[&out1]).unwrap();
        exec2.run_cached_direct(&[&in2], &[&out2]).unwrap();
        exec3.run_cached_direct(&[&in3], &[&out3]).unwrap();
        exec4.run_cached_direct(&[&in4], &[&out4]).unwrap();
    }

    // Benchmark individual kernels
    let n = 100;
    let mut times = Vec::new();
    for (name, exec, input, output) in [
        ("attn", &exec1, &in1, &out1),
        ("gate", &exec2, &in2, &out2),
        ("up", &exec3, &in3, &out3),
        ("down", &exec4, &in4, &out4),
    ] {
        let start = Instant::now();
        for _ in 0..n { exec.run_cached_direct(&[input], &[output]).unwrap(); }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
        times.push(ms);
        println!("  {name}: {ms:.3}ms");
    }

    let layer_ms: f64 = times.iter().sum();
    let total_48 = 48.0 * layer_ms;
    
    // Benchmark full layer (all 4 sequential)
    let start = Instant::now();
    for _ in 0..n {
        exec1.run_cached_direct(&[&in1], &[&out1]).unwrap();
        exec2.run_cached_direct(&[&in2], &[&out2]).unwrap();
        exec3.run_cached_direct(&[&in3], &[&out3]).unwrap();
        exec4.run_cached_direct(&[&in4], &[&out4]).unwrap();
    }
    let sequential_ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;

    println!("\n=== QWEN3.5-4B SCALE ON ANE ===");
    println!("Per layer: {layer_ms:.2}ms (sum) / {sequential_ms:.2}ms (sequential)");
    println!("48 layers: {total_48:.1}ms ({:.1} tok/s)", 1000.0 / total_48);
    println!("48 layers seq: {:.1}ms ({:.1} tok/s)", 48.0 * sequential_ms, 1000.0 / (48.0 * sequential_ms));
    println!("GPU baseline: ~10.5ms (95 tok/s)");
    println!("\nWeight bandwidth per layer: {:.1}MB (fp16 dynamic weights)", 
        ((2*dim*dim + 2*dim*hidden + hidden*dim) * 2) as f64 / 1_000_000.0);
}
