/// Test: can we beat fp16 matmul by doing int8 dequant + matmul on ANE?
/// 
/// Approach: store weights as int8 (half the bandwidth of fp16), 
/// cast to fp16 on ANE, then matmul.
///
/// The MIL cast op should work: cast(dtype="fp16", x=int8_tensor)
/// But we need to check if the graph builder supports int8 tensors.
///
/// Alternative: store weights as fp16 but at HALF the width (group of 2),
/// effectively doing 2x less bandwidth. Use reshape tricks.

use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn build_matmul(ic: usize, oc: usize, seq: usize) -> Graph {
    let sp = seq + oc;
    let mut g = Graph::new();
    let input = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: sp });
    let acts = g.slice(input, [0, 0, 0, 0], [1, ic, 1, seq]);
    let wts = g.slice(input, [0, 0, 0, seq], [1, ic, 1, oc]);
    let acts_r = g.reshape(acts, Shape { batch: 1, channels: 1, height: ic, width: seq });
    let acts_t = g.transpose(acts_r, [0, 1, 3, 2]);
    let wts_r = g.reshape(wts, Shape { batch: 1, channels: 1, height: ic, width: oc });
    let mm = g.matrix_multiplication(acts_t, wts_r, false, false);
    let mm_t = g.transpose(mm, [0, 1, 3, 2]);
    let _out = g.reshape(mm_t, Shape { batch: 1, channels: oc, height: 1, width: seq });
    g
}

fn main() {
    let dim = 2560;
    let seq = 64;
    
    // Test different "hidden" sizes to simulate bandwidth reduction
    // Full: 9216 (47MB per matmul at fp16)
    // Half: 4608 (23MB) — simulates 8-bit weights dequantized
    // Quarter: 2304 (12MB) — simulates 4-bit weights dequantized
    
    println!("Benchmarking matmul at different effective bandwidths...\n");
    
    for (label, oc) in [
        ("full fp16 (9216)", 9216),
        ("half bandwidth (4608)", 4608),
        ("quarter bandwidth (2304)", 2304),
        ("eighth bandwidth (1152)", 1152),
    ] {
        let exec = build_matmul(dim, oc, seq).compile(NSQualityOfService::Default).unwrap();
        let sp = seq + oc;
        let input = TensorData::with_f32(&vec![0.01f32; dim * sp],
            Shape { batch: 1, channels: dim, height: 1, width: sp });
        let output = TensorData::new(Shape { batch: 1, channels: oc, height: 1, width: seq });
        
        for _ in 0..5 { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
        let n = 200;
        let start = Instant::now();
        for _ in 0..n { exec.run_cached_direct(&[&input], &[&output]).unwrap(); }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let weight_mb = (dim * oc * 2) as f64 / 1_000_000.0;
        let bw_gbs = weight_mb / ms * 1000.0 / 1000.0;
        
        println!("{label}: {ms:.3}ms ({weight_mb:.1}MB weights, {bw_gbs:.1} GB/s effective)");
    }
    
    println!("\nIf we can dequantize 4-bit→fp16 on ANE faster than reading fp16,");
    println!("we could beat the current matmul by reducing bandwidth.");
    println!("\nAlternatively: pre-dequantize on CPU/NEON into IOSurface,");
    println!("which shifts the bottleneck to NEON dequant speed vs ANE eval speed.");
}
