/// Tiled constant-weight matmul: split IC into chunks of 256, 
/// do parallel small matmuls, accumulate results.
use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;

fn main() {
    let dim = 2560;      // split into 10 × 256
    let hidden = 9216;
    let seq = 64;
    let tile_ic = 256;   // max IC for constant weights
    let n_tiles = dim / tile_ic;

    println!("Building {n_tiles} tiled constant-weight matmuls ({tile_ic}x{hidden} each)...");
    
    // Build N tile kernels (each: 256 → 9216)
    let mut execs = Vec::new();
    let start = Instant::now();
    for t in 0..n_tiles {
        let mut g = Graph::new();
        let x = g.placeholder(Shape { batch: 1, channels: 1, height: 1, width: tile_ic * seq });
        let w = g.constant(&vec![0.001f32; hidden * tile_ic],
            Shape { batch: 1, channels: 1, height: tile_ic, width: hidden });
        let x_r = g.reshape(x, Shape { batch: 1, channels: 1, height: seq, width: tile_ic });
        let mm = g.matrix_multiplication(x_r, w, false, false);
        let mm_t = g.transpose(mm, [0, 1, 3, 2]);
        let _out = g.reshape(mm_t, Shape { batch: 1, channels: hidden, height: 1, width: seq });
        
        let exec = g.compile(NSQualityOfService::Default)
            .unwrap_or_else(|e| panic!("tile {t} compile failed: {e}"));
        execs.push(exec);
    }
    println!("All {n_tiles} tiles compiled in {:?}", start.elapsed());

    // Create I/O
    let mut inputs: Vec<TensorData> = Vec::new();
    let mut outputs: Vec<TensorData> = Vec::new();
    for _ in 0..n_tiles {
        inputs.push(TensorData::with_f32(&vec![0.01f32; tile_ic * seq],
            Shape { batch: 1, channels: 1, height: 1, width: tile_ic * seq }));
        outputs.push(TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: seq }));
    }

    // Warmup
    for _ in 0..3 {
        for t in 0..n_tiles {
            execs[t].run_cached_direct(&[&inputs[t]], &[&outputs[t]]).unwrap();
        }
    }

    // Benchmark: all tiles sequentially (simulate tiled matmul)
    let n = 100;
    let start = Instant::now();
    for _ in 0..n {
        for t in 0..n_tiles {
            execs[t].run_cached_direct(&[&inputs[t]], &[&outputs[t]]).unwrap();
        }
    }
    let total_ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let per_tile_ms = total_ms / n_tiles as f64;
    let weight_kb = tile_ic * hidden * 2 / 1024;

    // Compare with single dynamic-weight matmul
    println!("\n=== Tiled const-weight vs dynamic-weight matmul ===");
    println!("Tiled ({n_tiles}x{tile_ic}x{hidden}): {total_ms:.2}ms total ({per_tile_ms:.3}ms/tile, {weight_kb}KB/tile)");
    println!("Dynamic (2560x9216):        ~7.0ms (one dispatch, 47MB weights)");
    println!("Speedup: {:.1}x", 7.0 / total_ms);
    println!("\nFull FFN (3 matmuls): {:.1}ms tiled vs ~21ms dynamic", 3.0 * total_ms);
    println!("48 layers FFN: {:.0}ms tiled", 48.0 * 3.0 * total_ms);
}
