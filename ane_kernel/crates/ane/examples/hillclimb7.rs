use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;
const DIM: usize = 2048;
const HIDDEN: usize = 11008;
const LAYERS: usize = 36;
const SEQ: usize = 64;

fn bench(label: &str, nl: usize, build: impl FnOnce(&mut Graph, ane::Tensor) -> ane::Tensor) {
    eprint!("  {label:<55} ");
    let mut g = Graph::new();
    let h = g.placeholder(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
    let _ = build(&mut g, h);
    let exec = match g.compile(NSQualityOfService::Default) {
        Ok(e) => e,
        Err(_) => { eprintln!("FAIL"); return; }
    };
    let input = TensorData::with_f32(&vec![0.01; DIM * SEQ],
        Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
    let output = TensorData::new(Shape { batch: 1, channels: DIM, height: 1, width: SEQ });
    for _ in 0..10 { let _ = exec.run_cached_direct(&[&input], &[&output]); }
    let dispatches = (LAYERS + nl - 1) / nl;
    let tokens = 30;
    let start = Instant::now();
    for _ in 0..tokens {
        for _ in 0..dispatches { let _ = exec.run_cached_direct(&[&input], &[&output]); }
    }
    let dur = start.elapsed();
    let tok_s = tokens as f64 / dur.as_secs_f64();
    let ms_layer = dur.as_secs_f64() * 1000.0 / tokens as f64 / LAYERS as f64;
    eprintln!("{tok_s:.1} tok/s  ({ms_layer:.3}ms/L, {dispatches}d)");
}

fn split_ip(g: &mut Graph, h: ane::Tensor, ic: usize, oc: usize, s: usize) -> ane::Tensor {
    let c = oc / s;
    let p: Vec<ane::Tensor> = (0..s).map(|_| g.inner_product(h, &vec![0.01; c * ic], ic, c)).collect();
    g.concat(&p, 1)
}
fn split_down(g: &mut Graph, h: ane::Tensor, ic: usize, oc: usize, s: usize) -> ane::Tensor {
    let c = oc / s;
    let p: Vec<ane::Tensor> = (0..s).map(|_| g.inner_product(h, &vec![0.01; c * ic], ic, c)).collect();
    g.concat(&p, 1)
}

fn main() {
    eprintln!("=== HILL CLIMB R7: Final push ===\n");

    // Current best: split-16 all, 9L = 10.9 tok/s (2.538ms/L)

    // Try split-32
    eprintln!("--- Split-32 all ---");
    for nl in [6, 9, 12] {
        let label = format!("{nl}L split-32 all (1.4MB/chunk)");
        bench(&label, nl, |g, mut h| {
            for _ in 0..nl {
                let gate = split_ip(g, h, DIM, HIDDEN, 32);
                let up = split_ip(g, h, DIM, HIDDEN, 32);
                let gs = g.sigmoid(gate);
                let gl = g.multiplication(gate, gs);
                let mix = g.multiplication(gl, up);
                let down = split_down(g, mix, HIDDEN, DIM, 32);
                h = g.addition(h, down);
            }
            h
        });
    }

    eprintln!();

    // Try higher fusion with split-16 (18L was too big before)
    eprintln!("--- Higher fusion with split-16 ---");
    for nl in [12, 18, 36] {
        let label = format!("{nl}L split-16 all");
        bench(&label, nl, |g, mut h| {
            for _ in 0..nl {
                let gate = split_ip(g, h, DIM, HIDDEN, 16);
                let up = split_ip(g, h, DIM, HIDDEN, 16);
                let gs = g.sigmoid(gate);
                let gl = g.multiplication(gate, gs);
                let mix = g.multiplication(gl, up);
                let down = split_down(g, mix, HIDDEN, DIM, 16);
                h = g.addition(h, down);
            }
            h
        });
    }

    eprintln!();

    // Try higher fusion with split-32
    eprintln!("--- Higher fusion with split-32 ---");
    for nl in [12, 18, 36] {
        let label = format!("{nl}L split-32 all");
        bench(&label, nl, |g, mut h| {
            for _ in 0..nl {
                let gate = split_ip(g, h, DIM, HIDDEN, 32);
                let up = split_ip(g, h, DIM, HIDDEN, 32);
                let gs = g.sigmoid(gate);
                let gl = g.multiplication(gate, gs);
                let mix = g.multiplication(gl, up);
                let down = split_down(g, mix, HIDDEN, DIM, 32);
                h = g.addition(h, down);
            }
            h
        });
    }

    eprintln!();

    // The nuclear option: split-16 all, 36L (entire model in ONE dispatch)
    eprintln!("--- FULL MODEL in one dispatch ---");
    bench("36L split-16 all (FULL MODEL, 1 dispatch)", 36, |g, mut h| {
        for _ in 0..36 {
            let gate = split_ip(g, h, DIM, HIDDEN, 16);
            let up = split_ip(g, h, DIM, HIDDEN, 16);
            let gs = g.sigmoid(gate);
            let gl = g.multiplication(gate, gs);
            let mix = g.multiplication(gl, up);
            let down = split_down(g, mix, HIDDEN, DIM, 16);
            h = g.addition(h, down);
        }
        h
    });
}
