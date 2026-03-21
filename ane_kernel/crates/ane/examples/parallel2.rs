use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;
use std::thread;
use std::time::Duration;
use std::sync::Arc;

fn main() {
    let dim=256;let hidden=256;let seq=64;let nl=8;
    let wpl=4*dim+3*hidden;let sp=seq+nl*wpl;
    let mut g=Graph::new();
    let packed=g.placeholder(Shape{batch:1,channels:dim,height:1,width:sp});
    let mut h=g.slice(packed,[0,0,0,0],[1,dim,1,seq]);
    let mut wo=seq;
    for _ in 0..nl {
        let hr=g.reshape(h,Shape{batch:1,channels:1,height:dim,width:seq});
        let ht=g.transpose(hr,[0,1,3,2]);
        let qw=g.slice(packed,[0,0,0,wo],[1,dim,1,dim]);wo+=dim*4;
        let qr=g.reshape(qw,Shape{batch:1,channels:1,height:dim,width:dim});
        let q=g.matrix_multiplication(ht,qr,false,false);
        let qt=g.transpose(q,[0,1,3,2]);
        let ao=g.reshape(qt,Shape{batch:1,channels:dim,height:1,width:seq});
        h=g.addition(h,ao);
        let hr2=g.reshape(h,Shape{batch:1,channels:1,height:dim,width:seq});
        let ht2=g.transpose(hr2,[0,1,3,2]);
        let gw=g.slice(packed,[0,0,0,wo],[1,dim,1,hidden]);wo+=hidden;
        let gr=g.reshape(gw,Shape{batch:1,channels:1,height:dim,width:hidden});
        let gate=g.matrix_multiplication(ht2,gr,false,false);
        let uw=g.slice(packed,[0,0,0,wo],[1,dim,1,hidden]);wo+=hidden;
        let ur=g.reshape(uw,Shape{batch:1,channels:1,height:dim,width:hidden});
        let up=g.matrix_multiplication(ht2,ur,false,false);
        let gs=g.sigmoid(gate);let gl=g.multiplication(gate,gs);let mix=g.multiplication(gl,up);
        let dw=g.slice(packed,[0,0,0,wo],[1,dim,1,dim]);wo+=dim;
        let dr=g.reshape(dw,Shape{batch:1,channels:1,height:hidden,width:dim});
        let f=g.matrix_multiplication(mix,dr,false,false);
        let ft=g.transpose(f,[0,1,3,2]);
        let fo=g.reshape(ft,Shape{batch:1,channels:dim,height:1,width:seq});
        h=g.addition(h,fo);
    }
    let exec=Arc::new(g.compile(NSQualityOfService::Default).unwrap());
    // TensorData is !Send, so we need to create per-thread
    
    // Measure ANE alone
    {
        let input=TensorData::with_f32(&vec![0.01;dim*sp],Shape{batch:1,channels:dim,height:1,width:sp});
        let output=TensorData::new(Shape{batch:1,channels:dim,height:1,width:seq});
        for _ in 0..5{exec.run_cached_direct(&[&input],&[&output]).unwrap();}
        let n=500;let start=Instant::now();
        for _ in 0..n{exec.run_cached_direct(&[&input],&[&output]).unwrap();}
        let ane_ms=start.elapsed().as_secs_f64()*1000.0/n as f64;
        println!("ANE alone (8L): {ane_ms:.3}ms");
    }
    
    // GPU alone (sleep simulating 10.5ms compute)
    let gpu_ms=10.5f64;
    println!("GPU alone: {gpu_ms:.1}ms");
    
    // Parallel: main thread does ANE, spawned thread does CPU-intensive work (not sleep)
    // Use a spin-wait to simulate GPU busy time
    {
        let input=TensorData::with_f32(&vec![0.01;dim*sp],Shape{batch:1,channels:dim,height:1,width:sp});
        let output=TensorData::new(Shape{batch:1,channels:dim,height:1,width:seq});
        let n=200;
        let start=Instant::now();
        for _ in 0..n {
            // "GPU" work on another thread — spin wait to simulate busy GPU
            let gpu_start = Instant::now();
            let handle = thread::spawn(move || {
                while gpu_start.elapsed() < Duration::from_micros((gpu_ms * 1000.0) as u64) {
                    std::hint::spin_loop();
                }
            });
            // ANE work on main thread — runs on different hardware
            exec.run_cached_direct(&[&input],&[&output]).unwrap();
            handle.join().unwrap();
        }
        let par_ms=start.elapsed().as_secs_f64()*1000.0/n as f64;
        let seq_ms = 1.1 + gpu_ms; // ANE + GPU sequential
        println!("Sequential:  {seq_ms:.1}ms");
        println!("PARALLEL:    {par_ms:.2}ms");
        println!("Overlap:     {:.0}%", (1.0-(par_ms/seq_ms))*100.0);
        println!("Effective:   {:.0} tok/s (spec decode k=1 α=0.7)", 1.7/par_ms*1000.0);
    }
}
