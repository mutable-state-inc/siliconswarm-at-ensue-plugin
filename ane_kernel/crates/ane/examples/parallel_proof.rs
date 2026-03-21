use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;
use std::thread;
use std::time::Duration;

fn main() {
    let dim = 256; let hidden = 256; let seq = 64; let nl = 8;
    let wpl = 4*dim + 3*hidden; let sp = seq + nl*wpl;
    
    let mut g = Graph::new();
    let packed = g.placeholder(Shape{batch:1,channels:dim,height:1,width:sp});
    let mut h = g.slice(packed,[0,0,0,0],[1,dim,1,seq]);
    let mut wo = seq;
    for _ in 0..nl {
        let hr=g.reshape(h,Shape{batch:1,channels:1,height:dim,width:seq});
        let ht=g.transpose(hr,[0,1,3,2]);
        let qw=g.slice(packed,[0,0,0,wo],[1,dim,1,dim]); wo+=dim*4;
        let qr=g.reshape(qw,Shape{batch:1,channels:1,height:dim,width:dim});
        let q=g.matrix_multiplication(ht,qr,false,false);
        let qt=g.transpose(q,[0,1,3,2]);
        let ao=g.reshape(qt,Shape{batch:1,channels:dim,height:1,width:seq});
        h=g.addition(h,ao);
        let hr2=g.reshape(h,Shape{batch:1,channels:1,height:dim,width:seq});
        let ht2=g.transpose(hr2,[0,1,3,2]);
        let gw=g.slice(packed,[0,0,0,wo],[1,dim,1,hidden]); wo+=hidden;
        let gr=g.reshape(gw,Shape{batch:1,channels:1,height:dim,width:hidden});
        let gate=g.matrix_multiplication(ht2,gr,false,false);
        let uw=g.slice(packed,[0,0,0,wo],[1,dim,1,hidden]); wo+=hidden;
        let ur=g.reshape(uw,Shape{batch:1,channels:1,height:dim,width:hidden});
        let up=g.matrix_multiplication(ht2,ur,false,false);
        let gs=g.sigmoid(gate);let gl=g.multiplication(gate,gs);let mix=g.multiplication(gl,up);
        let dw=g.slice(packed,[0,0,0,wo],[1,dim,1,dim]); wo+=dim;
        let dr=g.reshape(dw,Shape{batch:1,channels:1,height:hidden,width:dim});
        let f=g.matrix_multiplication(mix,dr,false,false);
        let ft=g.transpose(f,[0,1,3,2]);
        let fo=g.reshape(ft,Shape{batch:1,channels:dim,height:1,width:seq});
        h=g.addition(h,fo);
    }
    let exec = g.compile(NSQualityOfService::Default).unwrap();
    let input = TensorData::with_f32(&vec![0.01;dim*sp],Shape{batch:1,channels:dim,height:1,width:sp});
    let output = TensorData::new(Shape{batch:1,channels:dim,height:1,width:seq});
    for _ in 0..5{exec.run_cached_direct(&[&input],&[&output]).unwrap();}
    
    // ANE alone: 2 dispatches = ~16 layers ≈ full draft model
    let n = 500;
    let start = Instant::now();
    for _ in 0..n{exec.run_cached_direct(&[&input],&[&output]).unwrap();}
    let ane_one = start.elapsed().as_secs_f64()*1000.0/n as f64;
    // Two dispatches for ~16 layers
    let start = Instant::now();
    for _ in 0..n{
        exec.run_cached_direct(&[&input],&[&output]).unwrap();
        exec.run_cached_direct(&[&input],&[&output]).unwrap();
    }
    let ane_two = start.elapsed().as_secs_f64()*1000.0/n as f64;
    
    let gpu_ms = 10.5f64;
    
    // PARALLEL: ANE runs while "GPU" works (simulated)
    let start = Instant::now();
    for _ in 0..n {
        let handle = thread::spawn(move || thread::sleep(Duration::from_micros((gpu_ms*1000.0) as u64)));
        // ANE draft: 2 dispatches covering full draft model
        exec.run_cached_direct(&[&input],&[&output]).unwrap();
        exec.run_cached_direct(&[&input],&[&output]).unwrap();
        handle.join().unwrap();
    }
    let parallel = start.elapsed().as_secs_f64()*1000.0/n as f64;

    println!("=== GPU + ANE PARALLEL PROOF (dim={dim}, {nl}L/dispatch) ===");
    println!("ANE 1 dispatch:  {ane_one:.2}ms");
    println!("ANE 2 dispatches: {ane_two:.2}ms (full draft model)");
    println!("GPU simulated:   {gpu_ms:.2}ms");
    println!("Sequential:      {:.2}ms", ane_two + gpu_ms);
    println!("PARALLEL:        {parallel:.2}ms");
    println!("Overlap:         {:.0}%", (1.0 - parallel / (ane_two + gpu_ms)) * 100.0);
    println!();
    println!("Speculative decode (k=1, α=0.7): {:.0} tok/s ({:.2}x over GPU-only 95)",
        1.7 / parallel * 1000.0, 1.7 / parallel * 1000.0 / 95.0);
}
