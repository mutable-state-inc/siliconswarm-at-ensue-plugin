use ane::{Graph, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use std::time::Instant;
fn main() {
    let dim=256; let hidden=256; let seq=64; let nl=8;
    let wpl=4*dim+3*hidden; let sp=seq+nl*wpl;
    if sp > 16384 { eprintln!("sp={sp} > 16384, skipping"); std::process::exit(1); }
    let mut g=Graph::new();
    let packed=g.placeholder(Shape{batch:1,channels:dim,height:1,width:sp});
    let mut h=g.slice(packed,[0,0,0,0],[1,dim,1,seq]);
    let mut wo=seq;
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
        let gs=g.sigmoid(gate); let gl=g.multiplication(gate,gs); let mix=g.multiplication(gl,up);
        let dw=g.slice(packed,[0,0,0,wo],[1,dim,1,dim]); wo+=dim;
        let dr=g.reshape(dw,Shape{batch:1,channels:1,height:hidden,width:dim});
        let f=g.matrix_multiplication(mix,dr,false,false);
        let ft=g.transpose(f,[0,1,3,2]);
        let fo=g.reshape(ft,Shape{batch:1,channels:dim,height:1,width:seq});
        h=g.addition(h,fo);
    }
    let exec=g.compile(NSQualityOfService::Default).unwrap();
    let input=TensorData::with_f32(&vec![0.01f32;dim*sp],Shape{batch:1,channels:dim,height:1,width:sp});
    let output=TensorData::new(Shape{batch:1,channels:dim,height:1,width:seq});
    for _ in 0..3{exec.run_cached_direct(&[&input],&[&output]).unwrap();}
    let n=200; let start=Instant::now();
    for _ in 0..n{exec.run_cached_direct(&[&input],&[&output]).unwrap();}
    let ms=start.elapsed().as_secs_f64()*1000.0/n as f64;
    let dispatches=(48.0/nl as f64).ceil();
    println!("{nl}L: {ms:.3}ms ({:.3}ms/L) → 48L={:.1}ms ({:.0} tok/s) [{} dispatches]",ms/nl as f64,dispatches*ms,1000.0/(dispatches*ms),dispatches as u32);
}
