use ane::{Graph, Shape};
use objc2_foundation::NSQualityOfService;
fn main() {
    let mut g = Graph::new();
    let x = g.placeholder(Shape { batch: 1, channels: 256, height: 1, width: 64 });
    match "power" {
        "mul_sig_add" => { let s=g.sigmoid(x); let m=g.multiplication(x,s); let _=g.addition(x,m); }
        "reduce_sum" => { let _=g.reduce_sum(x, 1); }
        "reduce_mean" => { let _=g.reduce_mean(x, 1); }
        "rsqrt" => { let _=g.reciprocal_square_root(x); }
        "power" => { let h=g.constant_with_scalar(-0.5,Shape{batch:1,channels:1,height:1,width:1}); let _=g.power(x,h); }
        _ => {}
    }
    match g.compile(NSQualityOfService::Default) {
        Ok(_) => println!("power: OK"),
        Err(_) => println!("power: FAIL"),
    }
}
