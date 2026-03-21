use ane::{Graph, Shape};
use objc2_foundation::NSQualityOfService;
fn main() {
    let mut g = Graph::new();
    let x = g.placeholder(Shape { batch: 1, channels: 2560, height: 1, width: 64 });
    let sq = g.multiplication(x, x);
    let m = g.reduce_mean(sq, 1);
    let eps = g.constant_with_scalar(1e-6, Shape{batch:1,channels:1,height:1,width:1});
    let me = g.addition(m, eps);
    let rs = g.reciprocal_square_root(me);
    let n = g.multiplication(x, rs);
    let sg = g.sigmoid(n);
    let sl = g.multiplication(n, sg);
    let _ = g.addition(x, sl);
    match g.compile(NSQualityOfService::Default) {
        Ok(_) => println!("ch={}: OK", 2560),
        Err(_) => println!("ch={}: FAIL", 2560),
    }
}
