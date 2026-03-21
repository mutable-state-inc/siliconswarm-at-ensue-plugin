use ane::{Graph, Shape};
use objc2_foundation::NSQualityOfService;

fn main() {
    let mut g = Graph::new();
    let x = g.placeholder(Shape { channels: 64, height: 1, width: 64, batch: 1 });
    let _y = g.addition(x, x);

    println!("Compiling graph on ANE...");
    match g.compile(NSQualityOfService::Default) {
        Ok(exec) => {
            println!("COMPILE SUCCEEDED!");
            drop(exec);
        }
        Err(e) => {
            println!("COMPILE FAILED: {e}");
        }
    }
}
