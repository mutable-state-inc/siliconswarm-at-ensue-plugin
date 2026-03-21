use ane::{Graph, Shape};
use objc2_foundation::NSQualityOfService;
fn main() {
    let mut g = Graph::new();
    let x = g.placeholder(Shape { channels: 64, height: 1, width: 64, batch: 1 });
    let w = g.constant(&vec![0.001f32; 128 * 64], Shape { channels: 128, height: 64, width: 1, batch: 1 });
    let _y = g.convolution_2d_1x1(x, w, None);
    match g.compile(NSQualityOfService::Default) {
        Ok(_) => println!("conv 1x1: OK"),
        Err(e) => println!("conv 1x1: FAILED: {e}"),
    }
}
