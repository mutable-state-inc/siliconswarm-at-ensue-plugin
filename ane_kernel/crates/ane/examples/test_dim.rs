use ane::{Graph, Shape};
use objc2_foundation::NSQualityOfService;
fn main() {
    let mut g = Graph::new();
    let x = g.placeholder(Shape { channels: 2560, height: 1, width: 64, batch: 1 });
    let w = g.constant(&vec![0.001f32; 10240 * 2560], Shape { channels: 10240, height: 2560, width: 1, batch: 1 });
    let _y = g.convolution_2d_1x1(x, w, None);
    match g.compile(NSQualityOfService::Default) {
        Ok(_) => println!("dim={} hidden={}: OK", 2560, 10240),
        Err(e) => println!("dim={} hidden={}: FAILED: {e}", 2560, 10240),
    }
}
