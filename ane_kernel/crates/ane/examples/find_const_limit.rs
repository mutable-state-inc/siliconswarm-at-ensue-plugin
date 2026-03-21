use ane::{Graph, Shape};
use objc2_foundation::NSQualityOfService;

fn main() {
    for (ic, oc) in [
        (64, 128), (128, 256), (256, 512), (256, 1024), (256, 2048),
        (512, 512), (512, 1024), (512, 2048),
        (768, 768), (768, 1024),
        (1024, 1024),
    ] {
        let seq = 64;
        let mut g = Graph::new();
        let x = g.placeholder(Shape { batch: 1, channels: 1, height: 1, width: ic * seq });
        let w = g.constant(&vec![0.001f32; oc * ic],
            Shape { batch: 1, channels: 1, height: ic, width: oc });
        let x_r = g.reshape(x, Shape { batch: 1, channels: 1, height: seq, width: ic });
        let _mm = g.matrix_multiplication(x_r, w, false, false);
        
        let weight_kb = ic * oc * 2 / 1024; // fp16 size in KB
        match g.compile(NSQualityOfService::Default) {
            Ok(_) => println!("{ic:4}x{oc:4} ({weight_kb:5}KB): OK"),
            Err(_) => println!("{ic:4}x{oc:4} ({weight_kb:5}KB): FAIL"),
        }
    }
}
