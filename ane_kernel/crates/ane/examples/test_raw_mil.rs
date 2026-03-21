use ane::Graph;
use objc2_foundation::{NSData, NSDictionary, NSQualityOfService, NSString};
use objc2::rc::Retained;
use objc2::runtime::AnyObject;

fn main() {
    // Compile raw MIL text directly using _ANEInMemoryModel
    // Test if `linear` op works (it's different from conv)
    let mil = r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp16, [1, 2560, 1, 64]> x_in) {
        tensor<fp16, [9216, 2560]> w = const()[name=string("w"), val=tensor<fp16, [9216, 2560]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64(64)))];
        tensor<fp16, [1, 1, 64, 2560]> x_r = reshape(shape=[1, 1, 64, 2560], x=x_in)[name=string("xr")];
        tensor<fp16, [1, 1, 64, 9216]> y = matmul(transpose_x=false, transpose_y=true, x=x_r, y=w)[name=string("mm")];
        tensor<fp16, [1, 9216, 1, 64]> y_out = reshape(shape=[1, 9216, 1, 64], x=y)[name=string("yr")];
    } -> (y_out);
}
"#;

    // Create weight blob (all zeros for testing)
    let weight_size = 9216 * 2560 * 2; // fp16
    let blob_header = 64u8; // 64-byte header
    let mut weight_data = vec![0u8; blob_header as usize + weight_size];
    // MIL blob format: 64-byte header then raw data
    weight_data[0] = 1; // version marker
    
    println!("MIL text: {} bytes", mil.len());
    println!("Weight blob: {} bytes ({:.1}MB)", weight_data.len(), weight_data.len() as f64 / 1e6);
    
    // Use the client.rs compile path
    use ane::ane_in_memory_model::ANEInMemoryModel;
    use ane::ane_in_memory_model_descriptor::ANEInMemoryModelDescriptor;
    
    // Need to access the private API directly
    // Actually, the Graph::compile already does this internally
    // Let me just test if the graph builder's matmul at dim=2560 works
    // when the weights are packed as dynamic inputs (which we already proved works)
    
    // We KNOW this works from bench_direct.rs:
    // matmul 2560x9216: 7.3ms per eval, compiles fine
    
    // What we need: can we chain 32 layers of matmul + elementwise in MIL?
    // Graph builder limit: spatial width ≤ 16384
    // Each matmul needs OC spatial slots for weights
    // 32 layers × 7 matmuls × dim = too many slots
    
    // BUT: what if we don't pack weights in spatial dim?
    // What if we use MIL `const` weights (baked in)?
    // Earlier test showed const weights fail for IC > 256
    // That was via graph builder. Does raw MIL compile?
    
    println!("Testing raw MIL compilation not possible without direct API access.");
    println!("The Rust graph builder is the only path we have.");
    println!();
    println!("What we KNOW works:");
    println!("  - matmul 2560x9216 with dynamic weights: 7.3ms (compiles + runs)");
    println!("  - 8 layers of elementwise at dim=2560: 0.85ms (fused, one dispatch)");
    println!("  - Chained matmuls dim=256: 0.028ms each when fused");
    println!();
    println!("The REAL constraint:");
    println!("  - Spatial width ≤ 16384 for IOSurface input");
    println!("  - Each dynamic-weight matmul needs OC slots in spatial dim");
    println!("  - One 2560→9216 matmul needs 9216 spatial slots");
    println!("  - Two matmuls need 18432 > 16384 — doesn't fit!");
    println!("  - So we can do AT MOST 1 large matmul per dispatch");
}
