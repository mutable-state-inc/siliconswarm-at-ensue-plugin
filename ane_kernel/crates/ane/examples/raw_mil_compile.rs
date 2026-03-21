use ane::ane_in_memory_model::ANEInMemoryModel;
use ane::ane_in_memory_model_descriptor::ANEInMemoryModelDescriptor;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_foundation::{NSData, NSDictionary, NSNumber, NSQualityOfService, NSString};
use objc2_io_surface::IOSurface;
use std::time::Instant;

fn nsdata_on_surface(data: &[u8]) -> (Retained<NSData>, Retained<IOSurface>) {
    let surface = ane::io_surface::IOSurfaceExt::with_byte_count(&(), data.len());
    ane::io_surface::IOSurfaceExt::write_bytes(&*surface, data);
    let nsdata = unsafe {
        NSData::dataWithBytesNoCopy_length_freeWhenDone(surface.baseAddress(), data.len(), false)
    };
    (nsdata, surface)
}

fn main() {
    let p = std::ffi::CString::new("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine").unwrap();
    unsafe { libc::dlopen(p.as_ptr(), libc::RTLD_NOW); }

    let dim = 2560usize;
    let hidden = 9216usize;
    let mil = format!(
"program(1.3)\n[buildInfo = dict<string, string>({{{{\"coremlc-component-MIL\", \"3510.2.1\"}}, {{\"coremlc-version\", \"3505.4.1\"}}, {{\"coremltools-component-milinternal\", \"\"}}, {{\"coremltools-version\", \"9.0\"}}}})]
{{
    func main<ios18>(tensor<fp16, [1, 1, {dim}]> x) {{
        tensor<fp16, [{hidden}, {dim}]> w = const()[name=string(\"w\"), val=tensor<fp16, [{hidden}, {dim}]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];
        tensor<fp16, [1, 1, {hidden}]> y = linear(weight=w, x=x)[name=string(\"y\")];
    }} -> (y);
}}");
    println!("MIL linear {dim}->{hidden}");
    let blob = vec![0u8; 64 + hidden * dim * 2];
    // Use simple NSData (not IOSurface backed) — just to test if linear compiles
    let mil_ns = unsafe { NSData::dataWithBytesNoCopy_length_freeWhenDone(mil.as_ptr() as *mut _, mil.len(), false) };
    let blob_ns = unsafe { NSData::dataWithBytesNoCopy_length_freeWhenDone(blob.as_ptr() as *mut _, blob.len(), false) };
    
    let offset = NSNumber::new_u64(0);
    let entry: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::from_slices(
        &[&*NSString::from_str("offset"), &*NSString::from_str("data")],
        &[offset.as_ref() as &AnyObject, blob_ns.as_ref() as &AnyObject],
    );
    let weights: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::from_slices(
        &[&*NSString::from_str("@model_path/weights/w.bin")],
        &[entry.as_ref() as &AnyObject],
    );
    
    let desc = ANEInMemoryModelDescriptor::new(&mil_ns, Some(&weights)).expect("desc");
    println!("  Descriptor OK");
    let model = ANEInMemoryModel::with_descriptor(&desc).expect("model");
    println!("  Model OK");
    if let Some(hex) = model.hex_string_identifier() {
        let dir = std::env::temp_dir().join(hex.to_string());
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join("model.mil"), mil.as_bytes());
        let wd = dir.join("weights");
        let _ = std::fs::create_dir_all(&wd);
        let _ = std::fs::write(wd.join("w.bin"), &blob);
    }
    let start = Instant::now();
    match model.compile(NSQualityOfService::Default) {
        Ok(()) => {
            println!("  COMPILE OK! ({:?})", start.elapsed());
            match model.load(NSQualityOfService::Default) {
                Ok(()) => println!("  LOAD OK!"),
                Err(e) => println!("  LOAD FAILED: {e}"),
            }
        }
        Err(e) => println!("  COMPILE FAILED: {e}"),
    }
}
