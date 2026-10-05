// SPDX-License-Identifier: Apache-2.0
use std::path::PathBuf;
use uairt::{as_bytes, as_bytes_mut, DType, Domain, Engine, Status};

#[test]
fn struct_layout_matches_the_c_header() {
    let mut tensor: uairt_sys::uairt_tensor = unsafe { std::mem::zeroed() };
    unsafe { uairt_sys::uairt_tensor_init(&mut tensor) };
    assert_eq!(
        tensor.struct_size as usize,
        std::mem::size_of::<uairt_sys::uairt_tensor>()
    );
    assert_eq!(tensor.dmabuf_fd, -1);
}

#[test]
fn version_and_backends() {
    assert_eq!(uairt::version().matches('.').count(), 2);
    assert!(uairt::backends().iter().any(|b| b == "reference"));
}

#[test]
fn reference_roundtrip() {
    let engine = Engine::new("reference", &[]).unwrap();
    let mut model = engine.load_model("unused").unwrap();
    assert_eq!(model.inputs().len(), 1);
    assert_eq!(model.inputs()[0].dtype, DType::Float32);
    assert_eq!(model.inputs()[0].shape, vec![1, 4]);
    assert_eq!(model.inputs()[0].nbytes(), 16);
    let input = [1f32, 2.0, 3.0, 4.0];
    let mut output = [0f32; 4];
    model
        .run(&[as_bytes(&input)], &mut [as_bytes_mut(&mut output)])
        .unwrap();
    assert_eq!(output, input);
}

#[test]
fn run_validates_sizes_and_counts() {
    let engine = Engine::new("reference", &[]).unwrap();
    let mut model = engine.load_model("unused").unwrap();
    let mut output = [0f32; 4];
    let short = [0f32; 3];
    let err = model
        .run(&[as_bytes(&short)], &mut [as_bytes_mut(&mut output)])
        .unwrap_err();
    assert_eq!(err.status(), Status::InvalidArgument);
    let err = model
        .run(&[], &mut [as_bytes_mut(&mut output)])
        .unwrap_err();
    assert_eq!(err.status(), Status::InvalidArgument);
}

#[test]
fn errors_carry_status_and_message() {
    let err = Engine::new("no_such_backend", &[]).err().unwrap();
    assert_eq!(err.status(), Status::NotFound);
    assert!(err.message().contains("no_such_backend"));
    assert!(err.to_string().contains("not found"));
    assert!(uairt::load_backend_library("/no/such/plugin.so").is_err());
    let err = Engine::new("reference", &[("bad\0key", "x")])
        .err()
        .unwrap();
    assert_eq!(err.status(), Status::InvalidArgument);
}

#[test]
fn host_buffers_and_unsupported_dmabuf() {
    let engine = Engine::new("reference", &[]).unwrap();
    let mut model = engine.load_model("unused").unwrap();
    let mut source = engine.alloc_buffer(16, Domain::Host).unwrap();
    let dest = engine.alloc_buffer(16, Domain::Host).unwrap();
    source
        .as_bytes_mut()
        .copy_from_slice(as_bytes(&[5f32, 6.0, 7.0, 8.0]));
    model.run_buffers(&[&source], &[&dest]).unwrap();
    assert_eq!(dest.as_bytes(), as_bytes(&[5f32, 6.0, 7.0, 8.0]));
    assert_eq!(source.fd(), -1);
    let err = engine.alloc_buffer(16, Domain::DmaBuf).err().unwrap();
    assert_eq!(err.status(), Status::Unsupported);
    let err = engine.alloc_buffer(16, Domain::Pinned).err().unwrap();
    assert_eq!(err.status(), Status::Unsupported);
}

#[test]
fn onnxruntime_two_inputs_two_outputs() {
    let Ok(plugin) = std::env::var("UAIRT_TEST_ORT_PLUGIN") else {
        eprintln!("skipped: set UAIRT_TEST_ORT_PLUGIN to run");
        return;
    };
    let model_path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "../../../tests/data/add_mul.onnx",
    ]
    .iter()
    .collect();
    uairt::load_backend_library(&plugin).unwrap();
    let engine = Engine::new("onnxruntime", &[("intra_op_threads", "1")]).unwrap();
    let mut model = engine.load_model(model_path.to_str().unwrap()).unwrap();
    assert_eq!(model.inputs()[0].name, "x");
    assert_eq!(model.outputs()[1].name, "product");
    let (x, y) = ([1f32, 2.0, 3.0, 4.0], [10f32, 20.0, 30.0, 40.0]);
    let (mut sum, mut product) = ([0f32; 4], [0f32; 4]);
    model
        .run(
            &[as_bytes(&x), as_bytes(&y)],
            &mut [as_bytes_mut(&mut sum), as_bytes_mut(&mut product)],
        )
        .unwrap();
    for i in 0..4 {
        assert_eq!(sum[i], x[i] + y[i]);
        assert_eq!(product[i], x[i] * y[i]);
    }
    let bytes = std::fs::read(&model_path).unwrap();
    assert_eq!(
        engine
            .load_model_from_bytes(&bytes, None)
            .unwrap()
            .outputs()
            .len(),
        2
    );
    assert_eq!(
        engine
            .load_model("/no/such/model.onnx")
            .err()
            .unwrap()
            .status(),
        Status::Io
    );
    assert_eq!(
        Engine::new("onnxruntime", &[("no_such_option", "1")])
            .err()
            .unwrap()
            .status(),
        Status::InvalidArgument
    );
}
