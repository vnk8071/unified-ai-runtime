// SPDX-License-Identifier: Apache-2.0
//! A small safe layer over the UAIRT C API. The declarations in `ffi` are written by hand from `include/uairt/uairt.h` and
//! `include/uairt/uairt_llm.h` (no bindgen), so keep them in step with those headers; `struct_size` fields make a mismatch fail
//! loudly instead of silently.
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fmt;
use std::ptr;

pub mod ffi {
    use super::*;

    pub const UAIRT_OK: i32 = 0;
    pub const MAX_RANK: usize = 8;

    #[repr(C)]
    pub struct Engine {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct Model {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct UairtOption {
        pub key: *const c_char,
        pub value: *const c_char,
    }

    #[repr(C)]
    pub struct ModelSource {
        pub struct_size: u32,
        pub path: *const c_char,
        pub data: *const c_void,
        pub size: usize,
        pub format: *const c_char,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Tensor {
        pub struct_size: u32,
        pub dtype: i32,
        pub rank: u32,
        pub dims: [i64; MAX_RANK],
        pub quant_scale: f32,
        pub quant_zero_point: i32,
        pub domain: u32,
        pub data: *mut c_void,
        pub dmabuf_fd: i32,
        pub nbytes: usize,
        pub name: *const c_char,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Sampling {
        pub struct_size: usize,
        pub temperature: f32,
        pub top_k: i32,
        pub top_p: f32,
        pub seed: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct GenerateOptions {
        pub struct_size: usize,
        pub max_tokens: usize,
        pub sampling: Sampling,
        pub stop_tokens: *const i32,
        pub num_stop_tokens: usize,
        pub add_special: c_int,
        pub ignore_end_of_generation: c_int,
    }

    #[repr(C)]
    pub struct GenerateResult {
        pub struct_size: usize,
        pub num_tokens: usize,
        pub finish_reason: c_int,
    }

    pub type TextCallback = unsafe extern "C" fn(text: *const c_char, len: usize, user_data: *mut c_void) -> c_int;

    extern "C" {
        pub fn uairt_version_string() -> *const c_char;
        pub fn uairt_status_string(status: i32) -> *const c_char;
        pub fn uairt_last_error() -> *const c_char;
        pub fn uairt_tensor_init(tensor: *mut Tensor);
        pub fn uairt_load_backend_library(path: *const c_char) -> i32;
        pub fn uairt_engine_create(backend: *const c_char, options: *const UairtOption, n: usize, out: *mut *mut Engine) -> i32;
        pub fn uairt_engine_destroy(engine: *mut Engine);
        pub fn uairt_model_load(engine: *mut Engine, source: *const ModelSource, out: *mut *mut Model) -> i32;
        pub fn uairt_model_destroy(model: *mut Model);
        pub fn uairt_model_num_inputs(model: *const Model) -> usize;
        pub fn uairt_model_num_outputs(model: *const Model) -> usize;
        pub fn uairt_model_input_info(model: *const Model, index: usize, out: *mut Tensor) -> i32;
        pub fn uairt_model_output_info(model: *const Model, index: usize, out: *mut Tensor) -> i32;
        pub fn uairt_model_run(model: *mut Model, inputs: *const Tensor, n_in: usize, outputs: *mut Tensor, n_out: usize) -> i32;

        pub fn uairt_sampling_default() -> Sampling;
        pub fn uairt_generate_options_default() -> GenerateOptions;
        pub fn uairt_llm_generate(
            model: *mut Model,
            session: *mut c_void,
            prompt: *const c_char,
            options: *const GenerateOptions,
            callback: Option<TextCallback>,
            user_data: *mut c_void,
            out_result: *mut GenerateResult,
        ) -> i32;
    }
}

#[derive(Debug)]
pub struct Error {
    pub status: i32,
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn text(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

fn check(status: i32, what: &str) -> Result<()> {
    if status == ffi::UAIRT_OK {
        return Ok(());
    }
    let (name, detail) = unsafe { (text(ffi::uairt_status_string(status)), text(ffi::uairt_last_error())) };
    Err(Error { status, message: format!("{what}: {name} ({detail})") })
}

fn cstring(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error { status: 1, message: format!("{s:?} contains a NUL byte") })
}

pub fn version() -> String {
    unsafe { text(ffi::uairt_version_string()) }
}

pub fn load_backend_library(path: &str) -> Result<()> {
    let path = cstring(path)?;
    check(unsafe { ffi::uairt_load_backend_library(path.as_ptr()) }, "load backend library")
}

pub struct Engine {
    raw: *mut ffi::Engine,
}

impl Engine {
    pub fn new(backend: &str, options: &[(&str, &str)]) -> Result<Engine> {
        let backend = cstring(backend)?;
        let owned: Vec<(CString, CString)> =
            options.iter().map(|(k, v)| Ok((cstring(k)?, cstring(v)?))).collect::<Result<_>>()?;
        let raw_options: Vec<ffi::UairtOption> =
            owned.iter().map(|(k, v)| ffi::UairtOption { key: k.as_ptr(), value: v.as_ptr() }).collect();
        let mut raw = ptr::null_mut();
        check(
            unsafe { ffi::uairt_engine_create(backend.as_ptr(), raw_options.as_ptr(), raw_options.len(), &mut raw) },
            "create engine",
        )?;
        Ok(Engine { raw })
    }

    /// The model borrows the engine, so the compiler enforces "destroy models before their engine".
    pub fn load_model(&self, path: &str) -> Result<Model<'_>> {
        let path = cstring(path)?;
        let source = ffi::ModelSource {
            struct_size: std::mem::size_of::<ffi::ModelSource>() as u32,
            path: path.as_ptr(),
            data: ptr::null(),
            size: 0,
            format: ptr::null(),
        };
        let mut raw = ptr::null_mut();
        check(unsafe { ffi::uairt_model_load(self.raw, &source, &mut raw) }, "load model")?;
        Ok(Model { raw, _engine: std::marker::PhantomData })
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { ffi::uairt_engine_destroy(self.raw) }
    }
}

pub struct Model<'e> {
    raw: *mut ffi::Model,
    _engine: std::marker::PhantomData<&'e Engine>,
}

impl Model<'_> {
    pub fn num_inputs(&self) -> usize {
        unsafe { ffi::uairt_model_num_inputs(self.raw) }
    }

    pub fn num_outputs(&self) -> usize {
        unsafe { ffi::uairt_model_num_outputs(self.raw) }
    }

    pub fn input_info(&self, index: usize) -> Result<ffi::Tensor> {
        let mut t = unsafe { std::mem::zeroed::<ffi::Tensor>() };
        unsafe { ffi::uairt_tensor_init(&mut t) };
        check(unsafe { ffi::uairt_model_input_info(self.raw, index, &mut t) }, "input info")?;
        Ok(t)
    }

    pub fn output_info(&self, index: usize) -> Result<ffi::Tensor> {
        let mut t = unsafe { std::mem::zeroed::<ffi::Tensor>() };
        unsafe { ffi::uairt_tensor_init(&mut t) };
        check(unsafe { ffi::uairt_model_output_info(self.raw, index, &mut t) }, "output info")?;
        Ok(t)
    }

    /// Runs a model with float32 inputs and outputs; the slices must match the model's sizes exactly.
    pub fn run_f32(&mut self, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<()> {
        let mut bound_in = Vec::new();
        for (i, data) in inputs.iter().enumerate() {
            let mut t = self.input_info(i)?;
            t.data = data.as_ptr() as *mut c_void;
            t.nbytes = std::mem::size_of_val(*data);
            bound_in.push(t);
        }
        let mut bound_out = Vec::new();
        for (i, data) in outputs.iter_mut().enumerate() {
            let mut t = self.output_info(i)?;
            t.data = data.as_mut_ptr() as *mut c_void;
            t.nbytes = std::mem::size_of_val(&**data);
            bound_out.push(t);
        }
        check(
            unsafe { ffi::uairt_model_run(self.raw, bound_in.as_ptr(), bound_in.len(), bound_out.as_mut_ptr(), bound_out.len()) },
            "run",
        )
    }

    /// Streams generated text to `on_text`; return `true` from it to stop. Needs a backend with sessions (llamacpp).
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        seed: u64,
        mut on_text: impl FnMut(&[u8]) -> bool,
    ) -> Result<(usize, i32)> {
        unsafe extern "C" fn trampoline(text: *const c_char, len: usize, user: *mut c_void) -> c_int {
            let callback = &mut *(user as *mut &mut dyn FnMut(&[u8]) -> bool);
            callback(std::slice::from_raw_parts(text as *const u8, len)) as c_int
        }
        let prompt = cstring(prompt)?;
        let mut options = unsafe { ffi::uairt_generate_options_default() };
        options.max_tokens = max_tokens;
        options.sampling = ffi::Sampling { temperature, top_k, top_p, seed, ..options.sampling };
        let mut result = ffi::GenerateResult {
            struct_size: std::mem::size_of::<ffi::GenerateResult>(),
            num_tokens: 0,
            finish_reason: 0,
        };
        let mut callback: &mut dyn FnMut(&[u8]) -> bool = &mut on_text;
        check(
            unsafe {
                ffi::uairt_llm_generate(
                    self.raw,
                    ptr::null_mut(),
                    prompt.as_ptr(),
                    &options,
                    Some(trampoline),
                    &mut callback as *mut _ as *mut c_void,
                    &mut result,
                )
            },
            "generate",
        )?;
        Ok((result.num_tokens, result.finish_reason))
    }
}

impl Drop for Model<'_> {
    fn drop(&mut self) {
        unsafe { ffi::uairt_model_destroy(self.raw) }
    }
}
