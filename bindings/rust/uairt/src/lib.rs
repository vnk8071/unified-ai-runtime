// SPDX-License-Identifier: Apache-2.0
//! Safe Rust API for the Unified AI Runtime.
//!
//! ```no_run
//! # fn main() -> uairt::Result<()> {
//! uairt::load_backend_library("libuairt_backend_onnxruntime.so")?;
//! let engine = uairt::Engine::new("onnxruntime", &[("intra_op_threads", "1")])?;
//! let mut model = engine.load_model("model.onnx")?;
//! let input = vec![0f32; 4];
//! let mut output = vec![0f32; 4];
//! model.run(&[uairt::as_bytes(&input)], &mut [uairt::as_bytes_mut(&mut output)])?;
//! # Ok(()) }
//! ```
//!
//! A [`Model`] and a [`Buffer`] borrow their [`Engine`], so the compiler enforces that the engine outlives them.
//! Free a buffer only after the models that ran with it. The raw handles make these types `!Send` and `!Sync`,
//! matching the C API's rule that one model must not run from several threads at once.

use std::ffi::{CStr, CString};
use std::fmt;
use std::marker::PhantomData;
use std::os::raw::c_void;
use uairt_sys as sys;

/// A UAIRT status code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Status {
    InvalidArgument,
    NotFound,
    Unsupported,
    BackendUnavailable,
    IncompatibleModel,
    OutOfMemory,
    Runtime,
    VersionMismatch,
    Io,
    Other(i32),
}

impl From<i32> for Status {
    fn from(code: i32) -> Self {
        match code {
            sys::UAIRT_ERR_INVALID_ARGUMENT => Status::InvalidArgument,
            sys::UAIRT_ERR_NOT_FOUND => Status::NotFound,
            sys::UAIRT_ERR_UNSUPPORTED => Status::Unsupported,
            sys::UAIRT_ERR_BACKEND_UNAVAILABLE => Status::BackendUnavailable,
            sys::UAIRT_ERR_INCOMPATIBLE_MODEL => Status::IncompatibleModel,
            sys::UAIRT_ERR_OUT_OF_MEMORY => Status::OutOfMemory,
            sys::UAIRT_ERR_RUNTIME => Status::Runtime,
            sys::UAIRT_ERR_VERSION_MISMATCH => Status::VersionMismatch,
            sys::UAIRT_ERR_IO => Status::Io,
            other => Status::Other(other),
        }
    }
}

/// A failed UAIRT call.
#[derive(Debug, Clone)]
pub struct Error {
    status: Status,
    code: i32,
    message: String,
}

impl Error {
    fn new(status: Status, message: impl Into<String>) -> Self {
        let code = match status {
            Status::InvalidArgument => sys::UAIRT_ERR_INVALID_ARGUMENT,
            Status::NotFound => sys::UAIRT_ERR_NOT_FOUND,
            Status::Unsupported => sys::UAIRT_ERR_UNSUPPORTED,
            Status::BackendUnavailable => sys::UAIRT_ERR_BACKEND_UNAVAILABLE,
            Status::IncompatibleModel => sys::UAIRT_ERR_INCOMPATIBLE_MODEL,
            Status::OutOfMemory => sys::UAIRT_ERR_OUT_OF_MEMORY,
            Status::Runtime => sys::UAIRT_ERR_RUNTIME,
            Status::VersionMismatch => sys::UAIRT_ERR_VERSION_MISMATCH,
            Status::Io => sys::UAIRT_ERR_IO,
            Status::Other(code) => code,
        };
        Error {
            status,
            code,
            message: message.into(),
        }
    }
    pub fn status(&self) -> Status {
        self.status
    }
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = unsafe { CStr::from_ptr(sys::uairt_status_string(self.code)) }.to_string_lossy();
        if self.message.is_empty() {
            write!(f, "{name}")
        } else {
            write!(f, "{name}: {}", self.message)
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn check(code: i32) -> Result<()> {
    if code == sys::UAIRT_OK {
        return Ok(());
    }
    // uairt_last_error is thread-local; this runs on the thread that made the failing call.
    let message = unsafe {
        let ptr = sys::uairt_last_error();
        if ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Err(Error::new(Status::from(code), message))
}

fn cstring(text: &str, what: &str) -> Result<CString> {
    CString::new(text).map_err(|_| {
        Error::new(
            Status::InvalidArgument,
            format!("{what} contains a NUL byte"),
        )
    })
}

/// A tensor element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DType {
    Float32,
    Float16,
    BFloat16,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    Int64,
    Bool,
    Unknown(i32),
}

impl From<i32> for DType {
    fn from(code: i32) -> Self {
        match code {
            sys::UAIRT_DTYPE_FLOAT32 => DType::Float32,
            sys::UAIRT_DTYPE_FLOAT16 => DType::Float16,
            sys::UAIRT_DTYPE_BFLOAT16 => DType::BFloat16,
            sys::UAIRT_DTYPE_INT8 => DType::Int8,
            sys::UAIRT_DTYPE_UINT8 => DType::UInt8,
            sys::UAIRT_DTYPE_INT16 => DType::Int16,
            sys::UAIRT_DTYPE_UINT16 => DType::UInt16,
            sys::UAIRT_DTYPE_INT32 => DType::Int32,
            sys::UAIRT_DTYPE_INT64 => DType::Int64,
            sys::UAIRT_DTYPE_BOOL => DType::Bool,
            other => DType::Unknown(other),
        }
    }
}

impl DType {
    /// Size of one element in bytes.
    pub fn size(self) -> usize {
        let code = match self {
            DType::Float32 => sys::UAIRT_DTYPE_FLOAT32,
            DType::Float16 => sys::UAIRT_DTYPE_FLOAT16,
            DType::BFloat16 => sys::UAIRT_DTYPE_BFLOAT16,
            DType::Int8 => sys::UAIRT_DTYPE_INT8,
            DType::UInt8 => sys::UAIRT_DTYPE_UINT8,
            DType::Int16 => sys::UAIRT_DTYPE_INT16,
            DType::UInt16 => sys::UAIRT_DTYPE_UINT16,
            DType::Int32 => sys::UAIRT_DTYPE_INT32,
            DType::Int64 => sys::UAIRT_DTYPE_INT64,
            DType::Bool => sys::UAIRT_DTYPE_BOOL,
            DType::Unknown(code) => code,
        };
        unsafe { sys::uairt_dtype_size(code) }
    }
}

/// A model input or output.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<i64>,
    /// Zero when the tensor is not quantized. Real value = (stored - quant_zero_point) * quant_scale.
    pub quant_scale: f32,
    pub quant_zero_point: i32,
}

impl TensorInfo {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product::<i64>() as usize
    }
    pub fn nbytes(&self) -> usize {
        self.num_elements() * self.dtype.size()
    }
    fn from_raw(t: &sys::uairt_tensor) -> TensorInfo {
        let name = if t.name.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(t.name) }
                .to_string_lossy()
                .into_owned()
        };
        TensorInfo {
            name,
            dtype: DType::from(t.dtype),
            shape: t.dims[..t.rank as usize].to_vec(),
            quant_scale: t.quant_scale,
            quant_zero_point: t.quant_zero_point,
        }
    }
}

/// Plain numeric types whose in-memory representation is a valid tensor element.
///
/// # Safety
/// Implementors must have no padding and no invalid bit patterns.
pub unsafe trait Element: Copy {}
unsafe impl Element for f32 {}
unsafe impl Element for i8 {}
unsafe impl Element for u8 {}
unsafe impl Element for i16 {}
unsafe impl Element for u16 {}
unsafe impl Element for i32 {}
unsafe impl Element for i64 {}

/// Views a slice of numbers as bytes without copying.
pub fn as_bytes<T: Element>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

/// Views a mutable slice of numbers as bytes without copying.
pub fn as_bytes_mut<T: Element>(values: &mut [T]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr() as *mut u8,
            std::mem::size_of_val(values),
        )
    }
}

/// The library version.
pub fn version() -> String {
    unsafe { CStr::from_ptr(sys::uairt_version_string()) }
        .to_string_lossy()
        .into_owned()
}

/// Names of the registered backends.
pub fn backends() -> Vec<String> {
    (0..unsafe { sys::uairt_backend_count() })
        .map(|i| {
            unsafe { CStr::from_ptr(sys::uairt_backend_name(i)) }
                .to_string_lossy()
                .into_owned()
        })
        .collect()
}

/// Loads a backend plugin and registers it. Plugins stay loaded for the process lifetime.
pub fn load_backend_library(path: &str) -> Result<()> {
    let path = cstring(path, "path")?;
    check(unsafe { sys::uairt_load_backend_library(path.as_ptr()) })
}

/// Where a buffer lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    Host,
    DmaBuf,
}

/// One backend configuration.
pub struct Engine {
    raw: *mut sys::uairt_engine,
}

impl Engine {
    pub fn new(backend: &str, options: &[(&str, &str)]) -> Result<Engine> {
        let backend = cstring(backend, "backend")?;
        let strings: Vec<(CString, CString)> = options
            .iter()
            .map(|(k, v)| Ok((cstring(k, "option key")?, cstring(v, "option value")?)))
            .collect::<Result<_>>()?;
        let raw_options: Vec<sys::uairt_option> = strings
            .iter()
            .map(|(k, v)| sys::uairt_option {
                key: k.as_ptr(),
                value: v.as_ptr(),
            })
            .collect();
        let mut raw = std::ptr::null_mut();
        check(unsafe {
            sys::uairt_engine_create(
                backend.as_ptr(),
                raw_options.as_ptr(),
                raw_options.len(),
                &mut raw,
            )
        })?;
        Ok(Engine { raw })
    }

    /// Loads a model from a file path.
    pub fn load_model(&self, path: &str) -> Result<Model<'_>> {
        let path = cstring(path, "path")?;
        let source = sys::uairt_model_source {
            struct_size: std::mem::size_of::<sys::uairt_model_source>() as u32,
            path: path.as_ptr(),
            data: std::ptr::null(),
            size: 0,
            format: std::ptr::null(),
        };
        self.load(&source)
    }

    /// Loads a model from memory. `format` is an optional hint.
    pub fn load_model_from_bytes(&self, data: &[u8], format: Option<&str>) -> Result<Model<'_>> {
        let format = format.map(|f| cstring(f, "format")).transpose()?;
        let source = sys::uairt_model_source {
            struct_size: std::mem::size_of::<sys::uairt_model_source>() as u32,
            path: std::ptr::null(),
            data: data.as_ptr() as *const c_void,
            size: data.len(),
            format: format.as_ref().map_or(std::ptr::null(), |f| f.as_ptr()),
        };
        self.load(&source)
    }

    fn load(&self, source: &sys::uairt_model_source) -> Result<Model<'_>> {
        let mut raw = std::ptr::null_mut();
        check(unsafe { sys::uairt_model_load(self.raw, source, &mut raw) })?;
        let mut model = Model {
            raw,
            inputs: Vec::new(),
            outputs: Vec::new(),
            _engine: PhantomData,
        };
        for i in 0..unsafe { sys::uairt_model_num_inputs(raw) } {
            model.inputs.push(model.describe(i, false)?);
        }
        for i in 0..unsafe { sys::uairt_model_num_outputs(raw) } {
            model.outputs.push(model.describe(i, true)?);
        }
        Ok(model)
    }

    /// Allocates a zero-copy buffer; `Domain::DmaBuf` needs backend support.
    pub fn alloc_buffer(&self, nbytes: usize, domain: Domain) -> Result<Buffer<'_>> {
        let domain = match domain {
            Domain::Host => sys::UAIRT_MEM_HOST,
            Domain::DmaBuf => sys::UAIRT_MEM_DMABUF,
        };
        let mut raw = std::ptr::null_mut();
        check(unsafe { sys::uairt_buffer_alloc(self.raw, nbytes, domain, &mut raw) })?;
        Ok(Buffer {
            raw,
            _engine: PhantomData,
        })
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { sys::uairt_engine_destroy(self.raw) }
    }
}

/// A zero-copy buffer owned by a backend.
pub struct Buffer<'e> {
    raw: *mut sys::uairt_buffer,
    _engine: PhantomData<&'e Engine>,
}

impl Buffer<'_> {
    pub fn len(&self) -> usize {
        unsafe { sys::uairt_buffer_size(self.raw) }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The DMABUF file descriptor, or -1 for host buffers.
    pub fn fd(&self) -> i32 {
        unsafe { sys::uairt_buffer_fd(self.raw) }
    }
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(sys::uairt_buffer_data(self.raw) as *const u8, self.len())
        }
    }
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(sys::uairt_buffer_data(self.raw) as *mut u8, self.len())
        }
    }
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        unsafe { sys::uairt_buffer_free(self.raw) }
    }
}

/// A loaded model.
pub struct Model<'e> {
    raw: *mut sys::uairt_model,
    inputs: Vec<TensorInfo>,
    outputs: Vec<TensorInfo>,
    _engine: PhantomData<&'e Engine>,
}

impl Model<'_> {
    pub fn inputs(&self) -> &[TensorInfo] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[TensorInfo] {
        &self.outputs
    }

    fn describe(&self, index: usize, output: bool) -> Result<TensorInfo> {
        let mut tensor: sys::uairt_tensor = unsafe { std::mem::zeroed() };
        check(unsafe {
            if output {
                sys::uairt_model_output_info(self.raw, index, &mut tensor)
            } else {
                sys::uairt_model_input_info(self.raw, index, &mut tensor)
            }
        })?;
        Ok(TensorInfo::from_raw(&tensor))
    }

    /// Runs the model. One byte slice per input and output, sized exactly as [`TensorInfo::nbytes`] reports.
    pub fn run(&mut self, inputs: &[&[u8]], outputs: &mut [&mut [u8]]) -> Result<()> {
        if inputs.len() != self.inputs.len() || outputs.len() != self.outputs.len() {
            return Err(Error::new(
                Status::InvalidArgument,
                format!(
                    "model takes {} inputs and {} outputs, got {} and {}",
                    self.inputs.len(),
                    self.outputs.len(),
                    inputs.len(),
                    outputs.len()
                ),
            ));
        }
        let mut in_tensors = Vec::with_capacity(inputs.len());
        for (i, data) in inputs.iter().enumerate() {
            let mut tensor = self.tensor(i, false, data.len(), "input")?;
            tensor.data = data.as_ptr() as *mut c_void;
            in_tensors.push(tensor);
        }
        let mut out_tensors = Vec::with_capacity(outputs.len());
        for (i, data) in outputs.iter_mut().enumerate() {
            let mut tensor = self.tensor(i, true, data.len(), "output")?;
            tensor.data = data.as_mut_ptr() as *mut c_void;
            out_tensors.push(tensor);
        }
        check(unsafe {
            sys::uairt_model_run(
                self.raw,
                in_tensors.as_ptr(),
                in_tensors.len(),
                out_tensors.as_mut_ptr(),
                out_tensors.len(),
            )
        })
    }

    fn tensor(
        &self,
        index: usize,
        output: bool,
        len: usize,
        kind: &str,
    ) -> Result<sys::uairt_tensor> {
        let info = if output {
            &self.outputs[index]
        } else {
            &self.inputs[index]
        };
        if len != info.nbytes() {
            return Err(Error::new(
                Status::InvalidArgument,
                format!(
                    "{kind} {index} ('{}') needs {} bytes, got {len}",
                    info.name,
                    info.nbytes()
                ),
            ));
        }
        let mut tensor: sys::uairt_tensor = unsafe { std::mem::zeroed() };
        unsafe { sys::uairt_tensor_init(&mut tensor) };
        check(unsafe {
            if output {
                sys::uairt_model_output_info(self.raw, index, &mut tensor)
            } else {
                sys::uairt_model_input_info(self.raw, index, &mut tensor)
            }
        })?;
        tensor.nbytes = len;
        Ok(tensor)
    }

    /// Runs with zero-copy buffers bound to the model's inputs and outputs.
    pub fn run_buffers(&mut self, inputs: &[&Buffer<'_>], outputs: &[&Buffer<'_>]) -> Result<()> {
        if inputs.len() != self.inputs.len() || outputs.len() != self.outputs.len() {
            return Err(Error::new(
                Status::InvalidArgument,
                "one buffer is needed per model input and output",
            ));
        }
        let bind = |buffers: &[&Buffer<'_>], output: bool| -> Result<Vec<sys::uairt_tensor>> {
            buffers
                .iter()
                .enumerate()
                .map(|(i, buffer)| {
                    let mut tensor: sys::uairt_tensor = unsafe { std::mem::zeroed() };
                    check(unsafe {
                        if output {
                            sys::uairt_model_output_info(self.raw, i, &mut tensor)
                        } else {
                            sys::uairt_model_input_info(self.raw, i, &mut tensor)
                        }
                    })?;
                    unsafe { sys::uairt_tensor_use_buffer(&mut tensor, buffer.raw) };
                    Ok(tensor)
                })
                .collect()
        };
        let in_tensors = bind(inputs, false)?;
        let mut out_tensors = bind(outputs, true)?;
        check(unsafe {
            sys::uairt_model_run(
                self.raw,
                in_tensors.as_ptr(),
                in_tensors.len(),
                out_tensors.as_mut_ptr(),
                out_tensors.len(),
            )
        })
    }
}

impl Drop for Model<'_> {
    fn drop(&mut self) {
        unsafe { sys::uairt_model_destroy(self.raw) }
    }
}
