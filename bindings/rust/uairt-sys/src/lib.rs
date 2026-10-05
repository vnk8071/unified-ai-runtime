// SPDX-License-Identifier: Apache-2.0
//! Raw declarations for `include/uairt/uairt.h`. Layouts must match the C header.
#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_void};

pub const UAIRT_MAX_RANK: usize = 8;

pub type uairt_status = i32;
pub const UAIRT_OK: uairt_status = 0;
pub const UAIRT_ERR_INVALID_ARGUMENT: uairt_status = 1;
pub const UAIRT_ERR_NOT_FOUND: uairt_status = 2;
pub const UAIRT_ERR_UNSUPPORTED: uairt_status = 3;
pub const UAIRT_ERR_BACKEND_UNAVAILABLE: uairt_status = 4;
pub const UAIRT_ERR_INCOMPATIBLE_MODEL: uairt_status = 5;
pub const UAIRT_ERR_OUT_OF_MEMORY: uairt_status = 6;
pub const UAIRT_ERR_RUNTIME: uairt_status = 7;
pub const UAIRT_ERR_VERSION_MISMATCH: uairt_status = 8;
pub const UAIRT_ERR_IO: uairt_status = 9;

pub type uairt_dtype = i32;
pub const UAIRT_DTYPE_UNKNOWN: uairt_dtype = 0;
pub const UAIRT_DTYPE_FLOAT32: uairt_dtype = 1;
pub const UAIRT_DTYPE_FLOAT16: uairt_dtype = 2;
pub const UAIRT_DTYPE_BFLOAT16: uairt_dtype = 3;
pub const UAIRT_DTYPE_INT8: uairt_dtype = 4;
pub const UAIRT_DTYPE_UINT8: uairt_dtype = 5;
pub const UAIRT_DTYPE_INT16: uairt_dtype = 6;
pub const UAIRT_DTYPE_INT32: uairt_dtype = 7;
pub const UAIRT_DTYPE_INT64: uairt_dtype = 8;
pub const UAIRT_DTYPE_BOOL: uairt_dtype = 9;
pub const UAIRT_DTYPE_UINT16: uairt_dtype = 10;

pub type uairt_memory_domain = u32;
pub const UAIRT_MEM_HOST: uairt_memory_domain = 1 << 0;
pub const UAIRT_MEM_DMABUF: uairt_memory_domain = 1 << 1;
pub const UAIRT_MEM_PINNED: uairt_memory_domain = 1 << 2;

#[repr(C)]
pub struct uairt_engine {
    _private: [u8; 0],
}
#[repr(C)]
pub struct uairt_model {
    _private: [u8; 0],
}
#[repr(C)]
pub struct uairt_buffer {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct uairt_option {
    pub key: *const c_char,
    pub value: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct uairt_model_source {
    pub struct_size: u32,
    pub path: *const c_char,
    pub data: *const c_void,
    pub size: usize,
    pub format: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct uairt_tensor {
    pub struct_size: u32,
    pub dtype: uairt_dtype,
    pub rank: u32,
    pub dims: [i64; UAIRT_MAX_RANK],
    pub quant_scale: f32,
    pub quant_zero_point: i32,
    pub domain: uairt_memory_domain,
    pub data: *mut c_void,
    pub dmabuf_fd: i32,
    pub nbytes: usize,
    pub name: *const c_char,
}

extern "C" {
    pub fn uairt_version_string() -> *const c_char;
    pub fn uairt_status_string(status: uairt_status) -> *const c_char;
    pub fn uairt_last_error() -> *const c_char;
    pub fn uairt_dtype_size(dtype: uairt_dtype) -> usize;
    pub fn uairt_tensor_init(tensor: *mut uairt_tensor);
    pub fn uairt_tensor_num_elements(tensor: *const uairt_tensor) -> i64;
    pub fn uairt_backend_count() -> usize;
    pub fn uairt_backend_name(index: usize) -> *const c_char;
    pub fn uairt_load_backend_library(path: *const c_char) -> uairt_status;
    pub fn uairt_engine_create(
        backend: *const c_char,
        options: *const uairt_option,
        num_options: usize,
        out_engine: *mut *mut uairt_engine,
    ) -> uairt_status;
    pub fn uairt_engine_destroy(engine: *mut uairt_engine);
    pub fn uairt_model_load(
        engine: *mut uairt_engine,
        source: *const uairt_model_source,
        out_model: *mut *mut uairt_model,
    ) -> uairt_status;
    pub fn uairt_model_destroy(model: *mut uairt_model);
    pub fn uairt_model_num_inputs(model: *const uairt_model) -> usize;
    pub fn uairt_model_num_outputs(model: *const uairt_model) -> usize;
    pub fn uairt_model_input_info(
        model: *const uairt_model,
        index: usize,
        out: *mut uairt_tensor,
    ) -> uairt_status;
    pub fn uairt_model_output_info(
        model: *const uairt_model,
        index: usize,
        out: *mut uairt_tensor,
    ) -> uairt_status;
    pub fn uairt_buffer_alloc(
        engine: *mut uairt_engine,
        nbytes: usize,
        domain: uairt_memory_domain,
        out_buffer: *mut *mut uairt_buffer,
    ) -> uairt_status;
    pub fn uairt_buffer_free(buffer: *mut uairt_buffer);
    pub fn uairt_buffer_data(buffer: *const uairt_buffer) -> *mut c_void;
    pub fn uairt_buffer_fd(buffer: *const uairt_buffer) -> i32;
    pub fn uairt_buffer_size(buffer: *const uairt_buffer) -> usize;
    pub fn uairt_tensor_use_buffer(tensor: *mut uairt_tensor, buffer: *const uairt_buffer);
    pub fn uairt_model_run(
        model: *mut uairt_model,
        inputs: *const uairt_tensor,
        num_inputs: usize,
        outputs: *mut uairt_tensor,
        num_outputs: usize,
    ) -> uairt_status;
}
