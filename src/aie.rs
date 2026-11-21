use std::{
    ffi::{c_int, c_void},
    slice,
};

#[repr(C)]
struct RawProgram {
    _private: [u8; 0],
}
#[repr(C)]
struct RawInstructions {
    _private: [u8; 0],
}
#[repr(C)]
struct RawBuffer {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn npu_program_load(data: *const u8, len: usize) -> *mut RawProgram;
    fn npu_program_free(p: *mut RawProgram);
    fn npu_instructions_load(
        p: *const RawProgram,
        data: *const u8,
        len: usize,
    ) -> *mut RawInstructions;
    fn npu_instructions_free(i: *mut RawInstructions);
    fn npu_buffer_alloc(p: *const RawProgram, bytes: usize) -> *mut RawBuffer;
    fn npu_buffer_sub(parent: *const RawBuffer, offset: usize, bytes: usize) -> *mut RawBuffer;
    fn npu_buffer_ptr(b: *mut RawBuffer) -> *mut c_void;
    fn npu_buffer_size(b: *const RawBuffer) -> usize;
    fn npu_buffer_free(b: *mut RawBuffer);
    fn npu_buffer_sync(b: *mut RawBuffer, to_device: c_int, offset: usize, bytes: usize);
    fn npu_program_bind(
        p: *mut RawProgram,
        insts: *const RawInstructions,
        buffers: *const *mut RawBuffer,
        n: usize,
    );
    fn npu_program_set_buffer(p: *mut RawProgram, slot: usize, buffer: *mut RawBuffer);
    fn npu_program_start(p: *mut RawProgram);
    fn npu_program_wait(p: *mut RawProgram);
}

pub struct AIEProgram {
    raw: *mut RawProgram,
    in_flight: bool,
}

impl AIEProgram {
    pub fn load(xclbin: &[u8]) -> Self {
        Self {
            raw: unsafe { npu_program_load(xclbin.as_ptr(), xclbin.len()) },
            in_flight: false,
        }
    }

    pub fn load_insts(&self, insts_bytes: &[u8]) -> AIEInstructions {
        assert_eq!(insts_bytes.len() % 4, 0);
        AIEInstructions {
            raw: unsafe {
                npu_instructions_load(self.raw, insts_bytes.as_ptr(), insts_bytes.len())
            },
        }
    }

    pub fn alloc(&self, bytes: usize) -> AIEBuffer {
        let raw = unsafe { npu_buffer_alloc(self.raw, bytes) };
        AIEBuffer {
            raw,
            ptr: unsafe { npu_buffer_ptr(raw) },
            capacity: unsafe { npu_buffer_size(raw) },
        }
    }

    pub fn bind(&mut self, insts: &AIEInstructions, buffers: &[&AIEBuffer]) {
        assert!(!self.in_flight && buffers.len() <= 5);
        let mut ptrs = [std::ptr::null_mut(); 5];
        for (i, b) in buffers.iter().enumerate() {
            ptrs[i] = b.raw;
        }
        unsafe { npu_program_bind(self.raw, insts.raw, ptrs.as_ptr(), buffers.len()) };
    }

    pub fn set_buffer(&mut self, slot: usize, buffer: &AIEBuffer) {
        assert!(!self.in_flight && slot < 5);
        unsafe { npu_program_set_buffer(self.raw, slot, buffer.raw) };
    }

    pub fn start(&mut self) {
        assert!(!self.in_flight);
        unsafe { npu_program_start(self.raw) };
        self.in_flight = true;
    }

    pub fn wait(&mut self) {
        if self.in_flight {
            unsafe { npu_program_wait(self.raw) };
            self.in_flight = false;
        }
    }

    pub fn run(&mut self) {
        self.start();
        self.wait();
    }
}

impl Drop for AIEProgram {
    fn drop(&mut self) {
        self.wait();
        unsafe { npu_program_free(self.raw) };
    }
}

pub struct AIEInstructions {
    raw: *mut RawInstructions,
}

impl Drop for AIEInstructions {
    fn drop(&mut self) {
        unsafe { npu_instructions_free(self.raw) };
    }
}

pub struct AIEBuffer {
    raw: *mut RawBuffer,
    ptr: *mut c_void,
    capacity: usize,
}

impl AIEBuffer {
    pub fn sub(&self, offset: usize, bytes: usize) -> Self {
        assert!(bytes > 0 && offset + bytes <= self.capacity);
        let raw = unsafe { npu_buffer_sub(self.raw, offset, bytes) };
        Self {
            raw,
            ptr: unsafe { npu_buffer_ptr(raw) },
            capacity: bytes,
        }
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
    pub fn byte_len(&self) -> usize {
        self.capacity
    }

    pub fn as_slice<T: Copy>(&self, len: usize) -> &[T] {
        assert!(len * size_of::<T>() <= self.capacity);
        unsafe { slice::from_raw_parts(self.ptr as *const T, len) }
    }

    pub fn as_mut_slice<T: Copy>(&mut self, len: usize) -> &mut [T] {
        assert!(len * size_of::<T>() <= self.capacity);
        unsafe { slice::from_raw_parts_mut(self.ptr as *mut T, len) }
    }

    pub fn sync_to_device(&self) {
        unsafe { npu_buffer_sync(self.raw, 1, 0, 0) };
    }

    pub fn sync_from_device(&self) {
        unsafe { npu_buffer_sync(self.raw, 0, 0, 0) };
    }

    pub fn sync_range_to_device(&self, offset: usize, bytes: usize) {
        unsafe { npu_buffer_sync(self.raw, 1, offset, bytes) };
    }

    pub fn sync_range_from_device(&self, offset: usize, bytes: usize) {
        unsafe { npu_buffer_sync(self.raw, 0, offset, bytes) };
    }
}

impl Drop for AIEBuffer {
    fn drop(&mut self) {
        unsafe { npu_buffer_free(self.raw) };
    }
}
