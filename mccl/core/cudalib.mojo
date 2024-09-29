from sys.ffi import (
    DLHandle,
    _get_dylib,
    _get_dylib_function,
    _get_global,
    _get_global_or_null,
    external_call,
    _external_call_const,
    _mlirtype_is_eq,
    os_is_macos,
    os_is_linux,
    os_is_windows,
)

struct CudaHandle:
    var handle: UnsafePointer[NoneType]

    fn __init__(inout self: CudaHandle):
        self.handle = UnsafePointer[NoneType]()