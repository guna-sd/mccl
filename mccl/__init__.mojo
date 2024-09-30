from sys.ffi import (
    c_char,
    c_int,
    OpaquePointer,
    c_size_t,
    c_ssize_t,
    c_char,
    c_int,
    c_long,
    c_long_long,
)
from utils import StringRef
from memory import UnsafePointer
from .cuda import CudaLib

alias mccllib = CudaLib()