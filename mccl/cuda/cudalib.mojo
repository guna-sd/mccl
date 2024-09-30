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
from os.env import setenv, getenv
from pathlib import Path
from memory import stack_allocation

fn cwd() -> Path:
    var buf = stack_allocation[1024, Int8]()

    var res = external_call["getcwd", UnsafePointer[c_char]](
        buf, Int(1024)
    )
    if res == UnsafePointer[c_char]():
        return ""
    return String(StringRef(buf))

struct CudaLib:
    var lib: DLHandle

    fn __init__(inout self):
        var clib = cwd()
        if clib.is_dir():
            _ = setenv("MCCLPATH", str(clib))
        var cudalib = getenv("MCCLPATH")
        self.lib = DLHandle(cudalib)

    fn load_function[type: AnyTrivialRegType](self, name: String) -> type:
        """Loads a function from the dynamic library by name.

        Args:
            name: The name of the function to load.

        Returns:
            A pointer to the loaded function of type `type`.
        """
        return self.lib.get_function[type](name)

    fn get_lib_handle(self) -> DLHandle:
        """Returns the DLHandle of the loaded CUDA library.

        Returns:
            The DLHandle of the library.
        """
        return self.lib
