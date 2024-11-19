from sys.ffi import DLHandle
from os.env import setenv, getenv
from pathlib import Path, cwd
from sys import exit

alias CUDA_NVTX_LIBRARY_PATH = "/usr/local/cuda/lib64/libnvToolsExt.so"
alias CUDA_CUDNN_LIBRARY_PATH = "/usr/lib/x86_64-linux-gnu/libcudnn.so.8"
alias McclPath = "/home/guna/Projects/Open/mccl/mccl/cuda/mccllib.so"

struct CudaLib:
    var lib: DLHandle

    fn __init__(inout self):
        try:
            var clib = cwd()       
            if clib.is_dir():
                #var libPath = str(clib) + "/mccllib.so"            
                _ = setenv("MCCLPATH", McclPath)
            else:
                raise Error("Current directory is not valid.")        
            var cudalib = getenv("MCCLPATH")

            if not cudalib:
                raise Error("MCCLPATH environment variable is not set correctly.")        

            self.lib = DLHandle(cudalib)        

            if not self.lib.check_symbol("initializeCUDA"):
                raise Error("Failed to find 'initializeCUDA' in the library.")        
    
            self.lib.get_function[fn () -> None]("initializeCUDA")()
            if not self.isIntialized():
                raise Error("Cannot initialize 'initializeCUDA' in the library")
        except e:
            self.lib = DLHandle("")
            print(e)
            exit(1)
    
    fn isIntialized(self) -> Bool:
        return self.load_function[fn () -> Bool]("isCUDAInitialized")()

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
