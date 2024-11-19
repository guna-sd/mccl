@value
struct CudaError(Stringable, Writable, Representable, KeyElement):
    alias cudaSuccess                           = CudaError(0)      # No errors
    alias cudaErrorMissingConfiguration         = CudaError(1)      # Missing configuration error
    alias cudaErrorMemoryAllocation             = CudaError(2)      # Memory allocation error
    alias cudaErrorInitializationError          = CudaError(3)      # Initialization error
    alias cudaErrorLaunchFailure                = CudaError(4)      # Launch failure
    alias cudaErrorPriorLaunchFailure           = CudaError(5)      # Prior launch failure
    alias cudaErrorLaunchTimeout                = CudaError(6)      # Launch timeout error
    alias cudaErrorLaunchOutOfResources         = CudaError(7)      # Launch out of resources error
    alias cudaErrorInvalidDeviceFunction        = CudaError(8)      # Invalid device function
    alias cudaErrorInvalidConfiguration         = CudaError(9)      # Invalid configuration
    alias cudaErrorInvalidDevice                = CudaError(10)     # Invalid device
    alias cudaErrorInvalidValue                 = CudaError(11)     # Invalid value
    alias cudaErrorInvalidPitchValue            = CudaError(12)     # Invalid pitch value
    alias cudaErrorInvalidSymbol                = CudaError(13)     # Invalid symbol
    alias cudaErrorMapBufferObjectFailed        = CudaError(14)     # Map buffer object failed
    alias cudaErrorUnmapBufferObjectFailed      = CudaError(15)     # Unmap buffer object failed
    alias cudaErrorInvalidHostPointer           = CudaError(16)     # Invalid host pointer
    alias cudaErrorInvalidDevicePointer         = CudaError(17)     # Invalid device pointer
    alias cudaErrorInvalidTexture               = CudaError(18)     # Invalid texture
    alias cudaErrorInvalidTextureBinding        = CudaError(19)     # Invalid texture binding
    alias cudaErrorInvalidChannelDescriptor     = CudaError(20)     # Invalid channel descriptor
    alias cudaErrorInvalidMemcpyDirection       = CudaError(21)     # Invalid memcpy direction
    alias cudaErrorAddressOfConstant            = CudaError(22)     # Address of constant error
    alias cudaErrorTextureFetchFailed           = CudaError(23)     # Texture fetch failed
    alias cudaErrorTextureNotBound              = CudaError(24)     # Texture not bound error
    alias cudaErrorSynchronizationError         = CudaError(25)     # Synchronization error
    alias cudaErrorInvalidFilterSetting         = CudaError(26)     # Invalid filter setting
    alias cudaErrorInvalidNormSetting           = CudaError(27)     # Invalid norm setting
    alias cudaErrorMixedDeviceExecution         = CudaError(28)     # Mixed device execution
    alias cudaErrorCudartUnloading              = CudaError(29)     # CUDA runtime unloading
    alias cudaErrorUnknown                      = CudaError(30)     # Unknown error condition
    alias cudaErrorNotYetImplemented            = CudaError(31)     # Function not yet implemented
    alias cudaErrorMemoryValueTooLarge          = CudaError(32)     # Memory value too large
    alias cudaErrorInvalidResourceHandle        = CudaError(33)     # Invalid resource handle
    alias cudaErrorNotReady                     = CudaError(34)     # Not ready error
    alias cudaErrorInsufficientDriver           = CudaError(35)     # CUDA runtime is newer than driver
    alias cudaErrorSetOnActiveProcess           = CudaError(36)     # Set on active process error
    alias cudaErrorNoDevice                     = CudaError(38)     # No available CUDA device
    alias cudaErrorECCUncorrectable             = CudaError(39)     # Uncorrectable ECC error detected
    alias cudaErrorStartupFailure               = CudaError(0x7f)   # Startup failure
    alias cudaErrorApiFailureBase               = CudaError(10000)  # API failure base

    var value: Int

    fn __init__(inout self, error: Int):
        self.value = error

    @no_inline
    fn __str__(self) -> String:
        """Gets the name of the CudaError.

        Returns:
            The name of the CudaError.
        """

        return String.write(self)

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        """Gets the representation of the CudaError e.g. `"CudaError.cudaErrorMemoryAllocation"`.

        Returns:
            The representation of the CudaError.
        """
        return "CudaError." + str(self)

    @always_inline("nodebug")
    fn __hash__(self) -> UInt:
        """Computes the hash value for the CudaError.

        Returns:
            An integer hash value based on the CudaError's value.
        """
        return hash(UInt8(self.value))
    
    @no_inline
    fn write_to[W: Writer](self, inout writer: W):
        if self == CudaError.cudaSuccess:
            return writer.write("No errors")
        elif self == CudaError.cudaErrorMissingConfiguration:
            return writer.write("Missing configuration error")
        elif self == CudaError.cudaErrorMemoryAllocation:
            return writer.write("Memory allocation error")
        elif self == CudaError.cudaErrorInitializationError:
            return writer.write("Initialization error")
        elif self == CudaError.cudaErrorLaunchFailure:
            return writer.write("Launch failure")
        elif self == CudaError.cudaErrorPriorLaunchFailure:
            return writer.write("Prior launch failure")
        elif self == CudaError.cudaErrorLaunchTimeout:
            return writer.write("Launch timeout error")
        elif self == CudaError.cudaErrorLaunchOutOfResources:
            return writer.write("Launch out of resources error")
        elif self == CudaError.cudaErrorInvalidDeviceFunction:
            return writer.write("Invalid device function")
        elif self == CudaError.cudaErrorInvalidConfiguration:
            return writer.write("Invalid configuration")
        elif self == CudaError.cudaErrorInvalidDevice:
            return writer.write("Invalid device")
        elif self == CudaError.cudaErrorInvalidValue:
            return writer.write("Invalid value")
        elif self == CudaError.cudaErrorInvalidPitchValue:
            return writer.write("Invalid pitch value")
        elif self == CudaError.cudaErrorInvalidSymbol:
            return writer.write("Invalid symbol")
        elif self == CudaError.cudaErrorMapBufferObjectFailed:
            return writer.write("Map buffer object failed")
        elif self == CudaError.cudaErrorUnmapBufferObjectFailed:
            return writer.write("Unmap buffer object failed")
        elif self == CudaError.cudaErrorInvalidHostPointer:
            return writer.write("Invalid host pointer")
        elif self == CudaError.cudaErrorInvalidDevicePointer:
            return writer.write("Invalid device pointer")
        elif self == CudaError.cudaErrorInvalidTexture:
            return writer.write("Invalid texture")
        elif self == CudaError.cudaErrorInvalidTextureBinding:
            return writer.write("Invalid texture binding")
        elif self == CudaError.cudaErrorInvalidChannelDescriptor:
            return writer.write("Invalid channel descriptor")
        elif self == CudaError.cudaErrorInvalidMemcpyDirection:
            return writer.write("Invalid memcpy direction")
        elif self == CudaError.cudaErrorAddressOfConstant:
            return writer.write("Address of constant error")
        elif self == CudaError.cudaErrorTextureFetchFailed:
            return writer.write("Texture fetch failed")
        elif self == CudaError.cudaErrorTextureNotBound:
            return writer.write("Texture not bound error")
        elif self == CudaError.cudaErrorSynchronizationError:
            return writer.write("Synchronization error")
        elif self == CudaError.cudaErrorInvalidFilterSetting:
            return writer.write("Invalid filter setting")
        elif self == CudaError.cudaErrorInvalidNormSetting:
            return writer.write("Invalid norm setting")
        elif self == CudaError.cudaErrorMixedDeviceExecution:
            return writer.write("Mixed device execution")
        elif self == CudaError.cudaErrorCudartUnloading:
            return writer.write("CUDA runtime unloading")
        elif self == CudaError.cudaErrorUnknown:
            return writer.write("Unknown error condition")
        elif self == CudaError.cudaErrorNotYetImplemented:
            return writer.write("Function not yet implemented")
        elif self == CudaError.cudaErrorMemoryValueTooLarge:
            return writer.write("Memory value too large")
        elif self == CudaError.cudaErrorInvalidResourceHandle:
            return writer.write("Invalid resource handle")
        elif self == CudaError.cudaErrorNotReady:
            return writer.write("Not ready error")
        elif self == CudaError.cudaErrorInsufficientDriver:
            return writer.write("CUDA runtime is newer than driver")
        elif self == CudaError.cudaErrorSetOnActiveProcess:
            return writer.write("Set on active process error")
        elif self == CudaError.cudaErrorNoDevice:
            return writer.write("No available CUDA device")
        elif self == CudaError.cudaErrorECCUncorrectable:
            return writer.write("Uncorrectable ECC error detected")
        elif self == CudaError.cudaErrorStartupFailure:
            return writer.write("Startup failure")
        elif self == CudaError.cudaErrorApiFailureBase:
            return writer.write("API failure base")
        else:
            return writer.write("Unknown error code: {}", self.value)

    @always_inline("nodebug")
    fn __eq__(self, rhs: CudaError) -> Bool:
        """Compares one CudaError to another for equality.

        Args:
            rhs: The CudaError to compare against.

        Returns:
            True if the CudaError are the same and False otherwise.
        """
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: CudaError) -> Bool:
        """Compares one CudaError to another for inequality.

        Args:
            rhs: The CudaError to compare against.

        Returns:
            False if the CudaError are the same and True otherwise.
        """
        return self.value != rhs.value


    @always_inline("nodebug")
    fn __is__(self, rhs: CudaError) -> Bool:
        """Compares one CudaError to another for equality.

        Args:
            rhs: The CudaError to compare against.

        Returns:
            True if the CudaErrors are the same and False otherwise.
        """
        return self == rhs

    @always_inline("nodebug")
    fn __isnot__(self, rhs: CudaError) -> Bool:
        """Compares one CudaError to another for inequality.

        Args:
            rhs: The CudaError to compare against.

        Returns:
            True if the CudaErrors are the same and False otherwise.
        """
        return self != rhs

@value
struct CudaResult:
    alias CUDA_SUCCESS                              = CudaResult(0)

    """
    This indicates that one or more of the parameters passed to the API call
    is not within an acceptable range of values.
    """
    alias CUDA_ERROR_INVALID_VALUE                  = CudaResult(1)

    """
    The API call failed because it was unable to allocate enough memory to
    perform the requested operation.
    """
    alias CUDA_ERROR_OUT_OF_MEMORY                  = CudaResult(2)

    """
    This indicates that the CUDA driver has not been initialized with
    ::cuInit() or that initialization has failed.
    """
    alias CUDA_ERROR_NOT_INITIALIZED                = CudaResult(3)

    """
    This indicates that the CUDA driver is in the process of shutting down.
    """
    alias CUDA_ERROR_DEINITIALIZED                  = CudaResult(4)

    """
    This indicates profiling APIs are called while application is running
    in visual profiler mode. 
    """
    alias CUDA_ERROR_PROFILER_DISABLED               = CudaResult(5)

    """
    This indicates profiling has not been initialized for this context. 
    Call cuProfilerInitialize() to resolve this. 
    """
    alias CUDA_ERROR_PROFILER_NOT_INITIALIZED       = CudaResult(6)

    """
    This indicates profiler has already been started and probably
    cuProfilerStart() is incorrectly called.
    """
    alias CUDA_ERROR_PROFILER_ALREADY_STARTED       = CudaResult(7)

    """
    This indicates profiler has already been stopped and probably
    cuProfilerStop() is incorrectly called.
    """
    alias CUDA_ERROR_PROFILER_ALREADY_STOPPED       = CudaResult(8)  

    """
    This indicates that no CUDA-capable devices were detected by the installed
    CUDA driver.
    """
    alias CUDA_ERROR_NO_DEVICE                      = CudaResult(100)

    """
    This indicates that the device ordinal supplied by the user does not
    correspond to a valid CUDA device.
    """
    alias CUDA_ERROR_INVALID_DEVICE                 = CudaResult(101)

    """
    This indicates that the device kernel image is invalid. This can also
    indicate an invalid CUDA module.
    """
    alias CUDA_ERROR_INVALID_IMAGE                  = CudaResult(200)

    """
    This most frequently indicates that there is no context bound to the
    current thread. This can also be returned if the context passed to an
    API call is not a valid handle (such as a context that has had
    ::cuCtxDestroy() invoked on it). This can also be returned if a user
    mixes different API versions (i.e. 3010 context with 3020 API calls).
    See ::cuCtxGetApiVersion() for more details.
    """
    alias CUDA_ERROR_INVALID_CONTEXT                = CudaResult(201)

    """
    This indicated that the context being supplied as a parameter to the
    API call was already the active context.
    deprecated
    This error return is deprecated as of CUDA 3.2. It is no longer an
    error to attempt to push the active context via ::cuCtxPushCurrent().
    """
    alias CUDA_ERROR_CONTEXT_ALREADY_CURRENT        = CudaResult(202)

    """
    This indicates that a map or register operation has failed.
    """
    alias CUDA_ERROR_MAP_FAILED                     = CudaResult(205)

    """
    This indicates that an unmap or unregister operation has failed.
    """
    alias CUDA_ERROR_UNMAP_FAILED                   = CudaResult(206)

    """
    This indicates that the specified array is currently mapped and thus
    cannot be destroyed.
    """
    alias CUDA_ERROR_ARRAY_IS_MAPPED                = CudaResult(207)

    """
    This indicates that the resource is already mapped.
    """
    alias CUDA_ERROR_ALREADY_MAPPED                 = CudaResult(208)

    """
    This indicates that there is no kernel image available that is suitable
    for the device. This can occur when a user specifies code generation
    options for a particular CUDA source file that do not include the
    corresponding device configuration.
    """
    alias CUDA_ERROR_NO_BINARY_FOR_GPU              = CudaResult(209)

    """
    This indicates that a resource has already been acquired.
    """
    alias CUDA_ERROR_ALREADY_ACQUIRED               = CudaResult(210)

    """
    This indicates that a resource is not mapped.
    """
    alias CUDA_ERROR_NOT_MAPPED                     = CudaResult(211)

    """
    This indicates that a mapped resource is not available for access as an
    array.
    """
    alias CUDA_ERROR_NOT_MAPPED_AS_ARRAY            = CudaResult(212)

    """
    This indicates that a mapped resource is not available for access as a
    pointer.
    """
    alias CUDA_ERROR_NOT_MAPPED_AS_POINTER          = CudaResult(213)

    """
    This indicates that an uncorrectable ECC error was detected during
    execution.
    """
    alias CUDA_ERROR_ECC_UNCORRECTABLE              = CudaResult(214)

    """
    This indicates that the ::CUlimit passed to the API call is not
    supported by the active device.
    """
    alias CUDA_ERROR_UNSUPPORTED_LIMIT              = CudaResult(215)

    """
    This indicates that the ::CUcontext passed to the API call can
    only be bound to a single CPU thread at a time but is already 
    bound to a CPU thread.
    """
    alias CUDA_ERROR_CONTEXT_ALREADY_IN_USE         = CudaResult(216)

    """
    This indicates that the device kernel source is invalid.
    """
    alias CUDA_ERROR_INVALID_SOURCE                 = CudaResult(300)

    """
    This indicates that the file specified was not found.
    """
    alias CUDA_ERROR_FILE_NOT_FOUND                 = CudaResult(301)

    """
    This indicates that a link to a shared object failed to resolve.
    """
    alias CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = CudaResult(302)

    """
    This indicates that initialization of a shared object failed.
    """
    alias CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      = CudaResult(303)

    """
    This indicates that an OS call failed.
    """
    alias CUDA_ERROR_OPERATING_SYSTEM               = CudaResult(304)

    """
    This indicates that a resource handle passed to the API call was not
    valid. Resource handles are opaque types like ::CUstream and ::CUevent.
    """
    alias CUDA_ERROR_INVALID_HANDLE                 = CudaResult(400)

    """
    This indicates that a named symbol was not found. Examples of symbols
    are global/constant variable names, texture names, and surface names.
    """
    alias CUDA_ERROR_NOT_FOUND                      = CudaResult(500)

    """
    This indicates that asynchronous operations issued previously have not
    completed yet. This result is not actually an error, but must be indicated
    differently than ::CUDA_SUCCESS (which indicates completion). Calls that
    may return this value include ::cuEventQuery() and ::cuStreamQuery().
    """
    alias CUDA_ERROR_NOT_READY                      = CudaResult(600)

    """
    An exception occurred on the device while executing a kernel. Common
    causes include dereferencing an invalid device pointer and accessing
    out of bounds shared memory. The context cannot be used, so it must
    be destroyed (and a new one should be created). All existing device
    memory allocations from this context are invalid and must be
    reconstructed if the program is to continue using CUDA.
    """
    alias CUDA_ERROR_LAUNCH_FAILED                  = CudaResult(700)

    """
    This indicates that a launch did not occur because it did not have
    appropriate resources. This error usually indicates that the user has
    attempted to pass too many arguments to the device kernel, or the
    kernel launch specifies too many threads for the kernel's register
    count. Passing arguments of the wrong size (i.e. a 64-bit pointer
    when a 32-bit int is expected) is equivalent to passing too many
    arguments and can also result in this error.
    """
    alias CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = CudaResult(701)

    """
    This indicates that the device kernel took too long to execute. This can
    only occur if timeouts are enabled - see the device attribute
    ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. The
    context cannot be used (and must be destroyed similar to
    ::CUDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from
    this context are invalid and must be reconstructed if the program is to
    continue using CUDA.
    """
    alias CUDA_ERROR_LAUNCH_TIMEOUT                 = CudaResult(702)

    """
    This error indicates a kernel launch that uses an incompatible texturing
    mode.
    """
    alias CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = CudaResult(703)

    """
    This error indicates that a call to ::cuCtxEnablePeerAccess() is
    trying to re-enable peer access to a context which has already
    had peer access to it enabled.
    """
    alias CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    = CudaResult(704)

    """
    This error indicates that ::cuCtxDisablePeerAccess() is 
    trying to disable peer access which has not been enabled yet 
    via ::cuCtxEnablePeerAccess(). 
    """
    alias CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        = CudaResult(705)

    """
    This error indicates that the primary context for the specified device
    has already been initialized.
    """
    alias CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         = CudaResult(708)

    """
    This error indicates that the context current to the calling thread
    has been destroyed using ::cuCtxDestroy, or is a primary context which
    has not yet been initialized.
    """
    alias CUDA_ERROR_CONTEXT_IS_DESTROYED           = CudaResult(709)

    """
    A device-side assert triggered during kernel execution. The context
    cannot be used anymore, and must be destroyed. All existing device 
    memory allocations from this context are invalid and must be 
    reconstructed if the program is to continue using CUDA.
    """
    alias CUDA_ERROR_ASSERT                         = CudaResult(710)

    """
    This error indicates that the hardware resources required to enable
    peer access have been exhausted for one or more of the devices 
    passed to ::cuCtxEnablePeerAccess().
    """
    alias CUDA_ERROR_TOO_MANY_PEERS                 = CudaResult(711)

    """
    This error indicates that the memory range passed to ::cuMemHostRegister() has already been registered.
    """
    alias CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = CudaResult(712)

    """
    This error indicates that the pointer passed to ::cuMemHostUnregister()
    does not correspond to any currently registered memory region.
    """
    alias CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     = CudaResult(713)

    """
    This indicates that an unknown internal error has occurred.
    """
    alias CUDA_ERROR_UNKNOWN                        = CudaResult(999)

    var value: Int