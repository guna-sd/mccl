@value
struct CudaError(Stringable, Formattable, Representable, KeyElement):
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

        return String.format_sequence(self)

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
    fn format_to(self, inout writer: Formatter):
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