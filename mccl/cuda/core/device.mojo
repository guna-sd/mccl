from sys.ffi import c_size_t

struct cudaExtent:
	var width: c_size_t
	var height: c_size_t
	var depth: c_size_t

@value
struct cudaDeviceProp:
    var name: String
    """String identifying device."""
    
    var totalGlobalMem: c_size_t
    """Global memory available on device in bytes."""
    
    var sharedMemPerBlock: c_size_t
    """Shared memory available per block in bytes."""
    
    var regsPerBlock: Int32
    """32-bit registers available per block."""
    
    var warpSize: Int32
    """Warp size in threads."""
    
    var memPitch: c_size_t
    """Maximum pitch in bytes allowed by memory copies."""
    
    var maxThreadsPerBlock: Int32
    """Maximum number of threads per block."""
    
    var maxThreadsDim: (Int32, Int32, Int32)
    """Maximum size of each dimension of a block."""
    
    var maxGridSize: (Int32, Int32, Int32)
    """Maximum size of each dimension of a grid."""
    
    var clockRate: Int32
    """Clock frequency in kilohertz."""
    
    var totalConstMem: c_size_t
    """Constant memory available on device in bytes."""
    
    var major: Int32
    """Major compute capability."""
    
    var minor: Int32
    """Minor compute capability."""
    
    var textureAlignment: c_size_t
    """Alignment requirement for textures."""
    
    var texturePitchAlignment: c_size_t
    """Pitch alignment requirement for texture references bound to pitched memory."""
    
    var deviceOverlap: Int32
    """Device can concurrently copy memory and execute a kernel (deprecated)."""
    
    var multiProcessorCount: Int32
    """Number of multiprocessors on device."""
    
    var kernelExecTimeoutEnabled: Int32
    """Specifies if there's a runtime limit on kernels."""
    
    var integrated: Int32
    """Device is integrated as opposed to discrete."""
    
    var canMapHostMemory: Int32
    """Device can map host memory."""
    
    var computeMode: Int32
    """Compute mode."""
    
    var maxTexture1D: Int32
    """Maximum 1D texture size."""
    
    var maxTexture1DLinear: Int32
    """Maximum size for 1D textures bound to linear memory."""
    
    var maxTexture2D: (Int32, Int32)
    """Maximum 2D texture dimensions."""
    
    var maxTexture2DLinear: (Int32, Int32, Int32)
    """Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory."""
    
    var maxTexture2DGather: (Int32, Int32)
    """Maximum 2D texture dimensions for gather operations."""
    
    var maxTexture3D: (Int32, Int32, Int32)
    """Maximum 3D texture dimensions."""
    
    var maxTextureCubemap: Int32
    """Maximum cubemap texture dimensions."""
    
    var maxTexture1DLayered: (Int32, Int32)
    """Maximum 1D layered texture dimensions."""
    
    var maxTexture2DLayered: (Int32, Int32, Int32)
    """Maximum 2D layered texture dimensions."""
    
    var maxTextureCubemapLayered: (Int32, Int32)
    """Maximum cubemap layered texture dimensions."""
    
    var maxSurface1D: Int32
    """Maximum 1D surface size."""
    
    var maxSurface2D: (Int32, Int32)
    """Maximum 2D surface dimensions."""
    
    var maxSurface3D: (Int32, Int32, Int32)
    """Maximum 3D surface dimensions."""
    
    var maxSurface1DLayered: (Int32, Int32)
    """Maximum 1D layered surface dimensions."""
    
    var maxSurface2DLayered: (Int32, Int32, Int32)
    """Maximum 2D layered surface dimensions."""
    
    var maxSurfaceCubemap: Int32
    """Maximum cubemap surface dimensions."""
    
    var maxSurfaceCubemapLayered: (Int32, Int32)
    """Maximum cubemap layered surface dimensions."""
    
    var surfaceAlignment: c_size_t
    """Alignment requirements for surfaces."""
    
    var concurrentKernels: Int32
    """Device can execute multiple kernels concurrently."""
    
    var ECCEnabled: Int32
    """Device has ECC support enabled."""
    
    var pciBusID: Int32
    """PCI bus ID of the device."""
    
    var pciDeviceID: Int32
    """PCI device ID of the device."""
    
    var pciDomainID: Int32
    """PCI domain ID of the device."""
    
    var tccDriver: Int32
    """1 if device is a Tesla device using TCC driver, 0 otherwise."""
    
    var asyncEngineCount: Int32
    """Number of asynchronous engines."""
    
    var unifiedAddressing: Int32
    """Device shares a unified address space with the host."""
    
    var memoryClockRate: Int32
    """Peak memory clock frequency in kilohertz."""
    
    var memoryBusWidth: Int32
    """Global memory bus width in bits."""
    
    var l2CacheSize: Int32
    """Size of L2 cache in bytes."""
    
    var maxThreadsPerMultiProcessor: Int32
    """Maximum resident threads per multiprocessor."""

@value
struct CudaDeviceAttr(Stringable, Writable, Representable, KeyElement):
    alias cudaDevAttrMaxThreadsPerBlock             = CudaDeviceAttr(1)  # Maximum number of threads per block
    alias cudaDevAttrMaxBlockDimX                   = CudaDeviceAttr(2)  # Maximum block dimension X
    alias cudaDevAttrMaxBlockDimY                   = CudaDeviceAttr(3)  # Maximum block dimension Y
    alias cudaDevAttrMaxBlockDimZ                   = CudaDeviceAttr(4)  # Maximum block dimension Z
    alias cudaDevAttrMaxGridDimX                    = CudaDeviceAttr(5)  # Maximum grid dimension X
    alias cudaDevAttrMaxGridDimY                    = CudaDeviceAttr(6)  # Maximum grid dimension Y
    alias cudaDevAttrMaxGridDimZ                    = CudaDeviceAttr(7)  # Maximum grid dimension Z
    alias cudaDevAttrMaxSharedMemoryPerBlock        = CudaDeviceAttr(8)  # Maximum shared memory available per block in bytes
    alias cudaDevAttrTotalConstantMemory            = CudaDeviceAttr(9)  # Memory available on device for __constant__ variables in a CUDA C kernel in bytes
    alias cudaDevAttrWarpSize                       = CudaDeviceAttr(10) # Warp size in threads
    alias cudaDevAttrMaxPitch                       = CudaDeviceAttr(11) # Maximum pitch in bytes allowed by memory copies
    alias cudaDevAttrMaxRegistersPerBlock           = CudaDeviceAttr(12) # Maximum number of 32-bit registers available per block
    alias cudaDevAttrClockRate                      = CudaDeviceAttr(13) # Peak clock frequency in kilohertz
    alias cudaDevAttrTextureAlignment               = CudaDeviceAttr(14) # Alignment requirement for textures
    alias cudaDevAttrGpuOverlap                     = CudaDeviceAttr(15) # Device can possibly copy memory and execute a kernel concurrently
    alias cudaDevAttrMultiProcessorCount            = CudaDeviceAttr(16) # Number of multiprocessors on device
    alias cudaDevAttrKernelExecTimeout              = CudaDeviceAttr(17) # Specifies whether there is a run time limit on kernels
    alias cudaDevAttrIntegrated                     = CudaDeviceAttr(18) # Device is integrated with host memory
    alias cudaDevAttrCanMapHostMemory               = CudaDeviceAttr(19) # Device can map host memory into CUDA address space
    alias cudaDevAttrComputeMode                    = CudaDeviceAttr(20) # Compute mode (See ::cudaComputeMode for details)
    alias cudaDevAttrMaxTexture1DWidth              = CudaDeviceAttr(21) # Maximum 1D texture width
    alias cudaDevAttrMaxTexture2DWidth              = CudaDeviceAttr(22) # Maximum 2D texture width
    alias cudaDevAttrMaxTexture2DHeight             = CudaDeviceAttr(23) # Maximum 2D texture height
    alias cudaDevAttrMaxTexture3DWidth              = CudaDeviceAttr(24) # Maximum 3D texture width
    alias cudaDevAttrMaxTexture3DHeight             = CudaDeviceAttr(25) # Maximum 3D texture height
    alias cudaDevAttrMaxTexture3DDepth              = CudaDeviceAttr(26) # Maximum 3D texture depth
    alias cudaDevAttrMaxTexture2DLayeredWidth       = CudaDeviceAttr(27) # Maximum 2D layered texture width
    alias cudaDevAttrMaxTexture2DLayeredHeight      = CudaDeviceAttr(28) # Maximum 2D layered texture height
    alias cudaDevAttrMaxTexture2DLayeredLayers      = CudaDeviceAttr(29) # Maximum layers in a 2D layered texture
    alias cudaDevAttrSurfaceAlignment               = CudaDeviceAttr(30) # Alignment requirement for surfaces
    alias cudaDevAttrConcurrentKernels              = CudaDeviceAttr(31) # Device can possibly execute multiple kernels concurrently
    alias cudaDevAttrEccEnabled                     = CudaDeviceAttr(32) # Device has ECC support enabled
    alias cudaDevAttrPciBusId                       = CudaDeviceAttr(33) # PCI bus ID of the device
    alias cudaDevAttrPciDeviceId                    = CudaDeviceAttr(34) # PCI device ID of the device
    alias cudaDevAttrTccDriver                      = CudaDeviceAttr(35) # Device is using TCC driver model
    alias cudaDevAttrMemoryClockRate                = CudaDeviceAttr(36) # Peak memory clock frequency in kilohertz
    alias cudaDevAttrGlobalMemoryBusWidth           = CudaDeviceAttr(37) # Global memory bus width in bits
    alias cudaDevAttrL2CacheSize                    = CudaDeviceAttr(38) # Size of L2 cache in bytes
    alias cudaDevAttrMaxThreadsPerMultiProcessor    = CudaDeviceAttr(39) # Maximum resident threads per multiprocessor
    alias cudaDevAttrAsyncEngineCount               = CudaDeviceAttr(40) # Number of asynchronous engines
    alias cudaDevAttrUnifiedAddressing              = CudaDeviceAttr(41) # Device shares a unified address space with the host    
    alias cudaDevAttrMaxTexture1DLayeredWidth       = CudaDeviceAttr(42) # Maximum 1D layered texture width
    alias cudaDevAttrMaxTexture1DLayeredLayers      = CudaDeviceAttr(43) # Maximum layers in a 1D layered texture
    alias cudaDevAttrMaxTexture2DGatherWidth        = CudaDeviceAttr(45) # Maximum 2D texture width if cudaArrayTextureGather is set
    alias cudaDevAttrMaxTexture2DGatherHeight       = CudaDeviceAttr(46) # Maximum 2D texture height if cudaArrayTextureGather is set
    alias cudaDevAttrMaxTexture3DWidthAlt           = CudaDeviceAttr(47) # Alternate maximum 3D texture width
    alias cudaDevAttrMaxTexture3DHeightAlt          = CudaDeviceAttr(48) # Alternate maximum 3D texture height
    alias cudaDevAttrMaxTexture3DDepthAlt           = CudaDeviceAttr(49) # Alternate maximum 3D texture depth
    alias cudaDevAttrPciDomainId                    = CudaDeviceAttr(50) # PCI domain ID of the device
    alias cudaDevAttrTexturePitchAlignment          = CudaDeviceAttr(51) # Pitch alignment requirement for textures
    alias cudaDevAttrMaxTextureCubemapWidth         = CudaDeviceAttr(52) # Maximum cubemap texture width/height
    alias cudaDevAttrMaxTextureCubemapLayeredWidth  = CudaDeviceAttr(53) # Maximum cubemap layered texture width/height
    alias cudaDevAttrMaxTextureCubemapLayeredLayers = CudaDeviceAttr(54) # Maximum layers in a cubemap layered texture
    alias cudaDevAttrMaxSurface1DWidth              = CudaDeviceAttr(55) # Maximum 1D surface width
    alias cudaDevAttrMaxSurface2DWidth              = CudaDeviceAttr(56) # Maximum 2D surface width
    alias cudaDevAttrMaxSurface2DHeight             = CudaDeviceAttr(57) # Maximum 2D surface height
    alias cudaDevAttrMaxSurface3DWidth              = CudaDeviceAttr(58) # Maximum 3D surface width
    alias cudaDevAttrMaxSurface3DHeight             = CudaDeviceAttr(59) # Maximum 3D surface height
    alias cudaDevAttrMaxSurface3DDepth              = CudaDeviceAttr(60) # Maximum 3D surface depth
    alias cudaDevAttrMaxSurface1DLayeredWidth       = CudaDeviceAttr(61) # Maximum 1D layered surface width
    alias cudaDevAttrMaxSurface1DLayeredLayers      = CudaDeviceAttr(62) # Maximum layers in a 1D layered surface
    alias cudaDevAttrMaxSurface2DLayeredWidth       = CudaDeviceAttr(63) # Maximum 2D layered surface width
    alias cudaDevAttrMaxSurface2DLayeredHeight      = CudaDeviceAttr(64) # Maximum 2D layered surface height
    alias cudaDevAttrMaxSurface2DLayeredLayers      = CudaDeviceAttr(65) # Maximum layers in a 2D layered surface
    alias cudaDevAttrMaxSurfaceCubemapWidth         = CudaDeviceAttr(66) # Maximum cubemap surface width
    alias cudaDevAttrMaxSurfaceCubemapLayeredWidth  = CudaDeviceAttr(67) # Maximum cubemap layered surface width
    alias cudaDevAttrMaxSurfaceCubemapLayeredLayers = CudaDeviceAttr(68) # Maximum layers in a cubemap layered surface
    alias cudaDevAttrMaxTexture1DLinearWidth        = CudaDeviceAttr(69) # Maximum 1D linear texture width
    alias cudaDevAttrMaxTexture2DLinearWidth        = CudaDeviceAttr(70) # Maximum 2D linear texture width
    alias cudaDevAttrMaxTexture2DLinearHeight       = CudaDeviceAttr(71) # Maximum 2D linear texture height
    alias cudaDevAttrMaxTexture2DLinearPitch        = CudaDeviceAttr(72) # Maximum 2D linear texture pitch in bytes
    alias cudaDevAttrMaxTexture2DMipmappedWidth     = CudaDeviceAttr(73) # Maximum mipmapped 2D texture width
    alias cudaDevAttrMaxTexture2DMipmappedHeight    = CudaDeviceAttr(74) # Maximum mipmapped 2D texture height
    alias cudaDevAttrComputeCapabilityMajor         = CudaDeviceAttr(75) # Major compute capability version number 
    alias cudaDevAttrComputeCapabilityMinor         = CudaDeviceAttr(76) # Minor compute capability version number
    alias cudaDevAttrMaxTexture1DMipmappedWidth     = CudaDeviceAttr(77) # Maximum mipmapped 1D texture width

    var value: Int

    fn __init__(inout self, value: Int):
        self.value = value
    

    fn __str__(self) -> String:
        """Gets the name of the CudaDeviceAttr.

        Returns:
            The name of the CudaDeviceAttr.
        """
        return String.write(self)

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        """Gets the representation of the CudaDeviceAttr.

        Returns:
            The representation of the CudaDeviceAttr.
        """
        return "CudaDeviceAttr." + str(self)

    @always_inline("nodebug")
    fn __hash__(self) -> UInt:
        """Computes the hash value for the CudaDeviceAttr.

        Returns:
            An integer hash value based on the CudaDeviceAttr's value.
        """
        return hash(UInt8(self.value))
    
    @no_inline
    fn write_to[W: Writer](self, inout writer: W):
        if self == CudaDeviceAttr.cudaDevAttrMaxThreadsPerBlock:
            writer.write("Maximum number of threads per block")
        elif self == CudaDeviceAttr.cudaDevAttrMaxBlockDimX:
            writer.write("Maximum block dimension X")
        elif self == CudaDeviceAttr.cudaDevAttrMaxBlockDimY:
            writer.write("Maximum block dimension Y")
        elif self == CudaDeviceAttr.cudaDevAttrMaxBlockDimZ:
            writer.write("Maximum block dimension Z")
        elif self == CudaDeviceAttr.cudaDevAttrMaxGridDimX:
            writer.write("Maximum grid dimension X")
        elif self == CudaDeviceAttr.cudaDevAttrMaxGridDimY:
            writer.write("Maximum grid dimension Y")
        elif self == CudaDeviceAttr.cudaDevAttrMaxGridDimZ:
            writer.write("Maximum grid dimension Z")
        elif self == CudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlock:
            writer.write("Maximum shared memory available per block in bytes")
        elif self == CudaDeviceAttr.cudaDevAttrTotalConstantMemory:
            writer.write("Memory available on device for __constant__ variables in bytes")
        elif self == CudaDeviceAttr.cudaDevAttrWarpSize:
            writer.write("Warp size in threads")
        elif self == CudaDeviceAttr.cudaDevAttrMaxPitch:
            writer.write("Maximum pitch in bytes allowed by memory copies")
        elif self == CudaDeviceAttr.cudaDevAttrMaxRegistersPerBlock:
            writer.write("Maximum number of 32-bit registers available per block")
        elif self == CudaDeviceAttr.cudaDevAttrClockRate:
            writer.write("Peak clock frequency in kilohertz")
        elif self == CudaDeviceAttr.cudaDevAttrTextureAlignment:
            writer.write("Alignment requirement for textures")
        elif self == CudaDeviceAttr.cudaDevAttrGpuOverlap:
            writer.write("Device can possibly copy memory and execute a kernel concurrently")
        elif self == CudaDeviceAttr.cudaDevAttrMultiProcessorCount:
            writer.write("Number of multiprocessors on device")
        elif self == CudaDeviceAttr.cudaDevAttrKernelExecTimeout:
            writer.write("Specifies whether there is a runtime limit on kernels")
        elif self == CudaDeviceAttr.cudaDevAttrIntegrated:
            writer.write("Device is integrated with host memory")
        elif self == CudaDeviceAttr.cudaDevAttrCanMapHostMemory:
            writer.write("Device can map host memory into CUDA address space")
        elif self == CudaDeviceAttr.cudaDevAttrComputeMode:
            writer.write("Compute mode (See ::cudaComputeMode for details)")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture1DWidth:
            writer.write("Maximum 1D texture width")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture2DWidth:
            writer.write("Maximum 2D texture width")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture2DHeight:
            writer.write("Maximum 2D texture height")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture3DWidth:
            writer.write("Maximum 3D texture width")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture3DHeight:
            writer.write("Maximum 3D texture height")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture3DDepth:
            writer.write("Maximum 3D texture depth")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture2DLayeredWidth:
            writer.write("Maximum 2D layered texture width")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture2DLayeredHeight:
            writer.write("Maximum 2D layered texture height")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture2DLayeredLayers:
            writer.write("Maximum layers in a 2D layered texture")
        elif self == CudaDeviceAttr.cudaDevAttrSurfaceAlignment:
            writer.write("Alignment requirement for surfaces")
        elif self == CudaDeviceAttr.cudaDevAttrConcurrentKernels:
            writer.write("Device can possibly execute multiple kernels concurrently")
        elif self == CudaDeviceAttr.cudaDevAttrEccEnabled:
            writer.write("Device has ECC support enabled")
        elif self == CudaDeviceAttr.cudaDevAttrPciBusId:
            writer.write("PCI bus ID of the device")
        elif self == CudaDeviceAttr.cudaDevAttrPciDeviceId:
            writer.write("PCI device ID of the device")
        elif self == CudaDeviceAttr.cudaDevAttrTccDriver:
            writer.write("Device is using TCC driver model")
        elif self == CudaDeviceAttr.cudaDevAttrMemoryClockRate:
            writer.write("Peak memory clock frequency in kilohertz")
        elif self == CudaDeviceAttr.cudaDevAttrGlobalMemoryBusWidth:
            writer.write("Global memory bus width in bits")
        elif self == CudaDeviceAttr.cudaDevAttrL2CacheSize:
            writer.write("Size of L2 cache in bytes")
        elif self == CudaDeviceAttr.cudaDevAttrMaxThreadsPerMultiProcessor:
            writer.write("Maximum resident threads per multiprocessor")
        elif self == CudaDeviceAttr.cudaDevAttrAsyncEngineCount:
            writer.write("Number of asynchronous engines")
        elif self == CudaDeviceAttr.cudaDevAttrUnifiedAddressing:
            writer.write("Device shares a unified address space with the host")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture1DLayeredWidth:
            writer.write("Maximum 1D layered texture width")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture1DLayeredLayers:
            writer.write("Maximum layers in a 1D layered texture")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture2DGatherWidth:
            writer.write("Maximum 2D texture width if cudaArrayTextureGather is set")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture2DGatherHeight:
            writer.write("Maximum 2D texture height if cudaArrayTextureGather is set")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture3DWidthAlt:
            writer.write("Alternate maximum 3D texture width")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture3DHeightAlt:
            writer.write("Alternate maximum 3D texture height")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture3DDepthAlt:
            writer.write("Alternate maximum 3D texture depth")
        elif self == CudaDeviceAttr.cudaDevAttrPciDomainId:
            writer.write("PCI domain ID of the device")
        elif self == CudaDeviceAttr.cudaDevAttrTexturePitchAlignment:
            writer.write("Pitch alignment requirement for textures")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTextureCubemapWidth:
            writer.write("Maximum cubemap texture width/height")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTextureCubemapLayeredWidth:
            writer.write("Maximum cubemap layered texture width/height")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTextureCubemapLayeredLayers:
            writer.write("Maximum layers in a cubemap layered texture")
        elif self == CudaDeviceAttr.cudaDevAttrMaxSurface1DWidth:
            writer.write("Maximum 1D surface width")
        elif self == CudaDeviceAttr.cudaDevAttrMaxSurface2DWidth:
            writer.write("Maximum 2D surface width")
        elif self == CudaDeviceAttr.cudaDevAttrMaxSurface2DHeight:
            writer.write("Maximum 2D surface height")
        elif self == CudaDeviceAttr.cudaDevAttrMaxSurface3DWidth:
            writer.write("Maximum 3D surface width")
        elif self == CudaDeviceAttr.cudaDevAttrMaxSurface3DHeight:
            writer.write("Maximum 3D surface height")
        elif self == CudaDeviceAttr.cudaDevAttrMaxSurface3DDepth:
            writer.write("Maximum 3D surface depth")
        elif self == CudaDeviceAttr.cudaDevAttrMaxSurface1DLayeredWidth:
            writer.write("Maximum 1D layered surface width")
        elif self == CudaDeviceAttr.cudaDevAttrMaxSurface1DLayeredLayers:
            writer.write("Maximum layers in a 1D layered surface")
        elif self == CudaDeviceAttr.cudaDevAttrMaxSurface2DLayeredWidth:
            writer.write("Maximum 2D layered surface width")
        elif self == CudaDeviceAttr.cudaDevAttrMaxSurface2DLayeredHeight:
            writer.write("Maximum 2D layered surface height")
        elif self == CudaDeviceAttr.cudaDevAttrMaxSurface2DLayeredLayers:
            writer.write("Maximum layers in a 2D layered surface")
        elif self == CudaDeviceAttr.cudaDevAttrMaxSurfaceCubemapWidth:
            writer.write("Maximum cubemap surface width")
        elif self == CudaDeviceAttr.cudaDevAttrMaxSurfaceCubemapLayeredWidth:
            writer.write("Maximum cubemap layered surface width")
        elif self == CudaDeviceAttr.cudaDevAttrMaxSurfaceCubemapLayeredLayers:
            writer.write("Maximum layers in a cubemap layered surface")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture1DLinearWidth:
            writer.write("Maximum 1D linear texture width")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture2DLinearWidth:
            writer.write("Maximum 2D linear texture width")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture2DLinearHeight:
            writer.write("Maximum 2D linear texture height")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture2DLinearPitch:
            writer.write("Maximum 2D linear texture pitch in bytes")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture2DMipmappedWidth:
            writer.write("Maximum mipmapped 2D texture width")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture2DMipmappedHeight:
            writer.write("Maximum mipmapped 2D texture height")
        elif self == CudaDeviceAttr.cudaDevAttrComputeCapabilityMajor:
            writer.write("Major compute capability version number")
        elif self == CudaDeviceAttr.cudaDevAttrComputeCapabilityMinor:
            writer.write("Minor compute capability version number")
        elif self == CudaDeviceAttr.cudaDevAttrMaxTexture1DMipmappedWidth:
            writer.write("Maximum mipmapped 1D texture width")
        else:
            writer.write("Unknown CudaDeviceAttr")

    @always_inline("nodebug")
    fn __eq__(self, rhs: CudaDeviceAttr) -> Bool:
        """Compares one CudaDeviceAttr to another for equality.

        Args:
            rhs: The CudaDeviceAttr to compare against.

        Returns:
            True if the CudaDeviceAttr are the same and False otherwise.
        """
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: CudaDeviceAttr) -> Bool:
        """Compares one CudaDeviceAttr to another for inequality.

        Args:
            rhs: The CudaDeviceAttr to compare against.

        Returns:
            False if the CudaDeviceAttr are the same and True otherwise.
        """
        return self.value != rhs.value

    @always_inline("nodebug")
    fn __is__(self, rhs: CudaDeviceAttr) -> Bool:
        """Compares one CudaDeviceAttr to another for equality.

        Args:
            rhs: The CudaDeviceAttr to compare against.

        Returns:
            True if the CudaDeviceAttr are the same and False otherwise.
        """
        return self == rhs

    @always_inline("nodebug")
    fn __isnot__(self, rhs: CudaDeviceAttr) -> Bool:
        """Compares one CudaDeviceAttr to another for inequality.

        Args:
            rhs: The CudaDeviceAttr to compare against.

        Returns:
            True if the CudaDeviceAttr are the same and False otherwise.
        """
        return self != rhs