from memory.buffer import NDBuffer



struct Tensor[rank: Int, shape: DimList, type: DType]:
    var data: DTypePointer[type]
    var buffer: NDBuffer[type, rank, shape]

    fn __init__(inout self):
        var size = shape.product[rank]().get()
        self.data = DTypePointer[type].alloc(size)
        memset_zero(self.data, size)
        self.buffer = NDBuffer[type, rank, shape](self.data)

    fn __del__(owned self):
        self.data.free()



fn main():
    var x = Tensor[3, DimList(2, 2, 2), DType.uint8]()
    x.data.simd_store(0, SIMD[DType.uint8, 8](1, 2, 3, 4, 5, 6, 7, 8))
    