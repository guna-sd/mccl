from sys.info import is_nvidia_gpu, _current_arch
from gpu.id import BlockDim, GridDim, sm_id, lane_id, ThreadIdx
from gpu.intrinsics import warpgroup_reg_alloc
from gpu.sys import Info
from gpu.host.function_v1 import Function
from gpu.host.func_attribute import FuncAttribute, Attribute
from gpu.host.nvtx import _nvtxMarkEx

struct temp:
    var vector : __mlir_type.`memref<16x32xf32>`

fn main():
    print(is_nvidia_gpu())
    print(StringLiteral(_current_arch()))