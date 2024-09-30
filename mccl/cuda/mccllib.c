#ifndef CUDA_MCCL
#define CUDA_MCCL

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <stdio.h>

#endif


int Num_devices() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return device_count;

}



void main()
{
    printf("%d\n", Num_devices());
    return;
}