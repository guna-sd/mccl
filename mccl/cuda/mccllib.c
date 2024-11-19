#include <cuda.h>                   // Low-level CUDA driver API
#include <cuda_runtime.h>           // Core CUDA runtime functions
#include <stdio.h>                  // Standard input/output
#include <stdbool.h>

static bool isInitialized = false;

__attribute__((constructor))
void initializeCUDA() {
    if (isInitialized) return;
    isInitialized = true;

    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        const char* errName;
        cuGetErrorName(result, &errName);
        printf("CUDA Driver API initialization failed: %s\n", errName);
        isInitialized = false;
    }

    cudaError_t err = cudaFree(0);
    if (err != cudaSuccess) {
        printf("CUDA Runtime API initialization failed: %s\n", cudaGetErrorString(err));
        isInitialized = false;
    }
    if (result != CUDA_SUCCESS || err != cudaSuccess)
    {
        printf("CUDA initialization failed. Some features may be unavailable.\n");
        isInitialized = false;
    }
}

bool isCUDAInitialized() {
    return isInitialized;
}

__attribute__((destructor))
void cleanupCUDA() {
    if (!isInitialized) return;
    isInitialized = false;
    cudaDeviceReset();
}