#include <cuda_runtime.h>

__global__ void reduce(const float* x, float* output) {
    __shared__ float buffer[256];

    int tid = threadIdx.x;

    buffer[tid] = x[tid];
    __syncthreads();

    // Complete the loop
    for (int stride = 128; stride >= 1; stride /= 2) {
        if (tid < stride) {
            buffer[tid] += buffer[tid + stride];
        }
        
        if (stride > 1) {
            __syncthreads();
        }
    }

    if (tid == 0) {
        *output = buffer[0];
    }
}