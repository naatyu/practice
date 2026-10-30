#include <cuda_runtime.h>

__device__ float warp_butterfly_transform(float value, int lane) {
    // Use xor warp shuffle to do butterfly operations across lanes
    for (int mask=1; mask < 32; mask *= 2) {
        float other = __shfl_xor_sync(
            0xffffffff,
            value,
            mask
        );

        // Use the mask to know if lane should add or sub
        if ((lane & mask) == 0) {
            value = value + other;
        } else {
            value = other - value;
        }
    }

    return value;
}

__global__ void walsh_hadamard_transform(
    const float* input,
    float* output
) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    
    // Do the first 5 (log2(32)) stages using the warp shuffle butterfly transform
    float value = input[tid];
    value = warp_butterfly_transform(value, lane);

    // Use shared memory for cross warp stages
    __shared__ float partial_transform[256];
    partial_transform[tid] = value;
    __syncthreads();

    // Butterfly transform across warps
    for (int mask = 32; mask < 256; mask *= 2) {
        // Read partner using value using xor indexing, like in the warp reduce
        float other = partial_transform[tid ^ mask];

        // Sync to make sure all threads have the pre-update value
        __syncthreads();

        // Compute the transform
        if ((tid & mask) == 0) {
            value = value + other;
        } else {
            value = other - value;
        }

        // Write transform to shared mem
        partial_transform[tid] = value;
        
        // Sync to make sure we don't go to next stage with unfinished threads
        __syncthreads();
    }

    // Write final result
    output[tid] = value;
}

void walsh_hadamard(const float* input, float* output) {

    dim3 threads_per_block(256);
    dim3 num_blocks(1);

    walsh_hadamard_transform<<<num_blocks, threads_per_block>>>(input, output);
}