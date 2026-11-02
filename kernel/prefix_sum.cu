#include <iostream>
#include <cuda_runtime.h>

__device__ float warp_prefix_sum_(float value, int lane) {
    for (int offset = 1; offset < 32; offset *= 2) {
        float other = __shfl_up_sync(
            0xffffffff,
            value,
            offset
        );

        if (lane >= offset) {
            value += other;
        }
    }

    return value;
}

__global__ void block_prefix_sum_(float* input, float* block_totals) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    int global_tid = blockDim.x * blockIdx.x + tid;

    // Warp level prefix sum
    float value = input[global_tid];
    value = warp_prefix_sum_(value, lane);

    // Save in shared memory
    __shared__ float warp_totals[32];
    if (lane == 31) {
        warp_totals[warp] = value;
    }
    __syncthreads();

    // Prefix sum on warp totals
    if (warp == 0) {
        float warp_sum = warp_totals[lane];
        warp_sum = warp_prefix_sum_(warp_sum, lane);
        warp_totals[lane] = warp_sum;
    }
    __syncthreads();

    // Add each warp total to the next one
    if (warp > 0) {
        value += warp_totals[warp - 1];
    }

    // Save to global input
    input[global_tid] = value;

    // Write final block sum
    if (tid == 1023) {
        block_totals[blockIdx.x] = value;
    }
}

__global__ void cross_block_prefix_sum(float* input, float* block_total) {
    int tid = blockDim.x * (blockIdx.x + 1) + threadIdx.x;
    
    // Compute local offset
    float offset = 0.0f;
    for (int i = 0; i < (blockIdx.x + 1); i += 1) {
        offset += block_total[i];
    }

    // Add offset to all block in place
    input[tid] += offset;
}

void prefix_sum_(float* input) {
    int N = 4096;

    // Allocate block totals
    float* block_totals = nullptr;
    cudaError_t err = cudaMalloc((void**)&block_totals, 4 * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA Malloc failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Compute block local prefix sum
    block_prefix_sum_<<<4, 1024>>>(input, block_totals);
    // Add last block offset to block for final prefix sum
    cross_block_prefix_sum<<<3, 1024>>>(input, block_totals);

    cudaFree(block_totals);
}