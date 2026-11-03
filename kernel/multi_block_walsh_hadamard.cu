#include <cuda_runtime.h>

THREADS_PER_BLOCK = 256;

__device__ float warp_walsh_hadamard_tranform(
    float value,
    int lane
) {
    for (int mask=1; mask < 32; mask *= 2) {
        float other = __shfl_xor_sync(
            0xffffffff,
            value,
            mask
        )

        if ((lane & mask) == 0) {
            value = value + other;
        } else {
            value = other - value;
        }
    };

    return value;
}

__global__ void walsh_hadamard_transform_warplevel(
    const float* input,
    float* output
){
    int tid = threadIdx.x;
    int lane = tid % 32;
    int global_tid = blockDim.x * blockIdx.x + tid;

    // Stage 1 to 5
    float value = input[global_tid]
    value = warp_walsh_hadamard_tranform(value, lane);

    // stage 6 to 8
    __shared__ float partial[THREADS_PER_BLOCK]
    partial[tid] = value;
    __syncthreads();

    for (int mask = 32; mask < 256; mask *= 2) {
        float other = partial[tid ^ mask];
        __syncthreads();

        if ((lane & mask) == 0) {
            value = value + other;
        } else {
            value = other - value;
        }

        partial[tid] = value;
        __syncthreads();
    }

    output[tid] = value
}

__global__ void walsh_hadamard_transform_global(
    const float* input,
    float* output,
    int stride,
    int N
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x

    if (tid >= N / 2)
        return;

    // Compute left and right butterfly indices
    int group = tid / stride;
    int position = tid % stride;
    int left = group * (2 * stride) + position;
    int right = left + stride;

    // load both values
    float value = input[right];
    float other = input[left];

    // Compute walsh-hadamard transform
    float a = value + other;
    float b = other - value;

    // Write back to global mem
    output[left] = a;
    output[right] = b;
}


float walsh_hadamard(const float* input, float* output, int N) {
    constexpr N = 1024;
    dim3 threads_per_block(THREADS_PER_BLOCK);
    dim3 num_blocks(4)

    // First 8 stages using warp level instructions
    walsh_hadamard_transform_warplevel<<<num_blocks, threads_per_block>>>(input, output);

    // Stage 9, 2 times less threads since one thread handle a pair now
    walsh_hadamard_transform_global<<<num_blocks/2, threads_per_block>>>(input, output, 256, N);

    // Stage 10
    walsh_hadamard_transform_global<<<num_blocks/2, threads_per_block>>>(input, output, 512, N)
}