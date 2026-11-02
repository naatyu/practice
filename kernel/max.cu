#include <cuda_runtime.h>

struct MaxPair {
    float value;
    int index;
};

__device__ MaxPair better(MaxPair a, MaxPair b) {
    if (b.value > a.value ||
        (b.value == a.value && b.index < a.index)) {
        return b;
    }
    return a;
}

__device__ MaxPair warp_max(MaxPair result) {
    int lane = threadIdx.x % 32;

    for (int offset = 16; offset >= 1; offset /= 2) {
        MaxPair other;

        other.value = __shfl_down_sync(
            0xffffffff, result.value, offset
        );

        other.index = __shfl_down_sync(
            0xffffffff, result.index, offset
        );

        if (lane < offset) {
            result = better(result, other);
        }
    }

    return result;
}

__global__ void block_argmax(
    const float* input,
    float* max_value,
    int* max_index
) {
    int tid = threadIdx.x;

    __shared__ float values[256];
    __shared__ int indices[256];

    values[tid] = input[tid];
    indices[tid] = tid;

    __syncthreads();

    // Block-level reduction
    for (int offset = 128; offset >= 32; offset /= 2) {

        if (tid < offset) {
            MaxPair a = {values[tid], indices[tid]};
            MaxPair b = {values[tid + offset], indices[tid + offset]};

            MaxPair best = better(a, b);

            values[tid] = best.value;
            indices[tid] = best.index;
        }

        __syncthreads();
    }

    // Warp-level reduction
    if (tid < 32) {
        MaxPair result = {values[tid], indices[tid]};

        result = warp_max(result);

        if (tid == 0) {
            *max_value = result.value;
            *max_index = result.index;
        }
    }
}

void get_max(
    const float* input,
    float* max_value,
    int* max_index
) {
    block_argmax<<<1, 256>>>(
        input, max_value, max_index
    );
}