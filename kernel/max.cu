#include <cuda_runtime.h>

// A candidate contains both its value and its ORIGINAL index.
struct MaxPair {
    float value;
    int index;
};

// Return the candidate with the highest value.
// Break ties by keeping the smallest index.
__device__ __forceinline__
MaxPair better(MaxPair a, MaxPair b) {
    if (b.value > a.value ||
        (b.value == a.value && b.index < a.index)) {
        return b;
    }
    return a;
}


// Warp level reduction
__device__ __forceinline__
MaxPair warp_max(MaxPair result) {
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


// Block level reduction
__device__ MaxPair block_argmax_reduce(MaxPair candidate) {
    int tid = threadIdx.x;

    __shared__ MaxPair shared[256];

    // Each thread writes its local candidate
    shared[tid] = candidate;
    __syncthreads();

    // First 3 stages: shared-memory reduction
    for (int offset = 128; offset >= 32; offset /= 2) {
        if (tid < offset) {
            shared[tid] = better(
                shared[tid],
                shared[tid + offset]
            );
        }

        __syncthreads();
    }

    // Last 5 stages: warp-level reduction
    MaxPair result = candidate;

    if (tid < 32) {
        result = warp_max(shared[tid]);
    }

    // Only thread 0's result is the block maximum.
    return result;
}


// =================================================
// KERNEL 1: Reduce 1,048,576 values -> 1024 pairs
// =================================================
__global__ void first_argmax(
    const float* input,
    MaxPair* partials
) {
    int tid = threadIdx.x;

    // Each block handles 256 * 4 = 1024 elements
    int base = blockIdx.x * 1024 + tid * 4;

    // -----------------------------------------
    // STEP 1: Thread-local reduction
    // -----------------------------------------

    MaxPair best = {input[base], base};

    // Each thread loads and reduces 4 elements
    for (int k = 1; k < 4; k++) {
        int idx = base + k;

        MaxPair candidate = {
            input[idx],
            idx
        };

        best = better(best, candidate);
    }

    // -----------------------------------------
    // STEP 2: Block-level reduction
    // -----------------------------------------

    MaxPair result = block_argmax_reduce(best);

    // Each block outputs one candidate
    if (tid == 0) {
        partials[blockIdx.x] = result;
    }
}


// =================================================
// KERNEL 2: Reduce 1024 pairs -> 1 final pair
// =================================================
__global__ void final_argmax(
    const MaxPair* partials,
    float* max_value,
    int* max_index
) {
    int tid = threadIdx.x;

    // -----------------------------------------
    // STEP 1: Thread-local reduction
    // -----------------------------------------

    MaxPair best = partials[tid * 4];

    for (int k = 1; k < 4; k++) {
        MaxPair candidate = partials[tid * 4 + k];

        best = better(best, candidate);
    }

    // -----------------------------------------
    // STEP 2: Block-level reduction
    // -----------------------------------------

    MaxPair result = block_argmax_reduce(best);

    // Thread 0 writes the final result
    if (tid == 0) {
        *max_value = result.value;
        *max_index = result.index;
    }
}


// =================================================
// HOST FUNCTION
// =================================================
void get_max(
    const float* input,
    float* max_value,
    int* max_index
) {
    constexpr int N = 1 << 20;
    constexpr int THREADS = 256;
    constexpr int ELEMENTS_PER_THREAD = 4;

    constexpr int BLOCKS =
        N / (THREADS * ELEMENTS_PER_THREAD);

    // BLOCKS = 1024
    MaxPair* partials = nullptr;

    cudaMalloc(
        (void**)&partials,
        BLOCKS * sizeof(MaxPair)
    );

    // Stage 1: 1,048,576 -> 1024 candidates
    first_argmax<<<BLOCKS, THREADS>>>(
        input,
        partials
    );

    // Stage 2: 1024 -> 1 candidate
    final_argmax<<<1, THREADS>>>(
        partials,
        max_value,
        max_index
    );

    cudaFree(partials);
}