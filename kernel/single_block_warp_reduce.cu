#include <cuda_runtime.h>

__device__ float warp_reduce(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(
            0xffffffff,
            value,
            offset
        );
    }

    return value;
}

__global__ void reduce(const float* input, float* output) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    // Create shared buffer to store patial warp reductions
    __shared__ float partial[8];

    // Load input directly in register for warp shuffle
    float value = input[tid];

    // Reduce within each warp
    float warp_sum = warp_reduce(value);

    // Write lane 0 results
    if (lane == 0) {
        partial[warp_id] = warp_sum;
    }
    __syncthreads();

    // First warp reduce the partial sums
    if (warp_id == 0) {
        // Only the first 8 lanes participates since there is 8 warps so 8 partial sum
        // We load the 8 partial sum into register of warp 0 to use warp shuffle for 
        // the final reduction
        float block_sum = (lane < 8) ? partial[lane] : 0.0f;
        float result = warp_reduce(block_sum);

        // After final warp reduction only lane 0 has the final sum
        if (lane == 0) {
            output[0] = result;
        }
    }
}

void solve(float* input, float* output) {
    // Grid with 8 warps
    reduce<<<1, 256>>>(input, output);
}