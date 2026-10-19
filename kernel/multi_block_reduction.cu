#include <cuda_runtime.h>
#include <vector>

__global__ void reduce(
    const float* x,
    float* output,
    int N
) {
    __shared__ float buffer[256];

    int tid = threadIdx.x;

    // TODO 1:
    // Calculate the global index
    int global_idx = blockDim.x * blockIdx.x + tid;

    // TODO 2:
    // Load input into shared memory
    // Handle threads outside the input bounds
    if (global_idx < N) {
        buffer[tid] = x[global_idx];
    } else {
        buffer[tid] = 0.0f;
    }
    
    __syncthreads();

    // TODO 3:
    // Perform the same reduction as before
    for (int stride = 128; stride >= 1; stride /= 2) {
        if (tid < stride){
            buffer[tid] += buffer[tid + stride];
        }
        
        if (stride > 1) {
            __syncthreads();
        }
    }

    // TODO 4:
    // Each block writes its own partial sum
    if (tid == 0) {
        output[blockIdx.x] = buffer[0];
    }
}


void solve(const float* x, float* output, int N) {
    const int threadsPerBlock = 256;

    // Assume N >= 1.

    // Special case: only one element
    if (N == 1) {
        cudaMemcpy(
            output,
            x,
            sizeof(float),
            cudaMemcpyDeviceToDevice
        );
        return;
    }

    // Current input
    const float* current_input = x;
    int current_N = N;

    // Keep track of temporary GPU allocations
    std::vector<float*> temporary_buffers;

    while (current_N > 1) {

        // Number of partial sums
        int num_partial_sums = (current_N + threadsPerBlock - 1) / threadsPerBlock;

        float* destination = nullptr;

        // If this is the final stage, write directly to output
        if (num_partial_sums == 1) {
            destination = output;
        } else {
            cudaMalloc(
                &destination,
                num_partial_sums * sizeof(float)
            );

            temporary_buffers.push_back(destination);
        }

        // Launch reduction
        reduce<<<num_partial_sums, threadsPerBlock>>>(
            current_input,
            destination,
            current_N
        );

        // Next iteration uses the previous output
        current_input = destination;
        current_N = num_partial_sums;
    }

    // Wait before releasing temporary memory
    cudaDeviceSynchronize();

    for (float* ptr : temporary_buffers) {
        cudaFree(ptr);
    }
}