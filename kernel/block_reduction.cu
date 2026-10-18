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
        output[blockIdx.x] = buffer[0]
    }
}