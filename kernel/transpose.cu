#include <cuda_runtime.h>

TILE_SIZE = 16;

__global__ void transpose(const float* input, float* output, int H, int W, int TILE_SIZE) {
    // Allocate tile in shared memory, extra column prevents column-access bank conflicts
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    // Get indexes variables
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Load tile in shared memory
    if (row < H && col < W) {
        tile[ty][tx] = input[row * W + col];
    }
    __syncthreads();

    // Swap threads and blocks for coalescing write
    // Swap ty and tx to transpose the tile
    // Swap block.x and block.y to transpose the tile position in the global output
    int out_row = blockIdx.x * TILE_SIZE + ty;
    int out_col = blockIdx.y * TILE_SIZE + tx;

    // Write transpose
    if (out_row < W && out_col < H) {
        output[out_row * H + out_col] = tile[tx][ty];
    }
}

// Host-side code
void solve(
    const float* input,
    float* output,
    int H,
    int W
) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);

    dim3 numBlocks(
        (W + TILE_SIZE - 1) / TILE_SIZE,
        (H + TILE_SIZE - 1) / TILE_SIZE
    );

    transpose<<<numBlocks, threadsPerBlock>>>(
        input, output, H, W, TILE_SIZE
    );
}