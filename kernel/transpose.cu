#include <cuda_runtime.h>

__global__ void transpose(const float* input, float* output, int H, int W, int tile_size) {
    // Allocate tile in shared memory, extra column prevents column-access bank conflicts
    __shared__ float tile[tile_size][tile_size + 1];

    // Get indexes variables
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * tile_size + ty;
    int col = blockIdx.x * tile_size + tx;

    // Load tile in shared memory
    if (row < H && col < W) {
        tile[ty][tx] = input[row * W + col];
    }
    __syncthreads();

    // Swap threads and blocks for coalescing write
    // Swap ty and tx to transpose the tile
    // Swap block.x and block.y to transpose the tile position in the global output
    int out_row = blockIdx.x * tile_size + ty;
    int out_col = blockIdx.y * tile_size + tx;

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
    constexpr int tile = 16;
    dim3 threadsPerBlock(tile, tile);

    dim3 numBlocks(
        (W + tile - 1) / tile,
        (H + tile - 1) / tile
    );

    transpose<<<numBlocks, threadsPerBlock>>>(
        input, output, H, W, tile
    );
}