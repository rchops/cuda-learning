#include <stdio.h>
#include <cuda_runtime.h>

// blockIdx => index of block in grid
// gridDim => number of blocks in grid
// blockDim => number of threads in block
// threadIdx => index of thread in block


__global__ void whoami(void){
    int block_id = // the block you are looking for -> like an apartment in a city
        blockIdx.x + // where on this floor
        blockIdx.y  * gridDim.x + // think of this as apartment - gridDim.x is a along a floor, blockIdx.y is going up floors
        blockIdx.z * gridDim.x * gridDim.y; // think of this as block - gridDim.x * gridDim.y is area of block, blockIdx.z is depth

        // from bottom up -> how many panes deep, rows high, offset on row
        // like apartment -> how many panes deep (buildings), floors up, apartment on that floor
        

    int block_offset = 
        block_id * // multiplied by our apartment number
        blockDim.x * blockDim.y * blockDim.z; // total threads per block -> multiplying threads in x, y, and z

        // finds the total number of threads are before the block you are looking for -> calculating which thread we are at
        // finding the global thread index -> e.g. if we are in block_id = 2 with x = 4, y = 2, z = 1
        // block_offset = 2 * 4 * 2 * 1 = 16 -> so we are the 16th thread in the grid


    int thread_offset = 
        threadIdx.x + 
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

        // same analogy as block_id, just at a lower level at threads instead of blocks


        int id = block_offset + thread_offset; // global thread id

        printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
               id,
               blockIdx.x, blockIdx.y, blockIdx.z, block_id,
               threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}

int main(int argc, char **argv){
    const int b_x = 2, b_y = 3, b_z = 4;
    const int t_x = 4, t_y = 4, t_z = 4; // max warp size is 32
    // with 4 threads per dimension -> 64 threads so 2 warp of 32 threads per block

    int blocks_per_grid = b_x * b_y * b_z;
    int threads_per_block = t_x * t_y * t_z;

    printf("%d blocks/grid\n", blocks_per_grid);
    printf("%d threads/block\n", threads_per_block);
    printf("%d total threads\n", blocks_per_grid * threads_per_block);

    dim3 blocksPerGrid(b_x, b_y, b_z);
    dim3 threadsPerBlock(t_x, t_y, t_z);

    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
}