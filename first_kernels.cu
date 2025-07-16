#include <stdio.h>
#include <cuda_runtime.h>

// blockIdx => index of block in grid
// gridDim => number of blocks in grid
// blockDim => number of threads in block
// threadIdx => index of thread in block


__global__ void whoami(void){
    int block_id = 
        blockIdx.x + // where on this floor
        blockIdx.y  * gridDim.x + // think of this as apartment - gridDim.x is a along a floor, blockIdx.y is going up floors
        blockIdx.z * gridDim.x * gridDim.y; // think of this as block - gridDim.x * gridDim.y is area of block, blockIdx.z is depth

        // from bottom up -> how many panes deep, rows high, offset on row
        // like apartment -> how many panes deep, floors up, apartment on that floor
        
    int block_offset = 
        block_id *
        blockDim.x * blockDim.y * blockDim.x; // total threads per block
}