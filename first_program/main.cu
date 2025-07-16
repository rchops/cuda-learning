#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_from_GPU(){
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main(){
    hello_from_GPU<<<1,4>>>();
    cudaDeviceSynchronize();
    return 0;
}