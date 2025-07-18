#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000000 // vector size of 10 million
#define BLOCK_SIZE 256

// E.g.
// A = [1, 2, 3, 4, 5]
// B = [6, 7, 8, 9, 10]
// C = A + B = [7, 9, 11, 13, 15]


// cpu vec addition
void vecAddCpu(float * a, float *b, float *c, int n){
    for(int i = 0; i < n; ++i){
        c[i] = a[i] + b[i];
    }
}

// gpu vec addition
__global__ void vecAddGpu(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        c[i] = a[i] + b[i]; // unrolls loop -> paralleizes across threads
    }
}

// vec initialization
void vecInit(float *vec, int n){
    for(int i = 0; i < n; ++i){
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// measure time
double getTime(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu; // vectors on host (CPU)
    float *d_a, *d_b, *d_c; // vectors on device (GPU)
    size_t size = N * sizeof(float);

    // allocate host memory
    // malloc allocates memory in heap of 'size' bytes
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c_cpu = (float *)malloc(size);
    h_c_gpu = (float *)malloc(size);

    srand(time(NULL));
    vecInit(h_a, N);
    vecInit(h_b, N);

    //allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // copying data from host to device -> have to copy data from CPU to GPU and then back when performing operations
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // define grid and block size
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // N = 1024, BLOCK_SIZE = 256, num_blocks = (1024 + 256 - 1) / 256 = 4
    // This is used instead of N / BLOCK_SIZE to make sure the value is rounded up so all threads are covered
    // If N is not divisible by BLOCK_SIZE it will round down, so N / BLOCK_SIZE would not work because it was miss needed threads
}