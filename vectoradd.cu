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
    // index of block we are at (number of blocks) * size of block (number of threads) +  whatever thread we are at
    // gives us the thread in that line of the grid -> thread index -> one for each element in vector (i < n)
    if(i < n) {
        c[i] = a[i] + b[i]; 
        // use thread index to access elements in vector and add
        // unrolls loop -> paralleizes across threads
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
    // If N is not divisible by BLOCK_SIZE it will round down, so N / BLOCK_SIZE would not work because it will miss needed threads

    // Warm-up for benchmarking
    printf("Warming up GPU...\n");
    vecAddCpu(h_a, h_b, h_c_cpu, N);
    vecAddGpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    // Benchmark CPU
    printf("Benchmarking CPU...\n");
    double cpu_total_time = 0.0;
    for(int i = 0; i < 20; ++i){
        double start_time = getTime();
        vecAddCpu(h_a, h_b, h_c_cpu, N);
        double end_time = getTime();
        cpu_total_time += end_time - start_time;
    }
    double average_cpu_time = cpu_total_time / 20.0;

    // Benchmark GPU
    printf("Benchmarking GPU...\n");
    double gpu_total_time = 0.0;
    for(int i = 0; i < 20; ++i){
        double start_time = getTime();
        vecAddGpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end_time = getTime();
        gpu_total_time += end_time - start_time;
    }
    double average_gpu_time = gpu_total_time / 20.0;

    printf("Average CPU time: %f milliseconds\n", average_cpu_time * 1000);
    printf("Average GPU time: %f milliseconds\n", average_gpu_time * 1000);
    printf("Speedup: %fx\n", average_cpu_time / average_gpu_time);

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}