#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <iostream>
#include <math.h>

#define N 10000000 // vectors size = 10 million
#define BLOCK_SIZE_1D 1024
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8

void vecAddCpu(float *a, float *b, float *c, int n){
    for(int i = 0; i < n; ++i){
        c[i] = a[i] + b[i];
    }
}

__global__ void vecAddGpu1D(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] = a[i] + b[i];
    }
}

__global__ void vecAddGpu3D(float *a, float *b, float *c, int nx, int ny, int nz){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i < nx && j < ny && k < nz){
        int idx = i + j * nx + k * nx * ny;
        c[idx] = a[idx] + b[idx];
    }

    // the 3D implementation uses a lot more functions, so only use it when necessary
    // e.g. when working with a 3D space or data
}

void init_vector(float *vec, int n){
    for(int i = 0; i < n; ++i){
        vec[i] = (float)rand() / RAND_MAX;
    }
}

double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu_1d, *h_c_gpu_3d;
    float *d_a, *d_b, *d_c_1d, *d_c_3d;
    size_t size = N * sizeof(float);

    // Alllocate host memory
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c_cpu = (float *)malloc(size);
    h_c_gpu_1d = (float *)malloc(size);
    h_c_gpu_3d = (float *)malloc(size);

    // Initialize vectors
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_1d, size);
    cudaMalloc(&d_c_3d, size);

    // Copy vectors from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define grid and dimensions for 1D
    int num_blocks_1d = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    // Define grid and dimensions for 3D
    int nx = 100, ny = 100, nz = 1000; // N = 100 * 100 * 1000 = 10000000
    dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 num_blocks_3d(
        (nx + block_size_3d.x - 1) / block_size_3d.x,
        (ny + block_size_3d.y - 1) / block_size_3d.y,
        (nz + block_size_3d.z - 1) / block_size_3d.z
    );

    // Warm up runs
    for(int i = 0; i < 3; ++i){
        vecAddCpu(h_a, h_b, h_c_cpu, N);
        vecAddGpu1D<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        vecAddGpu3D<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for(int i = 0; i < 20; ++i){
        double start_time = get_time();
        vecAddCpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double average_cpu_time = cpu_total_time / 20.0;

    // Benchmark 1D GPU implementation
    printf("Benchmarking GPU 1D implementation...\n");
    double gpu_1d_total_time = 0.0;
    for(int i = 0; i < 20; ++i){
        double start_time = get_time();
        vecAddGpu1D<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_1d_total_time += end_time - start_time;
    }
    double average_gpu_1d_time = gpu_1d_total_time / 20.0;

    // Verify results of 1D
    cudaMemcpy(h_c_gpu_1d, d_c_1d, size, cudaMemcpyDeviceToHost);
    bool correct_1d = true;
    for(int i = 0; i < N; ++i){
        if(fabs(h_c_cpu[i] - h_c_gpu_1d[i]) > 1e-5){
            correct_1d = false;
            std::cout << i << " cpu: " << h_c_cpu[i] << " != gpu: " << h_c_gpu_1d[i] << std::endl;
            break;
        }
    }
    printf("1D Results are %s\n", correct_1d ? "correct" : "incorrect");

    // Benchmark 3D GPU implementation
    printf("Benchmarking GPU 3D implementation...\n");
    double gpu_3d_total_time = 0.0;
    for(int i = 0; i < 20; ++i){
        double start_time = get_time();
        vecAddGpu3D<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_3d_total_time += end_time - start_time;
    }
    double average_gpu_3d_time = gpu_3d_total_time / 20.0;

    // Verify results of 3D
    cudaMemcpy(h_c_gpu_3d, d_c_3d, size, cudaMemcpyDeviceToHost);
    bool correct_3d = true;
    for(int i = 0; i < N; ++i){
        if(fabs(h_c_cpu[i] - h_c_gpu_3d[i]) > 1e-5){
            correct_3d = false;
            std::cout << i << " cpu: " << h_c_cpu[i] << " != gpu: " << h_c_gpu_3d[i] << std::endl;
            break;
        }
    }
    printf("3D Results are %s\n", correct_3d ? "correct" : "incorrect");

    printf("Average CPU time: %f milliseconds\n", average_cpu_time * 1000);
    printf("Average GPU 1D time: %f milliseconds\n", average_gpu_1d_time * 1000);
    printf("Average GPU 3D time: %f milliseconds\n", average_gpu_3d_time * 1000);
    printf("Speedup (CPU vs GPU 1D): %fx\n", average_cpu_time / average_gpu_1d_time);
    printf("Speedup (CPU vs GPU 3D): %fx\n", average_cpu_time / average_gpu_3d_time);
    printf("Speedup (GPU 1D vs GPU 3D): %fx\n", average_gpu_1d_time / average_gpu_3d_time);

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu_1d);
    free(h_c_gpu_3d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_1d);
    cudaFree(d_c_3d);
    
    return 0;
}