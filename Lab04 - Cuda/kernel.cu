#include <stdio.h>
#include <conio.h>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define N 1024 * 32

bool isEqual(int *cpu, int *gpu) 
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (cpu[i * N + j] != gpu[i * N + j]) {
                return false;
            }
        }
    }

    return true;
}

void cpu_compare(int *a, int *b, int *c)
{
    int tidx = 0;
    int tidy = 0;

    while (tidx < N) {
        while (tidy < N) {
            if (a[tidx * N + tidy] + b[tidx * N + tidy] > 50) {
                c[tidx * N + tidy] = 1;
            } else {
                c[tidx * N + tidy] = 0;
            }
            tidy++;
        }
        tidx++;
        tidy = 0;
    }
}

__global__ void kernel(int *a, int *b, int *c) 
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    if (tidx < N && tidy < N) {
        if ((a[tidx * N + tidy] + b[tidx * N + tidy]) > 50) {
            c[tidx * N + tidy] = 1;
        } else {  
            c[tidx * N + tidy] = 0;
        }
    }
}

void allocate_memory(int** arr, int n, int m) 
{
    *arr = (int*) malloc(n * m * sizeof(int));
}

void deallocate_memory(int** arr, int n){
    free(*arr); 
}

int main()
{
    int *a, *b, *c, *d;
    int *dev_a, *dev_b, *dev_c;
    
    allocate_memory(&a, N, N);
    allocate_memory(&b, N, N);
    allocate_memory(&c, N, N);
    allocate_memory(&d, N, N);

    cudaMalloc((void**) &dev_a, N * N * sizeof(int));
    cudaMalloc((void**) &dev_b, N * N * sizeof(int));
    cudaMalloc((void**) &dev_c, N * N * sizeof(int));
 
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = i + j;
            b[i * N + j] = i * j;
        }   
    }

    cudaMemcpy(dev_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, d, N * N * sizeof(int), cudaMemcpyHostToDevice);

    clock_t end_cpu, start_cpu = clock();

    cpu_compare(a, b, c);

    end_cpu = clock();

    
    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    float elapsedtime;

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    cudaEventCreate(&start);
    cudaEventCreate(&stop); 
    cudaEventRecord(start, 0);

    kernel<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(d, dev_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaStatus = cudaDeviceSynchronize();
    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&elapsedtime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    bool equal = isEqual(c, d);

    printf("CPU time: %.1lf ms\nGPU time: %.1lf ms\nIs arrays equal: %s", 
         (double) (end_cpu - start_cpu), elapsedtime, equal ? "yes" : "no");

    deallocate_memory(&a, N);
    deallocate_memory(&b, N);
    deallocate_memory(&c, N);
    cudaFree(&dev_a);
    cudaFree(&dev_b);
    cudaFree(&dev_c);

    getch();
}