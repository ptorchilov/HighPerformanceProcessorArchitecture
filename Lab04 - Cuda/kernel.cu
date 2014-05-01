#include <stdio.h>
#include <conio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define threads_number 512
#define blocks_number 8
#define N threads_number * blocks_number

bool isEqual(int cpu, int gpu) 
{
    if (cpu == gpu) {
        return true;
    }

    return false;
}

void cpu_compare(int *a, int *b, int *c)
{
    int tidx = 0;
    int tidy = 0;

    while (tidx < N * N) {
        //while (tidy < N) {
            if (a[tidx/* * N + tidy*/] + b[tidx /** N + tidy*/] > 50) {
                *c ^= (1u << tidx % 32);
            } 
         //   tidy++;
       // }
        tidx++;
      //  tidy = 0;
    }
}

__global__ void kernel(int *a, int *b, int *c) 
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    //int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = blockDim.x * gridDim.x;
    __shared__ int temp[512];
    temp[threadIdx.x] = 0;
    __syncthreads();
    

    
    while (tidx < N * N/* && tidy < N*/) {
        if ((a[tidx/* * N + tidy*/] + b[tidx/* * N + tidy*/]) > 50) {
            //atomicXor(c, 1u << tidx % 32);
            atomicXor(&(temp[threadIdx.x]), 1u << tidx % 32);
            //temp++;
            //__syncthreads();
            //atomicXor(c, temp[threadIdx.x]);
        }
        tidx += offset;
    }
    
    
    // __syncthreads();       
    if (temp[threadIdx.x] != 0) {
        atomicXor(c, temp[threadIdx.x]);
    }
}

void allocate_memory(int** arr, int n, int m)
{
    *arr = (int*) malloc(n * m * sizeof(int));
}

void deallocate_memory(int** arr, int n)
{
    free(*arr); 
}

int random() 
{
    return rand() % 51;
}

int main()
{
    int *a, *b;
    int c = 0; 
    int d = 0;
    int *dev_a, *dev_b, *dev_c;
    
    srand (time(NULL));

    allocate_memory(&a, N, N);
    allocate_memory(&b, N, N);

    cudaMalloc((void**) &dev_a, N * N * sizeof(int));
    cudaMalloc((void**) &dev_b, N * N * sizeof(int));
    cudaMalloc((void**) &dev_c, sizeof(int));
 
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = random();
            b[i * N + j] = random();
        }   
    }

    cudaMemcpy(dev_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(dev_c, 0, sizeof(int));

    clock_t end_cpu, start_cpu = clock();

    cpu_compare(a, b, &c);

    end_cpu = clock();

    
    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    float elapsedtime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop); 
    cudaEventRecord(start, 0);

    kernel<<<blocks_number, threads_number>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(&d, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    cudaStatus = cudaDeviceSynchronize();
    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&elapsedtime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    bool equal = isEqual(c, d);

    printf("CPU: time = %.1lf ms, value = %i\nGPU: time = %.1lf ms, value = %i\nIs values equal: %s", 
         (double) (end_cpu - start_cpu), c, elapsedtime, d, equal ? "yes" : "no");

    deallocate_memory(&a, N);
    deallocate_memory(&b, N);

    cudaFree(&dev_a);
    cudaFree(&dev_b);
    cudaFree(&dev_c);

    getch();
}