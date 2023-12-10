#include <stdio.h>
#include "cuda.h"
#include "../globals/globals.h"

__global__ void kernel(float* result, int N) 
{ 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  result[idx] = 1.0 / (100 * (float) (idx - N / 2) / (float) N);
}

float* internalCall() 
{
    float *hostArray, *deviceArray;
    hostArray = (float*) malloc(N * sizeof(float));
    cudaMalloc((void**) &deviceArray, N * sizeof(float));
    kernel<<< N / CORES, CORES >>>(deviceArray, N);
    cudaMemcpy(hostArray, deviceArray, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(deviceArray);
    return hostArray;
} 

float* calculate()
{
    float timerValueGPU, timerValueCPU;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float* hostArray = internalCall();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU, start, stop);
    printf("GPU calculation time: %f ms\n", timerValueGPU);
  
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return hostArray;
}

void checkCudaLaunched() 
{
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
  {
    fprintf(stderr, "Cannot launch CUDA kernel: %s\n", cudaGetErrorString(err));
    exit(err);
  } 
}