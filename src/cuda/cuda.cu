#include "cuda.h"
#include <stdio.h>
#include "../globals/globals.h"

__global__ void kernel(float* result, int N) 
{ 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  result[idx] = 1.0 / (100 * (float) (idx - N / 2) / (float) N);
}

float* call()
{
    float *hostArray, *deviceArray;
    hostArray = (float*) malloc(N * sizeof(float));
    cudaMalloc((void**) &deviceArray, N * sizeof(float));
    kernel<<< N / CORES, CORES >>>(deviceArray, N);
    cudaMemcpy(hostArray, deviceArray, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(deviceArray);
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
