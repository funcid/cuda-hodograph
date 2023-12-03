#include <stdio.h>
#define N (4096*4096)
#define CORES (1024)

__global__ void kernel(float* dA) { 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float x = 2.0f * 3.1415926f * (float) idx / (float) N;
  dA[idx] = sinf(sqrtf(x));
}

bool checkLaunched() {
  cudaError_t err;
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Cannot launch CUDA kernel: %s\n", cudaGetErrorString(err));
    return false;
  } 
  return true;
}

int main(void) {
  float timerValueGPU, timerValueCPU;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  
  float *hA, *dA;
  hA = (float*) malloc(N * sizeof(float));
  cudaMalloc((void**) &dA, N * sizeof(float));
  kernel <<< N / CORES, CORES >>> (dA);
  cudaMemcpy(hA, dA, N * sizeof(float), cudaMemcpyDeviceToHost);

  if (!checkLaunched()) return 1;

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timerValueGPU, start, stop);
  printf("\n GPU calculation time: %f ms\n", timerValueGPU);

  //for (int idx = 0; idx < N; idx++) {
  //  if (idx % 10000 == 0) {
  //    printf("a[%d] = %.5f\n", idx, hA[idx]);
  //  }
  //}

  cudaEventRecord(start, 0);
  for (int i = 0; i < N; i++) {
    hA[i] = sinf(sqrtf(2.0f * 3.1415926f * (float) i / (float) N));
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timerValueCPU, start, stop);
  printf("\n CPU calculation time: %f ms\n", timerValueCPU);
  printf("\n Rate: %fx\n", timerValueCPU / timerValueGPU);

  free(hA); cudaFree(dA);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}