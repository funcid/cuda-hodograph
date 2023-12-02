#include <stdio.h>
#define N (1024*1024)

__global__ void kernel(float* dA) { 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float x = 2.0f * 3.1415926f * (float) idx / (float) N;
  dA [idx] = sinf(sqrtf(x));
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
  float *hA, *dA; int cudaCores = 1024;
  hA = (float*) malloc (N * sizeof(float));
  cudaMalloc((void**) &dA, N * sizeof(float));
  kernel <<< N / cudaCores, cudaCores >>> (dA);
  cudaMemcpy(hA, dA, N * sizeof(float), cudaMemcpyDeviceToHost);

  if (!checkLaunched()) return 1;

  for (int idx = 0; idx < N; idx++) {
    printf("a[%d] = %.5f\n", idx, hA[idx]);
  }
  free(hA); cudaFree(dA);
  
  return 0;
}