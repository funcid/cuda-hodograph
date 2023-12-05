#include <stdio.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define WIDTH (1000)
#define HEIGHT (800)
#define N (2048*2048)
#define CORES (1024)

__global__ void kernel(float* dA) 
{ 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dA[idx] = 1.0 / (100 * (float) (idx - N / 2) / (float) N);
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

void checkGlfwAndGlewInit() 
{
  if (!glfwInit()) 
  {
    printf("Failed to initialize GLFW");
    exit(-389);
  }
  if (!glewInit())
  {
    printf("Failed to initialize GLEW");
    exit(-340);
  }
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void renderFrame(float* result) 
{
  glBegin(GL_LINE_STRIP);
  glVertex2f(-1, 0);
  glVertex2f(1, 0);
  glEnd();
  glBegin(GL_LINE_STRIP);
  glVertex2f(0, 1);
  glVertex2f(0, -1);
  glEnd();
  glBegin(GL_LINE_STRIP);
  for (int i = 0; i < N; i++) 
  { 
    glVertex2f(2 * (i * 1.0 / N - 0.5), result[i]);
  }
  glEnd();
}

int main(void) 
{
  // CUDA calculations
  float timerValueGPU, timerValueCPU;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  
  float *hostArray, *deviceArray;
  hostArray = (float*) malloc(N * sizeof(float));
  cudaMalloc((void**) &deviceArray, N * sizeof(float));
  kernel<<< N / CORES, CORES >>>(deviceArray);
  cudaMemcpy(hostArray, deviceArray, N * sizeof(float), cudaMemcpyDeviceToHost);
  checkCudaLaunched();
  checkGlfwAndGlewInit();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timerValueGPU, start, stop);
  printf("GPU calculation time: %f ms\n", timerValueGPU);

  cudaFree(deviceArray);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Windows application open
  GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Hodograph", NULL, NULL);
  if (window == NULL)
  {
    printf("Failed to open GLFW window");
    exit(-341);
  }

  glfwMakeContextCurrent(window);
  glViewport(0, 0, WIDTH, HEIGHT);
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

  printf("Renderer: %s\n", glGetString(GL_RENDERER));
  printf("OpenGL version: %s\n", glGetString(GL_VERSION));

  while(!glfwWindowShouldClose(window)) 
  {
    glfwPollEvents();  
    glClear(GL_COLOR_BUFFER_BIT);
    renderFrame(hostArray);
    glfwSwapBuffers(window);
  }

  free(hostArray); 
  glfwTerminate();
  return 0;
}