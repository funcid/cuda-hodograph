#include <stdio.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cassert>

#include "render/render.h"
#include "cuda/cuda.h"

#define WIDTH (1000)
#define HEIGHT (800)

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

int main(void) 
{
  checkGlfwInit();

  // CUDA calculations
  float* hostArray = calculate();
  checkCudaLaunched();

  // Windows application open
  GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Hodograph", NULL, NULL);
  assert(window != NULL && "Failed to open GLFW window");

  glfwMakeContextCurrent(window);
  checkGlewInit();
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

  printf("Renderer: %s\n", glGetString(GL_RENDERER));
  printf("OpenGL version: %s\n", glGetString(GL_VERSION));

  while(!glfwWindowShouldClose(window)) {
    glfwPollEvents();  
    glClear(GL_COLOR_BUFFER_BIT);
    renderFrame(hostArray);
    glfwSwapBuffers(window);
  }

  free(hostArray); 
  glfwTerminate();
  return 0;
}
