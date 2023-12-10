#include <stdio.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "render/render.h"
#include "cuda/cuda.h"

#define WIDTH (1000)
#define HEIGHT (800)

class VertexBufferObject
{
private:
  GLuint id;

public:
  VertexBufferObject()
  {
    glGenBuffers(1, &id);
  }

};

class VertexArrayObject
{
private:
  GLuint id;

public:
  void bind() 
  {
    glBindVertexArray(id);
  }

  void unbind() 
  {
    glBindVertexArray(0);
  }

  VertexArrayObject()
  {
    glGenVertexArrays(1, &id);
    bind();
  }
};

void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
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
  if (window == NULL)
  {
    printf("Failed to open GLFW window");
    exit(-341);
  }

  glfwMakeContextCurrent(window);
  checkGlewInit();
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

  printf("Renderer: %s\n", glGetString(GL_RENDERER));
  printf("OpenGL version: %s\n", glGetString(GL_VERSION));

  // GL bind array
  VertexArrayObject* object = new VertexArrayObject();

  while(!glfwWindowShouldClose(window)) 
  {
    glfwPollEvents();  
    glClear(GL_COLOR_BUFFER_BIT);
    renderFrame(hostArray);
    glfwSwapBuffers(window);
  }

  free(hostArray); 
  delete object;
  glfwTerminate();
  return 0;
}