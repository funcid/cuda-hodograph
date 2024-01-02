#include "render.h"
#include <stdio.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "../globals/globals.h"

#define GLEW_ERROR (-389)
#define GLFW_ERROR (-390)

void checkGlewInit()
{
  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    printf("Failed to initialize GLEW");
    exit(GLEW_ERROR);
  }
}

void openglSetup() 
{
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);
  glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
}

void checkGlfwInit() 
{
  if (!glfwInit()) 
  {
    printf("Failed to initialize GLFW");
    exit(GLFW_ERROR);
  } 
  else
  {
    openglSetup();
  }
}

void renderFrame(float* result)
{
  glVertexPointer(2, GL_FLOAT, 0, result);
  glEnableClientState(GL_VERTEX_ARRAY);
  glDrawArrays(GL_POINTS, 0, N);
  glDisableClientState(GL_VERTEX_ARRAY);
}
