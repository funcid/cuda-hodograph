#include <stdio.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cassert>

#include "render.h"
#include "../globals/globals.h"

void checkGlewInit()
{
  glewExperimental = GL_TRUE;
  assert(glewInit() == GLEW_OK && "Failed to initialize GLEW");
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
  assert(glfwInit() && "Failed to initialize GLFW");
  openglSetup();
}

void renderFrame(float* result)
{
  glVertexPointer(2, GL_FLOAT, 0, result);
  glEnableClientState(GL_VERTEX_ARRAY);
  glDrawArrays(GL_POINTS, 0, N);
  glDisableClientState(GL_VERTEX_ARRAY);
}
