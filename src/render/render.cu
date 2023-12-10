#include "render.h"
#include "../globals/globals.h"
#include <stdio.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

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

void checkGlfwInit() 
{
  if (!glfwInit()) 
  {
    printf("Failed to initialize GLFW");
    exit(GLFW_ERROR);
  } 
  else
  {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
  }
}

void renderFrame(float* result) // todo: rewrite with modern OpenGL
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
