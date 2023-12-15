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

void renderFrame(float* result)
{
  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  GLuint vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  glBufferData(GL_ARRAY_BUFFER, N * sizeof(float), result, GL_STATIC_DRAW);

  glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  glBindVertexArray(vao);
  glDrawArrays(GL_LINE_STRIP, 0, N);
  glBindVertexArray(0);

  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &vbo);
}

