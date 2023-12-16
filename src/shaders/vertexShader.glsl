#version 330 core
layout (location = 0) in float value;
void main()
{
    gl_Position = vec4(value, 0.0, 0.0, 1.0);
}
