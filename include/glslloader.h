#ifndef GLSL_LOADER_H
#define GLSL_LOADER_H

#include <iostream>
#include <string>
#include <map>
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace glsl {
    GLuint createShaderProgram(const char *vert_filename, const char *frag_filename);
    void linkShaderProgram(GLuint program);
    void getShaderProgramUniforms(GLuint program, std::map<std::string,GLint>& uniforms);
}

#endif // GLSL_LOADER_H
