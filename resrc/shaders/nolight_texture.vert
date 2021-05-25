#version 150

uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 projection_matrix;

in vec3 vertex_position;
in vec2 vertex_texcoord;

out vec2 world_texcoord;

void main() {
    gl_Position = projection_matrix * view_matrix * model_matrix * vec4(vertex_position, 1.0);
    world_texcoord = vertex_texcoord;
}
