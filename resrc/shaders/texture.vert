#version 150 core

in vec3 vertex_position;
in vec3 vertex_normal;
in vec2 vertex_texcoord;

uniform mat4 model_matrix;
uniform mat3 normal_matrix;
uniform mat4 view_matrix;
uniform mat4 projection_matrix;

out vec3 world_position;
out vec3 world_normal;
out vec2 world_texcoord;

void main() {
    vec4 position = model_matrix * vec4(vertex_position, 1.0);

    world_position = position.xyz;
    world_normal = normalize(normal_matrix * vertex_normal);
    world_texcoord = vertex_texcoord;

    gl_Position = projection_matrix * view_matrix * position;
}
