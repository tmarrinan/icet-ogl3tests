#version 150

#define EPSILON 0.000001

in vec3 vertex_position;
in vec2 vertex_texcoord;
in vec3 point_center;
in vec3 point_color;
in float point_size;

uniform vec3 camera_position;
uniform mat4 projection_matrix;
uniform mat4 view_matrix;
uniform mat4 model_matrix;

out vec3 world_position;
out mat3 world_normal_mat;
out vec2 model_texcoord;
out vec3 model_color;
out vec3 model_center;
out float model_size;

void main() {
    vec3 world_point = (model_matrix * vec4(point_center, 1.0)).xyz;
    vec3 vertex_direction = normalize(world_point - camera_position);

    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 right = cross(vertex_direction, up);
    vec3 cam_right = (length(right) > EPSILON) ? normalize(right) : vec3(0.0, 0.0, 1.0);
    vec3 cam_up = cross(cam_right, vertex_direction);

    world_position = world_point + cam_right * vertex_position.x * point_size +
                                         cam_up * vertex_position.y * point_size;

    vec3 n = -vertex_direction;
    vec3 u = normalize(cross(up, n));
    vec3 v = cross(n, u);
    world_normal_mat = mat3(u, v, n);

    model_texcoord = vertex_texcoord;
    model_color = point_color;
    model_center = world_point;
    model_size = point_size;

    gl_Position = projection_matrix * view_matrix * vec4(world_position, 1.0);
}
