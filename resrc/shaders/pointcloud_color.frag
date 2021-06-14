#version 150

in vec3 world_position;
in mat3 world_normal_mat;
in vec2 model_texcoord;
in vec3 model_color;
in vec3 model_center;
in float model_size;

uniform vec2 clip_z;
uniform int num_lights;
uniform vec3 light_ambient;
uniform vec3 light_position[10];
uniform vec3 light_color[10];
uniform vec3 camera_position;

out vec4 FragColor;

void main() {
    // BILLBOARD SPHERES
    vec2 norm_texcoord = (2.0 * model_texcoord) - vec2(1.0, 1.0);
    float magnitude = dot(norm_texcoord, norm_texcoord);
    if (magnitude > 1.0) {
        discard;
    }
    vec3 sphere_normal = vec3(norm_texcoord, sqrt(1.0 - magnitude));
    sphere_normal = normalize(world_normal_mat * sphere_normal);
    float sphere_radius = model_size / 2.0;
    vec3 sphere_position = (sphere_normal * sphere_radius) + model_center;

    // GPS globe only
    sphere_normal = normalize(model_center);
    //
    
    vec3 light_diffuse = vec3(0.0, 0.0, 0.0);
    for(int i = 0; i < num_lights; i++) {
        vec3 light_direction = normalize(light_position[i] - sphere_position);
        float n_dot_l = max(dot(sphere_normal, light_direction), 0.0);
        light_diffuse += light_color[i] * n_dot_l;
    }
    vec3 final_color = min((light_ambient * model_color) + (light_diffuse * model_color), 1.0);

    // Color
    FragColor = vec4(final_color, 1.0);

    // Depth
    float near = clip_z.x;
    float far = clip_z.y;
    float dist = length(sphere_position - camera_position);
    gl_FragDepth = (dist - near) / (far - near);
}