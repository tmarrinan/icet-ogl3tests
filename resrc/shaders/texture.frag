#version 150 core

in vec3 world_position;
in vec3 world_normal;
in vec2 world_texcoord;

// camera
uniform vec3 camera_position;
// ambient light
uniform vec3 light_ambient;
// point lights (up to 16)
uniform int num_lights;
uniform vec3 light_position[16];
uniform vec3 light_color[16];
uniform vec2 light_attenuation[16];
// spotlights (up to 8)
uniform int num_spotlights;
uniform vec3 spotlight_position[8];
uniform vec3 spotlight_direction[8];
uniform vec3 spotlight_color[8];
uniform vec2 spotlight_attenuation[8];
uniform float spotlight_fov[8];
// object material properties
uniform vec3 material_color;      // Ka and Kd
uniform vec3 material_specular;   // Ks
uniform float material_shininess; // n
uniform sampler2D image;

out vec4 FragColor;

void main() {
    vec3 light_diffuse = vec3(0.0, 0.0, 0.0);
    vec3 light_specular = vec3(0.0, 0.0, 0.0);

    int i;
    
    // point light illumination
    for(i = 0; i < num_lights; i++) {
        float full_light_dist = light_attenuation[i].x;
        float no_light_dist = light_attenuation[i].y;
        float attenuation_k = 1.0 / (full_light_dist * full_light_dist);
        vec3 light_vector = light_position[i] - world_position;
        float dist = length(light_vector);
        if (dist < no_light_dist) {
            //diffuse
            vec3 light_direction = normalize(light_vector);
            float light_attenuation = 1.0 / max(attenuation_k * dist * dist, 1.0);
            float n_dot_l = max(dot(world_normal, light_direction), 0.0);
            light_diffuse += light_color[i] * light_attenuation * n_dot_l;

            //specular
            vec3 reflection_direction = normalize(reflect(-light_direction, world_normal));
            vec3 view_direction = normalize(camera_position - world_position);
            float r_dot_v = max(dot(reflection_direction, view_direction), 0.0);
            light_specular += light_color[i] * light_attenuation * pow(r_dot_v, material_shininess);
        }
    }

    // spotlight illumination
    for(i = 0; i < num_spotlights; i++) {
        float full_spotlight_dist = spotlight_attenuation[i].x;
        float no_spotlight_dist = spotlight_attenuation[i].y;
        float attenuation_sk = 1.0 / (full_spotlight_dist * full_spotlight_dist);
        vec3 light_vector = spotlight_position[i] - world_position;
        float dist = length(light_vector);
        if (dist < no_spotlight_dist) {
            vec3 light_direction = normalize(light_vector);
            float angle = abs(acos(dot(-light_direction, normalize(spotlight_direction[i]))));
            if (angle <= spotlight_fov[i]) {
                float light_attenuation = 1.0 / max(attenuation_sk * dist * dist, 1.0);
                float edge_attenuation = min(10.0 * (spotlight_fov[i] - angle) / spotlight_fov[i], 1.0);
                light_attenuation *= edge_attenuation;

                //diffuse
                float n_dot_l = max(dot(world_normal, light_direction), 0.0);
                light_diffuse += spotlight_color[i] * light_attenuation * n_dot_l;
                //specular
                vec3 reflection_direction = normalize(reflect(-light_direction, world_normal));
                vec3 view_direction = normalize(camera_position - world_position);
                float r_dot_v = max(dot(reflection_direction, view_direction), 0.0);
                light_specular += spotlight_color[i] * light_attenuation * pow(r_dot_v, material_shininess);
            }
        }
    }

    light_diffuse = min(light_diffuse, 1.0);
    light_specular = min(light_specular, 1.0);
    
    vec4 tex_color = texture(image, world_texcoord);
    vec3 material_tex = ((1.0 - tex_color.a) * material_color) + (tex_color.a * tex_color.rgb);

    vec3 final_color = min((light_ambient * material_tex) + (light_diffuse * material_tex) +
                           (light_specular * material_specular), 1.0);

    FragColor = vec4(final_color, 1.0);
}