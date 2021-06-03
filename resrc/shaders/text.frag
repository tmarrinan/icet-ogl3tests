#version 150

uniform vec3 font_color;
uniform sampler2D image;

in vec2 world_texcoord;

out vec4 FragColor;

void main() {
    float alpha = texture(image, world_texcoord).r;
    //FragColor = vec4(alpha, font_color.y, 0.0, alpha);
    FragColor = vec4(font_color, alpha);
}
