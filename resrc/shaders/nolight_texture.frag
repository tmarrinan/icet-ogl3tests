#version 150

uniform sampler2D image;

in vec2 world_texcoord;

out vec4 FragColor;

void main() {
    FragColor = texture(image, world_texcoord);
}
