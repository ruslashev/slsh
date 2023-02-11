#version 450

layout(push_constant) uniform PushConstants {
    mat4 proj;
    vec3 color;
} constants;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(constants.color, 1.0);
}
