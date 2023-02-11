#version 450

layout(push_constant) uniform PushConstants {
    mat4 proj;
    vec3 color;
} constants;

layout(location = 0) in vec2 inPosition;

void main() {
    gl_Position = constants.proj * vec4(inPosition, 0.0, 1.0);
}
