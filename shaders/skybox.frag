#version 450

layout(push_constant) uniform PushConstants {
    vec2 res;
    vec2 view_angles;
} consts;

layout(location = 0) out vec4 outColor;

// https://www.shadertoy.com/view/NlXXWN

vec3 hash33(vec3 p3) {
    p3 = fract(p3 * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

float starField(vec3 rd) {
    rd *= 150.0;

    float col = 0.0;

    for (int i = 0; i < 4; i++) {
        vec3 cellUVs = floor(rd + float(i * 1199));
        vec3 hash = (hash33(cellUVs) * 2.0 - 1.0) * 0.8;
        float hashMagnitude = 1.0 - length(hash);
        vec3 UVgrid = fract(rd) - 0.5;
        float radius = clamp(hashMagnitude - 0.5, 0.0, 1.0);
        float radialGradient = length(UVgrid - hash) / radius;
        radialGradient = clamp(1.0 - radialGradient, 0.0, 1.0);
        radialGradient *= radialGradient;
        col += radialGradient;
    }

    return col;
}

void main() {
    // vec3 uv = vec3(1.4 * (gl_FragCoord.xy * 2.0 - consts.res) / consts.res.y, 2.0);
    vec3 uv = vec3(1.4 * (gl_FragCoord.xy * 2.0 - consts.res) / consts.res.y, 2.0);
    vec2 m = -consts.view_angles.yx;
    uv.yz *= mat2(cos(m.y), -sin(m.y), sin(m.y), cos(m.y));
    uv.xz *= mat2(cos(m.x), -sin(m.x), sin(m.x), cos(m.x));
    outColor = vec4(vec3(starField(normalize(uv))), 1.0);
}

// https://stackoverflow.com/a/1569893
mat3 EulerAnglesToMatrix(vec3 ang) {
    float Sx = sin(ang.x);
    float Sy = sin(ang.y);
    float Sz = sin(ang.z);
    float Cx = cos(ang.x);
    float Cy = cos(ang.y);
    float Cz = cos(ang.z);

    mat3 m;

    m[0][0] =  Cy * Cz;
    m[0][1] = -Cy * Sz;
    m[0][2] =  Sy;
    m[1][0] =  Cz * Sx * Sy + Cx * Sz;
    m[1][1] =  Cx * Cz - Sx * Sy * Sz;
    m[1][2] = -Cy * Sx;
    m[2][0] = -Cx * Cz * Sy + Sx * Sz;
    m[2][1] =  Cz * Sx + Cx * Sy * Sz;
    m[2][2] =  Cx * Cy;

    return m;
}

// https://www.shadertoy.com/view/MtV3Dd
// void main() {
//     // vec3 dir = anglesToDir4(consts.view_angles.x, consts.view_angles.y);
//     // vec3 up = vec3(0., 1., 0.);
//     // vec3 cameraForward = normalize(dir);
//     // vec3 cameraRight = normalize(cross(cameraForward, up));
//     // vec3 cameraUp = cross(cameraForward, cameraRight);
//     // mat3 cameraOrientation = mat3(cameraRight, cameraUp, cameraForward);

//     vec3 camAng = vec3(-consts.view_angles.x, consts.view_angles.y, 0.0);
//     mat3 cameraOrientation = EulerAnglesToMatrix(camAng);

//     // vec3 uv = vec3(((gl_FragCoord.xy / consts.res.y) * 2.0 - 1.0), 1.0);
//     vec2 uv = -1.0 + 2.0 * gl_FragCoord.xy / consts.res.xy;
//     uv.x *= consts.res.x / consts.res.y;

// #define PI 3.14159265359
//     // float fov = 0.7;
//     // float aperture = fov * 2.0*PI;
//     float aperture = 0.05 * PI * 2.0;
//     float f = 1.0/aperture;
//     float r = length(uv);
//     float phi = atan(uv.y, uv.x);
//     float theta;

//     theta = atan(r/f);

//     // theta = atan(r/(2.0*f))*2.0;

//     vec3 rd = cameraOrientation * vec3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));

//     outColor = vec4(vec3(starField(rd)), 1.0);
// }
