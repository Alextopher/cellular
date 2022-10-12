#version 450
#extension GL_EXT_shader_explicit_arithmetic_types : require

layout (push_constant) uniform PushConstants {
  uint32_t prng_seed;
} constants;

layout (local_size_x = 32, local_size_y = 32) in;

layout(set = 0, binding = 0, rgba8) uniform image2D img;

uint globalInvocationIdx() {
  return gl_GlobalInvocationID.x * imageSize(img).y + gl_GlobalInvocationID.y;
}

uint globalInvocationCount() {
  return imageSize(img).x * imageSize(img).y;
}

#include "common.glsl"

void main() {
  uint16_t cu8 = squares_rand(0);
  float r = float(cu8 & 0xff) / 255.0;
  float g = float((cu8 >> 8) & 0xff) / 255.0;
  cu8 = squares_rand(1);
  float b = float(cu8 & 0xff) / 255.0;
  vec4 color = vec4(r, g, b, 1.0);
  imageStore(img, ivec2(gl_GlobalInvocationID.xy), color);
}
