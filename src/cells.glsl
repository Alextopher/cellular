#version 450

layout (local_size_x = 32, local_size_y = 32) in;

layout(push_constant) uniform PushConstants {
  int height;
  int width;
} constants;

layout(set = 0, binding = 0) buffer ly_arena {
  vec4 arena[];
};

layout(set = 0, binding = 1) buffer ly_xblur {
  vec4 xblur[];
};

#include "kernel.glsl"

vec4 blurred_sample(ivec2 coords, int offset) {
  int column_stride = 2 * constants.width;
  int loop_stride = 2 * constants.width * (constants.height - 1);
  vec4 big_output = vec4(0.0, 0.0, 0.0, 0.0);
  vec4 small_output = vec4(0.0, 0.0, 0.0, 0.0);
  int lowy = coords.y, highy = coords.y;
  int lowoff = offset, highoff = offset;
  for (int i = 0; i < num_steps; i++) {
    big_output += big_weights[i] * (xblur[lowoff] + xblur[highoff]);
    small_output += small_weights[i] * (xblur[lowoff + 1] + xblur[highoff + 1]);
    lowoff -= column_stride;
    lowy -= 1;
    highoff += column_stride;
    highy += 1;
    if (highy >= constants.height) {
      highy = 0;
      highoff -= loop_stride;
    }
    if (lowy < 0) {
      lowy = constants.height - 1;
      lowoff += loop_stride;
    }
  }
  return big_output - small_output;
}

void main() {
  ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
  if (coords.x >= constants.width || coords.y >= constants.height) {
    return;
  }
  int offset = coords.y * constants.width + coords.x;
  vec4 n = blurred_sample(coords, 2 * offset);
  vec4 self = arena[offset];
  float step = 0.02;
  float two_sigma = 2.0 * 0.1;
  float mu = 0.3;
  vec4 diff = n - mu;
  vec4 factor = 2.0 * exp(- diff * diff / two_sigma) - 1.0;
  vec4 result = clamp(self + factor * step, 0.0, 1.0);
  arena[offset] = vec4(result.x, n.x, -n.x, 0.0);
}

