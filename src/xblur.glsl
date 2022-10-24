#version 450

layout (local_size_x = 32, local_size_y = 32) in;

layout(set = 0, binding = 0) buffer ly_arena {
  vec4 arena[];
};

layout(set = 0, binding = 1) buffer ly_xblur {
  vec4 xblur[];
};

layout(push_constant) uniform PushConstants {
  int height;
  int width;
} constants;

#include "kernel.glsl"

void main() {
  ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
  if (coords.x >= constants.width || coords.y >= constants.height) {
    return;
  }
  int offset = coords.y * constants.width + coords.x;
  int loop_stride = constants.width - 1;
  vec4 big_output = vec4(0.0, 0.0, 0.0, 0.0);
  vec4 small_output = vec4(0.0, 0.0, 0.0, 0.0);
  int lowx = coords.x, highx = coords.x;
  int lowoff = offset, highoff = offset;
  for (int i = 0; i < num_steps; i++) {
    big_output += big_weights[i] * (arena[lowoff] + arena[highoff]);
    small_output += small_weights[i] * (arena[lowoff] + arena[highoff]);
    lowx -= 1;
    lowoff -= 1;
    highx += 1;
    highoff += 1;
    if (lowx < 0) {
      lowx = constants.width - 1;
      lowoff += loop_stride;
    }
    if (highx >= constants.width) {
      highx = 0;
      highoff -= loop_stride;
    }
  }

  xblur[2 * offset] = big_output;
  xblur[2 * offset + 1] = small_output;
}
