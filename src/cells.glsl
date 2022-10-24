#version 450

layout (local_size_x = 32, local_size_y = 32) in;

layout(push_constant) uniform PushConstants {
  int height;
  int width;
} bconstants;

struct asdfasdf {
  int height;
  int width;
} constants = {
1015,
636
};

layout(set = 0, binding = 0) buffer ly_arena {
  vec4 arena[];
};

layout(set = 0, binding = 1) buffer ly_xblur {
  vec4 xblur[];
};

#include "kernel.glsl"

vec4 blurred_sample(ivec2 coords, int offset) {
  int column_stride = constants.width;
  int loop_stride = constants.width * (constants.height - 1);
  vec4 blur_output = vec4(0.0, 0.0, 0.0, 0.0);
  int lowy = coords.y, highy = coords.y;
  int lowoff = offset, highoff = offset;
  for (int i = 0; i < num_steps; i++) {
    blur_output += weights[i] * (xblur[lowoff] + xblur[highoff]);
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
  return blur_output;
}

void main() {
  ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
  if (coords.x >= constants.width || coords.y >= constants.height) {
//    return;
  }
  int offset = coords.y * constants.width + coords.x;
  vec4 n = blurred_sample(coords, offset);
//  vec4 self = arena[offset];
////  vec4 neg_self = (vec4(1.0, 1.0, 1.0, 1.0) - self);
////  vec4 clamp_scale = 0.5 + 4.0 * self * neg_self;
//  float step = 0.9;
//  vec4 live_growth = (n - 0.2) * (0.5 - n);
////  vec4 dead_growth = n - 0.2;
////  vec4 diff = dead_growth * neg_self + live_growth * self;
//  arena[offset] = self + live_growth * step;
  arena[offset] = vec4(offset / 20.0 / 636.0, 1.0, 1.0, 1.0);
}

