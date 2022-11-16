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

layout(set = 0, binding = 2) uniform sampler2D camera_image;

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

const float a = 0.8;
const float b = 0.8;
const float c = -0.8;
const mat4 cross_diff = mat4(
      a,    c,    b,    0,
      b,    a,    c,    0,
      c,    b,    a,    0,
      0,    0,    0,    0 
    );

void main() {
  ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
  if (coords.x >= constants.width || coords.y >= constants.height) {
    return;
  }
  int offset = coords.y * constants.width + coords.x;
  vec4 n = blurred_sample(coords, 2 * offset);
  vec4 self = arena[offset];

  // scale coordinates to [0, 1] for sampler
  vec2 sampler_coords = vec2(coords) / vec2(constants.width, constants.height);
  vec4 camera_value = texture(camera_image, sampler_coords);

  float step = 0.10;
  float blend_factor = 0.05;
  float fade_factor = 0.05;

  vec4 two_sigma = 2.0 * vec4(0.05, 0.02, 0.01, 1.0);
  vec4 mu = vec4(0.2, 0.3, 0.4, 0.0);
  vec4 diff = n - mu;
  vec4 factor = 2.0 * exp(- diff * diff / two_sigma) - 1.0;
  factor = cross_diff * factor;

  vec4 cell_diff = factor * step;
  vec4 cam_diff = (camera_value - step) * blend_factor;
  float cell_len = length(cell_diff);
  float cam_len = length(cam_diff);
//  if (cell_len > 1.0 * cam_len) {
//      cam_diff *= cell_len / cam_len * 1.0;
//  }
//  if (cam_len > 1.0 * cell_len) {
//      cell_diff *= cam_len / cell_len * 1.0;
//  }
  vec4 total_diff = cell_diff + cam_diff - fade_factor * self;

  vec4 result = clamp(self + total_diff, 0.0, 1.0);
  arena[offset] = result;
}

