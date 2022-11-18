#version 450

layout (local_size_x = 32, local_size_y = 32) in;

layout(push_constant) uniform PushConstants {
  int height;
  int width;
} constants;

layout(set = 0, binding = 0) buffer parameters {
  float mat_a;
  float mat_b;
  float mat_c;
  float step_factor;
  float fade_factor;
  float stdevs_r;
  float stdevs_g;
  float stdevs_b;
  float small_stdev;
  float big_stdev;
} params;

layout(set = 0, binding = 1) buffer ly_arena {
  vec4 arena[];
};

layout(set = 0, binding = 2) buffer ly_xblur {
  vec4 xblur[];
};

layout(set = 0, binding = 3) uniform sampler2D camera_image;

#include "kernel.glsl"

vec4 blurred_sample(ivec2 coords, int offset) {
  int column_stride = 2 * constants.width;
  int loop_stride = 2 * constants.width * (constants.height - 1);
  vec4 big_output = vec4(0.0, 0.0, 0.0, 0.0);
  vec4 small_output = vec4(0.0, 0.0, 0.0, 0.0);
  int lowy = coords.y, highy = coords.y;
  int lowoff = offset, highoff = offset;
  for (int i = 0; i < num_steps; i++) {
    big_output += kernel_weight(params.big_stdev, i) * (xblur[lowoff] + xblur[highoff]);
    small_output += kernel_weight(params.small_stdev, i) * (xblur[lowoff + 1] + xblur[highoff + 1]);
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
  float integral = big_stdev * big_stdev - small_stdev * small_stdev;
  return (big_output - small_output) / integral;
}

void main() {
  const float a = params.mat_a;
  const float b = params.mat_b;
  const float c = params.mat_c;

  const mat4 cross_diff = mat4(
      a,    c,    b,    0,
      b,    a,    c,    0,
      c,    b,    a,    0,
      0,    0,    0,    0 
      );
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

  float step = params.step_factor;
  float fade_factor = params.fade_factor;

  vec4 two_sigma = 2.0 * vec4(params.stdevs_r, params.stdevs_g, params.stdevs_b, 1.0);
  vec4 mu = camera_value;
  vec4 diff = n - mu;
  vec4 factor = 2.0 * exp(- diff * diff / two_sigma) - 1.0;
  factor = cross_diff * factor;

  vec4 cell_diff = factor * step;
  vec4 total_diff = cell_diff - fade_factor * self;

  vec4 result = clamp(self + total_diff, 0.0, 1.0);
  arena[offset] = result;
}

