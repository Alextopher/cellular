#version 450
#extension GL_EXT_shader_explicit_arithmetic_types : require

layout (push_constant) uniform PushConstants {
  uint32_t prng_seed;
  uint32_t flags;
  float mult1;
  float mult2;
  int invocation_size_x;
  int invocation_size_y;
} constants;

layout (local_size_x = 32, local_size_y = 32) in;

layout(set = 0, binding = 0, rgba8) uniform image2D img;

uint globalInvocationIdx() {
  return gl_GlobalInvocationID.x * constants.invocation_size_y + gl_GlobalInvocationID.y;
}

uint globalInvocationCount() {
  return constants.invocation_size_x * constants.invocation_size_y;
}

#include "common.glsl"

float repulsion(vec3 a, vec3 b) {
  vec3 d = abs(a - b);
  return (d.x + d.y + d.z);
}

const float kernel1_width = 10.0;
const float kernel2_width = 3.0;
const float kernel1_scale = 1.0 / kernel1_width / kernel1_width;
const float kernel2_scale = 1.0 / kernel2_width / kernel2_width;
const int kernel_radius = 20;
const int kernel_reps = 2 * kernel_radius + 2;

float energy_diff(vec3 c0, vec3 c1, ivec2 w0, ivec2 w1, ivec2 ul, float mult1, float mult2) {
  float energy = 0.0;
  ivec2 imgsize = imageSize(img);
  for (int ii = 0; ii < kernel_reps; ii++) {
    for (int jj = 0; jj < kernel_reps; jj++) {
      int i = ii + ul.x - kernel_radius;
      int j = jj + ul.y - kernel_radius;
      if (i >= 0 && j >= 0 && i < imgsize.x && j < imgsize.y && ivec2(i, j) != w0 && ivec2(i, j) != w1) {
        vec3 c = imageLoad(img, ivec2(i, j)).xyz;
        float drepul = repulsion(c0, c) - repulsion(c1, c);
        ivec4 diffs = ivec4(w1, w0) - ivec4(i, j, i, j);
        ivec4 diffssq = diffs * diffs;
        vec2 sqdists = vec2(diffssq.x + diffssq.y, diffssq.z + diffssq.w);
        vec2 rmults1 = exp2(-sqdists * kernel1_scale);
        vec2 rmults2 = exp2(-sqdists * kernel2_scale);
        vec2 rmults = rmults1 * mult1 + rmults2 * mult2; 
        energy += drepul * (rmults.y - rmults.x);
      }
    }
  }
  return energy;
}

void maybe_swap(ivec2 upper_left, ivec2[2] swap_tiles, uint16_t rand) {
  vec4 c0 = imageLoad(img, swap_tiles[0]);
  vec4 c1 = imageLoad(img, swap_tiles[1]);
  float de = energy_diff(c0.xyz, c1.xyz, swap_tiles[0], swap_tiles[1], upper_left, constants.mult1 * kernel1_scale, constants.mult2 * kernel2_scale);
  float prob = 1.0 / (1.0 + exp2(de));
  if (rand <= uint16_t(prob * 65535.0)) {
    imageStore(img, swap_tiles[0], c1);
    imageStore(img, swap_tiles[1], c0);
  }
}

void main() {
  ivec2 imgsize = imageSize(img);
  ivec2 base_idx;
  if ((constants.flags & 1) == 0) {
    base_idx = ivec2(gl_GlobalInvocationID.xy * 2);
  } else {
    base_idx = ivec2(gl_GlobalInvocationID.xy * 2) + ivec2(1, 1);
  }
  // if our cell extends past the end of the boundary, do nothing
  if (base_idx.x + 1 >= imgsize.x || base_idx.y + 1 >= imgsize.y) {
    return;
  }
  uint16_t r = squares_rand(0);

  // pick which two tiles to consider swapping
  ivec2 swap_tiles[2];
  if ((r & 12) == 13) { // they're diagonal
    if ((r & 1) == 0) {
      swap_tiles[0] = base_idx;
      swap_tiles[1] = base_idx + ivec2(1, 1);
    } else {
      swap_tiles[0] = base_idx + ivec2(1, 0);
      swap_tiles[1] = base_idx + ivec2(0, 1);
    }
  } else if ((r & 2) == 0) {
    swap_tiles[0] = base_idx + ivec2(r & 1, 0);
    swap_tiles[1] = base_idx + ivec2(r & 1, 1);
  } else {
    swap_tiles[0] = base_idx + ivec2(0, r & 1);
    swap_tiles[1] = base_idx + ivec2(1, r & 1);
  }

  maybe_swap(base_idx, swap_tiles, squares_rand(0xffe | constants.flags));
}
