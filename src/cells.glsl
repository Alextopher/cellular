#version 450

layout (local_size_x = 32, local_size_y = 32) in;

layout(set = 0, binding = 0, rgba8) uniform image2D arena;

layout(set = 0, binding = 1) uniform sampler2D xblur;

#include "kernel.glsl"

vec4 blurred_sample() {
  vec4 blur_output = vec4(0.0, 0.0, 0.0, 0.0);
  ivec2 imgsz = imageSize(arena);
  float pixel_offset = 1.0 / float(imgsz.y);
  vec2 coords = (vec2(gl_GlobalInvocationID.xy) + vec2(0.5, 0.5)) / imgsz;
  for (int i = 0; i < num_steps; i++) {
    blur_output += weights[i] *
      (
       texture(xblur, coords + vec2(0.0, pixel_offset * offsets[i])) +
       texture(xblur, coords - vec2(0.0, pixel_offset * offsets[i]))
      );
  }
  return blur_output;
}

void main() {
  vec4 n = blurred_sample();
  vec4 self = imageLoad(arena, ivec2(gl_GlobalInvocationID.xy));
//  vec4 neg_self = (vec4(1.0, 1.0, 1.0, 1.0) - self);
//  vec4 clamp_scale = 0.5 + 4.0 * self * neg_self;
  float step = 0.9;
  vec4 live_growth = (n - 0.2) * (0.5 - n);
//  vec4 dead_growth = n - 0.2;
//  vec4 diff = dead_growth * neg_self + live_growth * self;
  imageStore(arena, ivec2(gl_GlobalInvocationID.xy), self + live_growth * step);
}

