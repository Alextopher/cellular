#version 450

layout (local_size_x = 32, local_size_y = 32) in;

layout(set = 0, binding = 0) uniform sampler2D arena;

layout(set = 0, binding = 1, rgba8) uniform writeonly image2D xblur;

#include "kernel.glsl"

void main() {
  vec4 blur_output = vec4(0.0, 0.0, 0.0, 0.0);
  ivec2 imgsz = imageSize(xblur);
  float pixel_offset = 1.0 / float(imgsz.x);
  vec2 coords = (vec2(gl_GlobalInvocationID.xy) + vec2(0.5, 0.5)) / imgsz;
  for (int i = 0; i < num_steps; i++) {
    blur_output += weights[i] *
      (
       texture(arena, coords + vec2(pixel_offset * offsets[i], 0.0)) +
       texture(arena, coords - vec2(pixel_offset * offsets[i], 0.0))
      );
  }
  imageStore(xblur, ivec2(gl_GlobalInvocationID.xy), blur_output);
}
