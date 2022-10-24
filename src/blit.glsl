#version 450

layout (local_size_x = 32, local_size_y = 32) in;

layout(push_constant) uniform PushConstants {
  int height;
  int width;
} constants;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D swapchain;

layout(set = 0, binding = 1) buffer ly_arena {
  vec4 arena[];
};

void main() {
  ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
  if (coords.x >= constants.width || coords.y >= constants.height) {
    return;
  }
  int offset = coords.y * constants.width + coords.x;
  vec4 c = arena[offset];
  imageStore(swapchain, coords, vec4(c.xxx, 1.0));
}

