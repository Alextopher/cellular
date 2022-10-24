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

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D swapchain;

layout(set = 0, binding = 1) buffer ly_arena {
  vec4 arena[];
};

void main() {
  ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
  if (coords.x >= constants.width || coords.y >= constants.height) {
//    return;
  }
  int offset = coords.y * constants.width + coords.x;
  vec4 c = arena[offset];
  if (coords.x * 2 > constants.width) {
  if (((coords.y & 15) == 0) || ((coords.y & 15) != 1) && (((1 << (coords.y >> 4)) & bconstants.height) == 0)) {
    imageStore(swapchain, coords, vec4(0.0, c.yz, 1.0));
  } else {
    imageStore(swapchain, coords, vec4(1.0, c.yz, 1.0));
  }
  } else {
  if (((coords.y & 15) == 0) || ((coords.y & 15) != 1) && (((1 << (coords.y >> 4)) & bconstants.width) == 0)) {
    imageStore(swapchain, coords, vec4(0.0, c.yz, 1.0));
  } else {
    imageStore(swapchain, coords, vec4(1.0, c.yz, 1.0));
  }
  }
}

