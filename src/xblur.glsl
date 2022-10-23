#version 450

layout (local_size_x = 32, local_size_y = 32) in;

layout(set = 0, binding = 0, rgba8) uniform image2D arena;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D xblur;
