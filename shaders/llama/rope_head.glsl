#include "types.glsl"

#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_descriptor_heap : require
#extension GL_EXT_nonuniform_qualifier : require

#include "rope_params.glsl"

layout(local_size_x = 1, local_size_y = 256, local_size_z = 1) in;

layout (descriptor_heap) readonly buffer X {A_TYPE rope_data_a[];} xHeap[];
layout (descriptor_heap) readonly buffer Y {int rope_data_pos[];} yHeap[];
layout (descriptor_heap) readonly buffer Z {float rope_data_ff[];} zHeap[];
layout (descriptor_heap) writeonly buffer D {ROPE_D_TYPE rope_data_d[];} dHeap[];
layout (descriptor_heap) readonly buffer I {uvec2 rope_data_i[];} iHeap[];  // indices for set_rows


layout (push_constant) uniform parameter {
    rope_params pc;
    // Descriptor-heap indices in `llama_rope_bindings` order (X, Y, Z, D, I).
    uint heap[5];
};

#define rope_data_a xHeap[heap[0]].rope_data_a
#define rope_data_pos yHeap[heap[1]].rope_data_pos
#define rope_data_ff zHeap[heap[2]].rope_data_ff
#define rope_data_d dHeap[heap[3]].rope_data_d
#define rope_data_i iHeap[heap[4]].rope_data_i
