#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_descriptor_heap : require
#extension GL_EXT_nonuniform_qualifier : require

#include "utils.glsl"


layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

layout (descriptor_heap) readonly buffer A {A_TYPE data_a[];} aHeap[];
layout (descriptor_heap) readonly buffer B {A_TYPE data_b[];} bHeap[];
layout (descriptor_heap) writeonly buffer D {D_TYPE data_d[];} dHeap[];

layout (push_constant) uniform parameter
{
    uint N;
    uint ne00;
    uint ne20;
    uint mode;
    float alpha;
    float limit;
    uint nb00;
    uint nb01;
    uint nb02;
    uint nb03;
    uint nb10;
    uint nb11;
    uint nb12;
    uint nb13;
    uint nb20;
    uint nb21;
    uint nb22;
    uint nb23;
    uint ne21;
    uint ne22;
    uint misalign_offsets;
    uint ne2_012mp; uint ne2_012L;
    uint ne2_01mp;  uint ne2_01L;
    uint ne2_0mp;   uint ne2_0L;
    // Descriptor-heap indices in `llama_glu_bindings` order (A, B, D).
    uint heap[3];
} p;

#define data_a aHeap[p.heap[0]].data_a
#define data_b bHeap[p.heap[1]].data_b
#define data_d dHeap[p.heap[2]].data_d

uint get_aoffset() { return p.misalign_offsets >> 16; }
uint get_boffset() { return (p.misalign_offsets >> 8) & 0xFF; }
uint get_doffset() { return p.misalign_offsets & 0xFF; }

