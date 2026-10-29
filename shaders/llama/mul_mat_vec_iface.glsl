#include "types.glsl"

#extension GL_EXT_descriptor_heap : require
#extension GL_EXT_nonuniform_qualifier : require

#define MAT_VEC_FUSION_FLAGS_BIAS0 0x1
#define MAT_VEC_FUSION_FLAGS_BIAS1 0x2
#define MAT_VEC_FUSION_FLAGS_SCALE0 0x4
#define MAT_VEC_FUSION_FLAGS_SCALE1 0x8

layout (descriptor_heap) readonly buffer A {A_TYPE data_a[];} aHeap[];
#if defined(A_TYPEV4)
layout (descriptor_heap) readonly buffer AV4 {A_TYPEV4 data_a_v4[];} aV4Heap[];
#endif
#if defined(A_TYPE_PACKED16)
layout (descriptor_heap) readonly buffer A_PACKED16 {A_TYPE_PACKED16 data_a_packed16[];} aPacked16Heap[];
#endif
#if defined(A_TYPE_PACKED32)
layout (descriptor_heap) readonly buffer A_PACKED32 {A_TYPE_PACKED32 data_a_packed32[];} aPacked32Heap[];
#endif
#if defined(A_TYPE_PACKED64)
layout (descriptor_heap) readonly buffer A_PACKED64 {A_TYPE_PACKED64 data_a_packed64[];} aPacked64Heap[];
#endif

layout (descriptor_heap) readonly buffer B {B_TYPE data_b[];} bHeap[];
#ifdef B_TYPEV2
layout (descriptor_heap) readonly buffer BV2 {B_TYPEV2 data_b_v2[];} bV2Heap[];
#endif
#ifdef B_TYPEV4
layout (descriptor_heap) readonly buffer BV4 {B_TYPEV4 data_b_v4[];} bV4Heap[];
#endif

layout (descriptor_heap) writeonly buffer D {D_TYPE data_d[];} dHeap[];

layout (descriptor_heap) readonly buffer Fuse0 {D_TYPE data_fuse0[];} fuse0Heap[];
layout (descriptor_heap) readonly buffer Fuse1 {D_TYPE data_fuse1[];} fuse1Heap[];

#ifdef MUL_MAT_ID
layout (descriptor_heap) readonly buffer IDS {int data_ids[];} idsHeap[];
#endif

// Descriptor-heap indices in `llama_mat_vec_bindings` order (A, B, D, FUSE0, FUSE1).
#define data_a aHeap[p.heap[0]].data_a
#define data_b bHeap[p.heap[1]].data_b
#define data_d dHeap[p.heap[2]].data_d
#define data_fuse0 fuse0Heap[p.heap[3]].data_fuse0
#define data_fuse1 fuse1Heap[p.heap[4]].data_fuse1
#if defined(A_TYPEV4)
#define data_a_v4 aV4Heap[p.heap[0]].data_a_v4
#endif
#if defined(A_TYPE_PACKED16)
#define data_a_packed16 aPacked16Heap[p.heap[0]].data_a_packed16
#endif
#if defined(A_TYPE_PACKED32)
#define data_a_packed32 aPacked32Heap[p.heap[0]].data_a_packed32
#endif
#if defined(A_TYPE_PACKED64)
#define data_a_packed64 aPacked64Heap[p.heap[0]].data_a_packed64
#endif
#ifdef B_TYPEV2
#define data_b_v2 bV2Heap[p.heap[1]].data_b_v2
#endif
#ifdef B_TYPEV4
#define data_b_v4 bV4Heap[p.heap[1]].data_b_v4
#endif
#ifdef MUL_MAT_ID
#define data_ids idsHeap[p.heap[5]].data_ids
#endif
