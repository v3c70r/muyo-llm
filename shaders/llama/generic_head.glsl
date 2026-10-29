#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_descriptor_heap : require
#extension GL_EXT_nonuniform_qualifier : require

layout (push_constant) uniform parameter
{
    uint KX;
    uint KY;
    float param1;
    float param2;
    float param3;
    float param4;
    // Descriptor-heap indices in `llama_argmax_bindings` order (A, D). The graph is heap-only, so
    // the node resolves them and records them as push data.
    uint heap[2];
} p;
