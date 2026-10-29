#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

#include "RenderGraph/RenderGraphNodeContext.h"

namespace muyo_llm
{
/// Resolve one resource use to its descriptor-heap index for the executing node.
///
/// Muyo's graph path is heap-only since B1b: there is no descriptor set or binding to declare, the
/// graph writes each use's descriptor at `Build()` (or rings it per execution for an imported
/// resource), and the node passes the index to its shader as push data. Every node that binds a
/// buffer goes through this helper so a missing index is a named error at record time rather than a
/// zero index reading someone else's descriptor.
/// @param ctx Context of the node whose callback is recording.
/// @param handle Resource handle exactly as the node declared it in `resourceUses`.
/// @param usage The same usage the node declared for that handle.
/// @return The index to put in the shader's `heap[]` push-data entry.
/// @throws std::runtime_error when the node did not register that (handle, usage) pair.
inline uint32_t ResolveHeapIndex(Muyo::RenderGraph::RenderGraphNodeContext& ctx, const char* handle,
                                 Muyo::RenderGraph::ResourceUsage usage)
{
    const std::optional<uint32_t> index = ctx.GetHeapIndex(handle, usage);
    if (!index.has_value())
    {
        throw std::runtime_error(std::string("no descriptor-heap index for '") + handle + "'");
    }
    return *index;
}

/// Fill one push-data `heap[]` entry from a node's resource use.
/// @param heap The node's push-data index table, laid out in the shader's binding order.
/// @param slot Index into that table; the `llama_*_bindings` names give it a readable one.
/// @param ctx See ResolveHeapIndex.
/// @param handle See ResolveHeapIndex.
/// @param usage See ResolveHeapIndex.
inline void SetHeapIndex(uint32_t* heap, uint32_t slot, Muyo::RenderGraph::RenderGraphNodeContext& ctx,
                         const char* handle, Muyo::RenderGraph::ResourceUsage usage)
{
    heap[slot] = ResolveHeapIndex(ctx, handle, usage);
}

/// Record a node's whole push-data block at offset 0.
///
/// A heap pipeline has no `VkPipelineLayout`, so `vkCmdPushConstants` is not available; the graph
/// forwards these bytes with `vkCmdPushDataEXT`, and the shader reads them through its
/// `push_constant` block. The block is the caller's ABI, which is why it is a struct and not an API.
/// @param ctx Context of the node whose callback is recording.
/// @param data The push-data block, exactly as the shader declares it.
template <class T>
void PushData(Muyo::RenderGraph::RenderGraphNodeContext& ctx, const T& data)
{
    ctx.PushData(0, std::as_bytes(std::span{&data, 1}));
}
}  // namespace muyo_llm
