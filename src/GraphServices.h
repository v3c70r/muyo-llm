#pragma once

#include "DescriptorManager.h"
#include "RenderGraph/RenderGraphBuilder.h"
#include "RenderResourceManager.h"
#include "VkRenderDevice.h"

namespace muyo_llm
{
/// The services a RenderGraph is given by construction (#53): the host owns the device and the two
/// allocators, so they are passed in rather than looked up inside the graph.
///
/// This is the constructor Muyo's `docs/Consumers.md` means when it says the "injected, not global"
/// guarantee flips to `holds` for a consumer once it stops using the transitional
/// `RenderGraphBuilder(VkRenderDevice*)` shim. The three handles are still obtained from the
/// process-wide accessors here, because initialising them is the host's job until the device layer
/// is injected too; what changes is that the graph itself no longer reaches for a global.
inline Muyo::RenderGraph::RenderGraphServices GraphServices()
{
    return Muyo::RenderGraph::RenderGraphServices{.device = Muyo::GetRenderDevice(),
                                                  .resourceManager = Muyo::GetRenderResourceManager(),
                                                  .descriptorManager = Muyo::GetDescriptorManager()};
}
}  // namespace muyo_llm
