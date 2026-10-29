#pragma once

#include "DescriptorHeapService.h"
#include "RenderGraph/RenderGraphBuilder.h"
#include "RenderResourceManager.h"
#include "VkRenderDevice.h"

namespace muyo_llm
{
/// The heap every graph built in this process shares.
///
/// Muyo's graph path is descriptor-heap-only since B1b: `RenderGraphServices::descriptorHeap` is
/// required and the descriptor-set allocator is no longer a graph service. The service is device-scoped
/// and must outlive every builder, so `HeadlessContext` owns it and publishes the address here rather
/// than making each call site reach for a global of its own. It is the same shape as the device and
/// resource-manager accessors below, and it goes away when Muyo #53 injects the device layer.
/// @return The current context's heap service, or nullptr before `HeadlessContext` is constructed.
inline Muyo::DescriptorHeapService*& DescriptorHeapServiceHandle()
{
    static Muyo::DescriptorHeapService* pHeap = nullptr;
    return pHeap;
}

/// The services a RenderGraph is given by construction (#53): the host owns the device, the resource
/// allocator and the descriptor heap, so they are passed in rather than looked up inside the graph.
///
/// This is the constructor Muyo's `docs/Consumers.md` means when it says the "injected, not global"
/// guarantee flips to `holds` for a consumer once it stops using the transitional
/// `RenderGraphBuilder(VkRenderDevice*)` shim. The handles are still obtained from the process-wide
/// accessors here, because initialising them is the host's job until the device layer is injected too;
/// what changes is that the graph itself no longer reaches for a global.
inline Muyo::RenderGraph::RenderGraphServices GraphServices()
{
    return Muyo::RenderGraph::RenderGraphServices{.device = Muyo::GetRenderDevice(),
                                                  .resourceManager = Muyo::GetRenderResourceManager(),
                                                  .descriptorHeap = DescriptorHeapServiceHandle()};
}
}  // namespace muyo_llm
