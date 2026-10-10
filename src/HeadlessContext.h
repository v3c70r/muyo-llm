#pragma once

#include "DescriptorManager.h"
#include "RenderResourceManager.h"
#include "SamplerManager.h"
#include "VkMemoryAllocator.h"
#include "VkRenderDevice.h"

namespace muyo_llm
{
/// Owns the Muyo subsystems a windowless, compute-only consumer needs.
///
/// Modelled on Muyo's `GraphicsTestEnv`, minus everything that assumes a window or a
/// test framework: no surface, no swapchain, no ImGui, no swapchain images. The order
/// below is the one `GraphicsTestEnv` validates and the destructor reverses it, because
/// the resource managers allocate against the device and must outlive it.
///
/// The engine's subsystems are process-wide singletons reached by free function
/// (`GetRenderDevice()` and friends). Their being globals rather than injected services is
/// the gap Muyo #53 owns; this class is the smallest thing that works until then.
///
/// Construction requires a GPU. It does not require a window.
class HeadlessContext
{
public:
    /// Initialise the device, allocator, resource, descriptor and sampler managers.
    HeadlessContext()
    {
        Muyo::GetRenderDevice()->Initialize({}, {});
        Muyo::GetRenderDevice()->CreateDevice({}, {}, nullptr, {});
        Muyo::GetRenderDevice()->CreateCommandPools();
        Muyo::GetMemoryAllocator()->Initalize(Muyo::GetRenderDevice());
        Muyo::GetRenderResourceManager()->Initialize();
        Muyo::GetDescriptorManager()->CreateDescriptorPool();
        Muyo::GetDescriptorManager()->CreateDescriptorSetLayouts();
        Muyo::GetSamplerManager()->createSamplers();
    }

    /// Tear every subsystem down in the inverse of the construction order.
    ~HeadlessContext()
    {
        Muyo::GetSamplerManager()->destroySamplers();
        Muyo::GetDescriptorManager()->DestroyDescriptorSetLayouts();
        Muyo::GetDescriptorManager()->DestroyDescriptorPool();
        Muyo::GetRenderResourceManager()->Unintialize();
        Muyo::GetRenderDevice()->DestroyCommandPools();
        Muyo::GetMemoryAllocator()->Unintialize();
        Muyo::GetRenderDevice()->DestroyDevice();
        Muyo::GetRenderDevice()->Unintialize();
    }

    /// Owns process-wide subsystems; copying would tear them down twice.
    HeadlessContext(const HeadlessContext&) = delete;
    /// Owns process-wide subsystems; copying would tear them down twice.
    HeadlessContext& operator=(const HeadlessContext&) = delete;
};
}  // namespace muyo_llm
