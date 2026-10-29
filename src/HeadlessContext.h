#pragma once

#include <vector>

#include "DescriptorHeapService.h"
#include "DescriptorManager.h"
#include "GraphServices.h"
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

        // No feature structs are passed: the two 16-bit-storage flags the F16 kernels need are enabled
        // by `CreateDevice` itself since Muyo #78 (F8), and the promoted Vulkan 1.1/1.2 structs are
        // already in its chain, so passing another is a duplicate.
        // The graph path is descriptor-heap-only since Muyo B1b. Requesting both extensions here makes
        // a device without them fail at creation naming the extension, rather than at the first graph
        // node once the heap service is asked to initialise. `CreateDevice` still verifies the
        // `descriptorHeap` and `shaderUntypedPointers` features before enabling them.
        Muyo::GetRenderDevice()->CreateDevice(
            {VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME}, {}, nullptr, {});
        Muyo::GetRenderDevice()->CreateCommandPools();
        Muyo::GetMemoryAllocator()->Initalize(Muyo::GetRenderDevice());
        Muyo::GetRenderResourceManager()->Initialize();
        Muyo::GetDescriptorManager()->CreateDescriptorPool();
        Muyo::GetDescriptorManager()->CreateDescriptorSetLayouts();
        Muyo::GetSamplerManager()->createSamplers();

        // The graph's descriptors now live in one process-wide heap the host injects. Initialise it
        // after the device and before any graph is built; it must outlive every builder.
        //
        // The defaults (256 static / 32 ring slots) are sized for a frame-sized graph, not for a
        // whole-model decode graph: qwen2.5-0.5B has 371 distinct static descriptors (291 weights,
        // 48 KV-cache buffers, 24 score masks, 8 I/O/dummy buffers) and about 820 per-execution
        // descriptor uses per token, so Build() exhausts the default and Execute() would exhaust the
        // ring. Size both with headroom for larger models; F19 records the upstream error/default
        // problem. inFlightCount matches the builder's default of one execution at a time.
        m_descriptorHeap.Initialize(Muyo::GetRenderDevice(), Muyo::DescriptorHeapService::Config{.imageSlots = 64,
                                                                                                 .bufferSlots = 2048,
                                                                                                 .accelerationSlots = 8,
                                                                                                 .samplerSlots = 16,
                                                                                                 .ringSlots = 2048,
                                                                                                 .inFlightCount = 1});
        DescriptorHeapServiceHandle() = &m_descriptorHeap;
    }

    /// Tear every subsystem down in the inverse of the construction order.
    ~HeadlessContext()
    {
        DescriptorHeapServiceHandle() = nullptr;
        m_descriptorHeap.Shutdown();
        Muyo::GetSamplerManager()->destroySamplers();
        Muyo::GetDescriptorManager()->DestroyDescriptorSetLayouts();
        Muyo::GetDescriptorManager()->DestroyDescriptorPool();
        Muyo::GetRenderResourceManager()->Unintialize();
        Muyo::GetRenderDevice()->DestroyCommandPools();
        Muyo::GetMemoryAllocator()->Unintialize();
        Muyo::GetRenderDevice()->DestroyDevice();
        Muyo::GetRenderDevice()->Unintialize();
    }

    /// @return The heap every graph built under this context shares.
    Muyo::DescriptorHeapService& GetDescriptorHeap() { return m_descriptorHeap; }

    /// Owns process-wide subsystems; copying would tear them down twice.
    HeadlessContext(const HeadlessContext&) = delete;
    /// Owns process-wide subsystems; copying would tear them down twice.
    HeadlessContext& operator=(const HeadlessContext&) = delete;

private:
    Muyo::DescriptorHeapService m_descriptorHeap;
};
}  // namespace muyo_llm
