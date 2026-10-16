#include "Swiglu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "GraphServices.h"
#include "LlamaContracts.h"
#include "RenderGraph/RenderGraphBuilder.h"
#include "RenderGraph/RenderGraphNodeContext.h"
#include "RenderGraph/RenderGraphNodeResource.h"
#include "RenderGraph/RenderGraphResourceDesc.h"
#include "RenderResource.h"
#include "RenderResourceManager.h"
#include "VkRenderDevice.h"
#include "vk_mem_alloc.h"

namespace muyo_llm
{
using Muyo::BufferResource;
using Muyo::GetRenderDevice;
using Muyo::GetRenderResourceManager;
using Muyo::RenderGraph::BufferResourceDesc;
using Muyo::RenderGraph::DescriptorBinding;
using Muyo::RenderGraph::QueueType;
using Muyo::RenderGraph::RenderGraphBuilder;
using Muyo::RenderGraph::RenderGraphNodeContext;
using Muyo::RenderGraph::RenderGraphNodeCreateInfo;
using Muyo::RenderGraph::ResourceHandle;
using Muyo::RenderGraph::ResourceIOType;
using Muyo::RenderGraph::ResourceKind;
using Muyo::RenderGraph::ResourceLifetime;
using Muyo::RenderGraph::ResourceUsage;
using Muyo::RenderGraph::ResourceUse;
using Muyo::RenderGraph::SpecializationValue;

namespace
{
/// Not a multiple of the shader's 512-thread coverage, so the `i >= p.N` guard is exercised.
constexpr uint32_t K_ELEMENT_COUNT = 600;

/// ggml's `init_fastdiv_values`: the multiplier and shift that let the shader divide by `d` with
/// `(mulhi(n, mp) + n) >> L`. Reproduced because the shader's index math depends on it.
void InitFastDiv(uint32_t d, uint32_t& mp, uint32_t& L)
{
    L = 0;
    while (L < 32 && (uint32_t{1} << L) < d)
    {
        ++L;
    }
    mp = static_cast<uint32_t>((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
}

float Silu(float value) { return value / (1.0F + std::exp(-value)); }
}  // namespace

int RunSwiglu()
{
    std::vector<float> hostGate(K_ELEMENT_COUNT);
    std::vector<float> hostUp(K_ELEMENT_COUNT);
    for (uint32_t index = 0; index < K_ELEMENT_COUNT; ++index)
    {
        hostGate[index] = 0.01F * static_cast<float>(static_cast<int>(index % 17) - 8);
        hostUp[index] = 0.5F + 0.01F * static_cast<float>(index % 7);
    }

    RenderGraphBuilder builder(GraphServices());

    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const auto buffer = [](uint32_t count, uint32_t stride, VkBufferUsageFlags usage)
    {
        return BufferResourceDesc{.count = count,
                                  .stride = stride,
                                  .usage = usage,
                                  .memoryProperties = VMA_MEMORY_USAGE_CPU_TO_GPU,
                                  .lifetime = ResourceLifetime::Persistent};
    };
    builder.AddResource("GluGate", buffer(K_ELEMENT_COUNT, sizeof(float), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    builder.AddResource("GluUp", buffer(K_ELEMENT_COUNT, sizeof(float), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    builder.AddResource("GluOutput", buffer(K_ELEMENT_COUNT, sizeof(float), storage));

    const auto binding = [](uint32_t index) { return DescriptorBinding{.set = 0, .binding = index}; };
    const auto readUse = [&binding](const char* handle, uint32_t index)
    {
        return ResourceUse{.handle = ResourceHandle(handle),
                           .io = ResourceIOType::READ,
                           .usage = ResourceUsage::STORAGE_BUFFER,
                           .kind = ResourceKind::BUFFER,
                           .descriptorBinding = binding(index)};
    };

    RenderGraphNodeCreateInfo gluPass = {
        .nodeName = "LlamaSwiglu",
        .queueType = QueueType::COMPUTE,
        .resourceUses =
            {
                readUse("GluGate", llama_glu_bindings::A),
                readUse("GluUp", llama_glu_bindings::B),
                ResourceUse{.handle = ResourceHandle("GluOutput"),
                            .io = ResourceIOType::READ_WRITE,
                            .usage = ResourceUsage::STORAGE_BUFFER,
                            .kind = ResourceKind::BUFFER,
                            .descriptorBinding = binding(llama_glu_bindings::D)},
            },
        .shaderNames = {"swiglu_f32.comp"},
        .execute = [](RenderGraphNodeContext& ctx)
        {
            // ggml's `ggml_vk_glu` for a split 1-D tensor: all three operands are [N] with element
            // stride 1 and outer strides N. The `ne2_*` fastdiv values divide the linearised dst
            // index back into (i20, i21, i22, i23); with ne21 = ne22 = 1 all three divide by N.
            LlamaGluPushConstants pushConstants{};
            pushConstants.N = K_ELEMENT_COUNT;
            pushConstants.ne00 = K_ELEMENT_COUNT;
            pushConstants.ne20 = K_ELEMENT_COUNT;
            pushConstants.mode = llama_glu_modes::SPLIT;
            pushConstants.alpha = 0.0F;
            pushConstants.limit = 0.0F;
            pushConstants.nb00 = 1;
            pushConstants.nb01 = K_ELEMENT_COUNT;
            pushConstants.nb02 = K_ELEMENT_COUNT;
            pushConstants.nb03 = K_ELEMENT_COUNT;
            pushConstants.nb10 = 1;
            pushConstants.nb11 = K_ELEMENT_COUNT;
            pushConstants.nb12 = K_ELEMENT_COUNT;
            pushConstants.nb13 = K_ELEMENT_COUNT;
            pushConstants.nb20 = 1;
            pushConstants.nb21 = K_ELEMENT_COUNT;
            pushConstants.nb22 = K_ELEMENT_COUNT;
            pushConstants.nb23 = K_ELEMENT_COUNT;
            pushConstants.ne21 = 1;
            pushConstants.ne22 = 1;
            pushConstants.misalignOffsets = 0;
            InitFastDiv(pushConstants.ne22 * pushConstants.ne21 * pushConstants.ne20, pushConstants.ne2_012mp,
                        pushConstants.ne2_012L);
            InitFastDiv(pushConstants.ne21 * pushConstants.ne20, pushConstants.ne2_01mp, pushConstants.ne2_01L);
            InitFastDiv(pushConstants.ne20, pushConstants.ne2_0mp, pushConstants.ne2_0L);
            vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(pushConstants), &pushConstants);

            // Local size is 512 and the shader linearises z*262144 + y*512 + x, so one workgroup
            // covers 512 elements in x.
            const uint32_t blocks = (K_ELEMENT_COUNT + 511) / 512;
            vkCmdDispatch(ctx.commandBuffer, 1, blocks, 1);

            VkMemoryBarrier2 toHostBarrier{};
            toHostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            toHostBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            toHostBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            toHostBarrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
            toHostBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
            VkDependencyInfo dependencyInfo{};
            dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dependencyInfo.memoryBarrierCount = 1;
            dependencyInfo.pMemoryBarriers = &toHostBarrier;
            vkCmdPipelineBarrier2(ctx.commandBuffer, &dependencyInfo);
        }};

    builder.AddNode(gluPass);
    builder.Build();

    std::printf("[graph] execution order:");
    for (const std::string& nodeName : builder.GetExecutionOrder())
    {
        std::printf(" %s", nodeName.c_str());
    }
    std::printf("\n");

    auto* pGate = GetRenderResourceManager()->GetResource<BufferResource>("GluGate");
    auto* pUp = GetRenderResourceManager()->GetResource<BufferResource>("GluUp");
    auto* pOutput = GetRenderResourceManager()->GetResource<BufferResource>("GluOutput");
    if (pGate == nullptr || pUp == nullptr || pOutput == nullptr)
    {
        std::fprintf(stderr, "[fail] the graph did not allocate the swiglu buffers\n");
        return 1;
    }
    pGate->SetData(hostGate.data(), hostGate.size() * sizeof(float));
    pUp->SetData(hostUp.data(), hostUp.size() * sizeof(float));

    builder.Execute();

    const auto* pResult = static_cast<const float*>(pOutput->Map());
    uint32_t nWrong = 0;
    for (uint32_t index = 0; index < K_ELEMENT_COUNT; ++index)
    {
        const float expected = Silu(hostGate[index]) * hostUp[index];
        if (std::fabs(pResult[index] - expected) > 1e-5F * (1.0F + std::fabs(expected)))
        {
            if (nWrong < 4)
            {
                std::fprintf(stderr, "[fail] D[%u] = %g, expected %g\n", index, pResult[index], expected);
            }
            ++nWrong;
        }
    }
    pOutput->Unmap();

    if (nWrong != 0)
    {
        std::fprintf(stderr, "[fail] %u of %u elements wrong\n", nWrong, K_ELEMENT_COUNT);
        return 1;
    }

    std::printf("[swiglu] OK: llama.cpp swiglu over %u elements matches the CPU reference\n", K_ELEMENT_COUNT);
    return 0;
}
}  // namespace muyo_llm
