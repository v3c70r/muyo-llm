#include "Argmax.h"

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
/// One row of logits, exactly the shader's reduction width.
constexpr uint32_t K_LOGIT_COUNT = 32;
constexpr uint32_t K_BLOCK_SIZE = 32;

/// The token the logits are built to favour, off-centre so a wrong reduction width does not
/// accidentally land on it.
constexpr uint32_t K_EXPECTED_TOKEN = 21;
}  // namespace

int RunArgmax()
{
    std::vector<float> hostLogits(K_LOGIT_COUNT);
    for (uint32_t index = 0; index < K_LOGIT_COUNT; ++index)
    {
        hostLogits[index] = -1.0F + 0.001F * static_cast<float>(index);
    }
    hostLogits[K_EXPECTED_TOKEN] = 5.0F;

    RenderGraphBuilder builder(GraphServices());

    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const auto buffer = [](uint32_t count, uint32_t stride, VkBufferUsageFlags usage)
    {
        return BufferResourceDesc{.count = count,
                                  .stride = stride,
                                  .usage = usage,
                                  .memoryProperties = VMA_MEMORY_USAGE_CPU_TO_GPU,
                                  .lifetime = ResourceLifetime::Transient};
    };
    builder.AddResource("Logits", buffer(K_LOGIT_COUNT, sizeof(float), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    builder.AddResource("TokenIndex", buffer(1, sizeof(int32_t), storage));

    const auto binding = [](uint32_t index) { return DescriptorBinding{.set = 0, .binding = index}; };

    RenderGraphNodeCreateInfo argmaxPass = {
        .nodeName = "LlamaArgmax",
        .queueType = QueueType::COMPUTE,
        .resourceUses =
            {
                ResourceUse{.handle = ResourceHandle("Logits"),
                            .io = ResourceIOType::READ,
                            .usage = ResourceUsage::STORAGE_BUFFER,
                            .kind = ResourceKind::BUFFER,
                            .descriptorBinding = binding(llama_argmax_bindings::A)},
                ResourceUse{.handle = ResourceHandle("TokenIndex"),
                            .io = ResourceIOType::READ_WRITE,
                            .usage = ResourceUsage::STORAGE_BUFFER,
                            .kind = ResourceKind::BUFFER,
                            .descriptorBinding = binding(llama_argmax_bindings::D)},
            },
        .shaderNames = {"argmax_f32.comp"},
        .specializationConstants = {{.id = llama_argmax_spec_constants::BLOCK_SIZE, .value = K_BLOCK_SIZE}},
        .execute = [](RenderGraphNodeContext& ctx)
        {
            // `generic_head.glsl`'s block: KX is the row length, KY the row count.
            LlamaGenericHeadPushConstants pushConstants{};
            pushConstants.KX = K_LOGIT_COUNT;
            pushConstants.KY = 1;
            pushConstants.param1 = 0.0F;
            pushConstants.param2 = 0.0F;
            pushConstants.param3 = 0.0F;
            pushConstants.param4 = 0.0F;
            vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(pushConstants), &pushConstants);

            // One workgroup per row.
            vkCmdDispatch(ctx.commandBuffer, 1, 1, 1);

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

    builder.AddNode(argmaxPass);
    builder.Build();

    std::printf("[graph] execution order:");
    for (const std::string& nodeName : builder.GetExecutionOrder())
    {
        std::printf(" %s", nodeName.c_str());
    }
    std::printf("\n");

    auto* pLogits = GetRenderResourceManager()->GetResource<BufferResource>("Logits");
    auto* pIndex = GetRenderResourceManager()->GetResource<BufferResource>("TokenIndex");
    if (pLogits == nullptr || pIndex == nullptr)
    {
        std::fprintf(stderr, "[fail] the graph did not allocate the argmax buffers\n");
        return 1;
    }
    pLogits->SetData(hostLogits.data(), hostLogits.size() * sizeof(float));

    builder.Execute();

    const auto* pResult = static_cast<const int32_t*>(pIndex->Map());
    const int32_t result = pResult[0];
    pIndex->Unmap();

    if (result != static_cast<int32_t>(K_EXPECTED_TOKEN))
    {
        std::fprintf(stderr, "[fail] argmax = %d, expected %u\n", result, K_EXPECTED_TOKEN);
        return 1;
    }

    std::printf("[argmax] OK: llama.cpp argmax picked token %d from %u logits\n", result, K_LOGIT_COUNT);
    return 0;
}
}  // namespace muyo_llm
