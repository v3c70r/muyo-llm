#include "SoftMax.h"

#include <algorithm>
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
constexpr uint32_t K_COLUMNS = 64;
constexpr uint32_t K_BLOCK_SIZE = 32;

/// The attention scale `1/sqrt(headDim)`.
constexpr float K_SCALE = 1.0F / 8.0F;
}  // namespace

int RunSoftMax()
{
    std::vector<float> hostInput(K_COLUMNS);
    for (uint32_t column = 0; column < K_COLUMNS; ++column)
    {
        hostInput[column] = static_cast<float>(static_cast<int>(column % 9) - 4);
    }

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
    builder.AddResource("Scores", buffer(K_COLUMNS, sizeof(float), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    // Bindings 1 and 2 are declared unconditionally; they are read only for a mask (KY > 0) or
    // attention sinks (hasSinks != 0), neither of which is used here.
    builder.AddResource("SoftMaxMask", buffer(K_COLUMNS, sizeof(float), storage));
    builder.AddResource("SoftMaxSinks", buffer(1, sizeof(float), storage));
    builder.AddResource("SoftMaxOutput", buffer(K_COLUMNS, sizeof(float), storage));

    const auto binding = [](uint32_t index) { return DescriptorBinding{.set = 0, .binding = index}; };
    const auto readUse = [&binding](const char* handle, uint32_t index)
    {
        return ResourceUse{.handle = ResourceHandle(handle),
                           .io = ResourceIOType::READ,
                           .usage = ResourceUsage::STORAGE_BUFFER,
                           .kind = ResourceKind::BUFFER,
                           .descriptorBinding = binding(index)};
    };

    RenderGraphNodeCreateInfo softmaxPass = {
        .nodeName = "LlamaSoftMax",
        .queueType = QueueType::COMPUTE,
        .resourceUses =
            {
                readUse("Scores", llama_soft_max_bindings::X),
                readUse("SoftMaxMask", llama_soft_max_bindings::Y),
                readUse("SoftMaxSinks", llama_soft_max_bindings::Z),
                // The shader scales the tail columns in place (`*=`), so D is read-write.
                ResourceUse{.handle = ResourceHandle("SoftMaxOutput"),
                            .io = ResourceIOType::READ_WRITE,
                            .usage = ResourceUsage::STORAGE_BUFFER,
                            .kind = ResourceKind::BUFFER,
                            .descriptorBinding = binding(llama_soft_max_bindings::D)},
            },
        .shaderNames = {"soft_max_f32.comp"},
        .specializationConstants = {{.id = llama_soft_max_spec_constants::BLOCK_SIZE, .value = K_BLOCK_SIZE}},
        .execute = [](RenderGraphNodeContext& ctx)
        {
            // ggml's `ggml_vk_soft_max` with no mask and no sinks: KY = 0, the mask strides zero,
            // hasSinks = 0, and one row.
            LlamaSoftMaxPushConstants pushConstants{};
            pushConstants.KX = K_COLUMNS;
            pushConstants.KY = 0;
            pushConstants.ne00 = K_COLUMNS;
            pushConstants.ne01 = 1;
            pushConstants.ne02 = 1;
            pushConstants.ne12 = 0;
            pushConstants.ne13 = 0;
            pushConstants.nb11 = 0;
            pushConstants.nb12 = 0;
            pushConstants.nb13 = 0;
            pushConstants.scale = K_SCALE;
            pushConstants.maxBias = 0.0F;
            pushConstants.m0 = 0.0F;
            pushConstants.m1 = 0.0F;
            pushConstants.nHeadLog2 = 0;
            pushConstants.nrowsX = 1;
            pushConstants.hasSinks = 0;
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

    builder.AddNode(softmaxPass);
    builder.Build();

    std::printf("[graph] execution order:");
    for (const std::string& nodeName : builder.GetExecutionOrder())
    {
        std::printf(" %s", nodeName.c_str());
    }
    std::printf("\n");

    auto* pScores = GetRenderResourceManager()->GetResource<BufferResource>("Scores");
    auto* pOutput = GetRenderResourceManager()->GetResource<BufferResource>("SoftMaxOutput");
    if (pScores == nullptr || pOutput == nullptr)
    {
        std::fprintf(stderr, "[fail] the graph did not allocate the soft_max buffers\n");
        return 1;
    }
    pScores->SetData(hostInput.data(), hostInput.size() * sizeof(float));

    builder.Execute();

    // Reference: softmax(x * scale), computed with a max subtraction for stability.
    float maxScaled = -1e30F;
    std::vector<float> scaled(K_COLUMNS);
    for (uint32_t column = 0; column < K_COLUMNS; ++column)
    {
        scaled[column] = hostInput[column] * K_SCALE;
        maxScaled = std::max(maxScaled, scaled[column]);
    }
    double sumExp = 0.0;
    for (uint32_t column = 0; column < K_COLUMNS; ++column)
    {
        sumExp += std::exp(scaled[column] - maxScaled);
    }

    const auto* pResult = static_cast<const float*>(pOutput->Map());
    uint32_t nWrong = 0;
    for (uint32_t column = 0; column < K_COLUMNS; ++column)
    {
        const float expected = static_cast<float>(std::exp(scaled[column] - maxScaled) / sumExp);
        if (std::fabs(pResult[column] - expected) > 1e-5F * (1.0F + std::fabs(expected)))
        {
            if (nWrong < 4)
            {
                std::fprintf(stderr, "[fail] D[%u] = %g, expected %g\n", column, pResult[column], expected);
            }
            ++nWrong;
        }
    }
    pOutput->Unmap();

    if (nWrong != 0)
    {
        std::fprintf(stderr, "[fail] %u of %u softmax values wrong\n", nWrong, K_COLUMNS);
        return 1;
    }

    std::printf("[softmax] OK: llama.cpp soft_max over %u columns matches the CPU reference\n", K_COLUMNS);
    return 0;
}
}  // namespace muyo_llm
