#include "GetRows.h"

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
constexpr uint32_t K_EMBEDDING_DIM = 512;
constexpr uint32_t K_VOCAB_SIZE = 1024;
constexpr int32_t K_TOKEN = 7;
}  // namespace

int RunGetRows()
{
    std::vector<float> hostWeight(static_cast<size_t>(K_VOCAB_SIZE) * K_EMBEDDING_DIM);
    for (uint32_t token = 0; token < K_VOCAB_SIZE; ++token)
    {
        for (uint32_t channel = 0; channel < K_EMBEDDING_DIM; ++channel)
        {
            hostWeight[static_cast<size_t>(token) * K_EMBEDDING_DIM + channel] =
                static_cast<float>(static_cast<int>(token % 5) - 2) + 0.001F * static_cast<float>(channel);
        }
    }
    std::vector<int32_t> hostIndices = {K_TOKEN};

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
    builder.AddResource("EmbeddingWeight", buffer(K_VOCAB_SIZE * K_EMBEDDING_DIM, sizeof(float),
                                                  storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    builder.AddResource("EmbeddingIndices", buffer(1, sizeof(int32_t), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    builder.AddResource("EmbeddingOutput", buffer(K_EMBEDDING_DIM, sizeof(float), storage));

    const auto binding = [](uint32_t index) { return DescriptorBinding{.set = 0, .binding = index}; };
    const auto readUse = [&binding](const char* handle, uint32_t index)
    {
        return ResourceUse{.handle = ResourceHandle(handle),
                           .io = ResourceIOType::READ,
                           .usage = ResourceUsage::STORAGE_BUFFER,
                           .kind = ResourceKind::BUFFER,
                           .descriptorBinding = binding(index)};
    };

    RenderGraphNodeCreateInfo gatherPass = {.nodeName = "LlamaGetRows",
                                            .queueType = QueueType::COMPUTE,
                                            .resourceUses =
                                                {
                                                    readUse("EmbeddingWeight", llama_add_bindings::A),
                                                    readUse("EmbeddingIndices", llama_add_bindings::B),
                                                    ResourceUse{.handle = ResourceHandle("EmbeddingOutput"),
                                                                .io = ResourceIOType::READ_WRITE,
                                                                .usage = ResourceUsage::STORAGE_BUFFER,
                                                                .kind = ResourceKind::BUFFER,
                                                                .descriptorBinding = binding(llama_add_bindings::D)},
                                                },
                                            .shaderNames = {"get_rows_f32.comp"},
                                            .execute = [](RenderGraphNodeContext& ctx)
                                            {
                                                // ggml's `ggml_vk_get_rows`: src0 is the [n_vocab, n_embd] weight in
                                                // ggml's order - ne00 = n_embd, ne01 = n_vocab, strides in elements.
                                                // The index tensor is 1-D; only its element 0 is read.
                                                LlamaAddPushConstants pushConstants{};
                                                pushConstants.ne = K_VOCAB_SIZE * K_EMBEDDING_DIM;
                                                pushConstants.ne00 = K_EMBEDDING_DIM;
                                                pushConstants.ne01 = K_VOCAB_SIZE;
                                                pushConstants.ne02 = 1;
                                                pushConstants.ne03 = 1;
                                                pushConstants.nb00 = 1;
                                                pushConstants.nb01 = K_EMBEDDING_DIM;
                                                pushConstants.nb02 = K_VOCAB_SIZE * K_EMBEDDING_DIM;
                                                pushConstants.nb03 = K_VOCAB_SIZE * K_EMBEDDING_DIM;
                                                pushConstants.ne10 = 1;
                                                pushConstants.ne11 = 1;
                                                pushConstants.ne12 = 1;
                                                pushConstants.ne13 = 1;
                                                pushConstants.nb10 = 1;
                                                pushConstants.nb11 = 1;
                                                pushConstants.nb12 = 1;
                                                pushConstants.nb13 = 1;
                                                pushConstants.ne20 = K_EMBEDDING_DIM;
                                                pushConstants.ne21 = 1;
                                                pushConstants.ne22 = 1;
                                                pushConstants.ne23 = 1;
                                                pushConstants.nb20 = 1;
                                                pushConstants.nb21 = K_EMBEDDING_DIM;
                                                pushConstants.nb22 = K_EMBEDDING_DIM;
                                                pushConstants.nb23 = K_EMBEDDING_DIM;
                                                pushConstants.misalignOffsets = 0;
                                                pushConstants.param1 = 0.0F;
                                                pushConstants.param2 = 0.0F;
                                                pushConstants.param3 = 0;
                                                vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout,
                                                                   VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                                                   sizeof(pushConstants), &pushConstants);

                                                // ggml dispatches `{ne00, ne10, ne11 * ne12}` against wg_denoms `{512,
                                                // 1, 1}`.
                                                const uint32_t blocks = (K_EMBEDDING_DIM + 511) / 512;
                                                vkCmdDispatch(ctx.commandBuffer, blocks, 1, 1);

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

    builder.AddNode(gatherPass);
    builder.Build();

    std::printf("[graph] execution order:");
    for (const std::string& nodeName : builder.GetExecutionOrder())
    {
        std::printf(" %s", nodeName.c_str());
    }
    std::printf("\n");

    auto* pWeight = GetRenderResourceManager()->GetResource<BufferResource>("EmbeddingWeight");
    auto* pIndices = GetRenderResourceManager()->GetResource<BufferResource>("EmbeddingIndices");
    auto* pOutput = GetRenderResourceManager()->GetResource<BufferResource>("EmbeddingOutput");
    if (pWeight == nullptr || pIndices == nullptr || pOutput == nullptr)
    {
        std::fprintf(stderr, "[fail] the graph did not allocate the get_rows buffers\n");
        return 1;
    }
    pWeight->SetData(hostWeight.data(), hostWeight.size() * sizeof(float));
    pIndices->SetData(hostIndices.data(), hostIndices.size() * sizeof(int32_t));

    builder.Execute();

    const auto* pResult = static_cast<const float*>(pOutput->Map());
    uint32_t nWrong = 0;
    for (uint32_t channel = 0; channel < K_EMBEDDING_DIM; ++channel)
    {
        const float expected = hostWeight[static_cast<size_t>(K_TOKEN) * K_EMBEDDING_DIM + channel];
        if (pResult[channel] != expected)
        {
            if (nWrong < 4)
            {
                std::fprintf(stderr, "[fail] D[%u] = %g, expected %g\n", channel, pResult[channel], expected);
            }
            ++nWrong;
        }
    }
    pOutput->Unmap();

    if (nWrong != 0)
    {
        std::fprintf(stderr, "[fail] %u of %u embedding channels wrong\n", nWrong, K_EMBEDDING_DIM);
        return 1;
    }

    std::printf("[getrows] OK: llama.cpp get_rows gathered token %d's %u-channel embedding\n", K_TOKEN,
                K_EMBEDDING_DIM);
    return 0;
}
}  // namespace muyo_llm
