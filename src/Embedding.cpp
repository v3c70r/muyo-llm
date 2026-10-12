#include "Embedding.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <glm/gtc/packing.hpp>
#include <glm/vec2.hpp>
#include <string>
#include <vector>

#include "Gguf.h"
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

int RunEmbedding(const std::string& path, int32_t token)
{
    const GgufReader reader(path);
    const GgufTensorInfo* tensor = reader.FindTensor("token_embd.weight");
    if (tensor == nullptr)
    {
        std::fprintf(stderr, "[fail] %s has no token_embd.weight\n", path.c_str());
        return 1;
    }
    // F32 (the tiny test model) or F16 (the real one). Both widen to F32 in the shader, so the
    // host reference is exact.
    const bool isF16 = tensor->type == 1;
    if (tensor->type != 0 && !isF16)
    {
        std::fprintf(stderr, "[fail] token_embd.weight is type %u, expected F32 or F16\n", tensor->type);
        return 1;
    }

    // GGUF order: fastest-varying dimension first, so dims = [embd, vocab] and the raw bytes are
    // already one token vector after another.
    const uint32_t embeddingDim = static_cast<uint32_t>(tensor->dimensions[0]);
    const uint32_t vocabSize = static_cast<uint32_t>(tensor->dimensions[1]);
    if (token < 0 || static_cast<uint32_t>(token) >= vocabSize)
    {
        std::fprintf(stderr, "[fail] token %d out of range for vocab %u\n", token, vocabSize);
        return 1;
    }
    const std::vector<uint8_t> bytes = reader.ReadTensorData("token_embd.weight");
    std::vector<float> expected(embeddingDim);
    if (isF16)
    {
        const auto* halves = reinterpret_cast<const uint16_t*>(bytes.data());
        for (uint32_t d = 0; d < embeddingDim; ++d)
        {
            expected[d] = glm::unpackHalf2x16(halves[static_cast<size_t>(token) * embeddingDim + d]).x;
        }
    }
    else
    {
        const auto* floats = reinterpret_cast<const float*>(bytes.data());
        for (uint32_t d = 0; d < embeddingDim; ++d)
        {
            expected[d] = floats[static_cast<size_t>(token) * embeddingDim + d];
        }
    }

    RenderGraphBuilder builder(GraphServices());
    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    builder.AddResource("TokenEmbedding", BufferResourceDesc{.count = static_cast<uint64_t>(vocabSize) * embeddingDim,
                                                             .stride = sizeof(float),
                                                             .usage = storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                             .memoryProperties = VMA_MEMORY_USAGE_CPU_TO_GPU,
                                                             .lifetime = ResourceLifetime::Transient});
    builder.AddResource("TokenIndex", BufferResourceDesc{.count = 1,
                                                         .stride = sizeof(int32_t),
                                                         .usage = storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                         .memoryProperties = VMA_MEMORY_USAGE_CPU_TO_GPU,
                                                         .lifetime = ResourceLifetime::Transient});
    builder.AddResource("TokenVector", BufferResourceDesc{.count = embeddingDim,
                                                          .stride = sizeof(float),
                                                          .usage = storage,
                                                          .memoryProperties = VMA_MEMORY_USAGE_CPU_TO_GPU,
                                                          .lifetime = ResourceLifetime::Transient});

    const auto binding = [](uint32_t index) { return DescriptorBinding{.set = 0, .binding = index}; };
    const auto use = [&binding](const char* handle, ResourceIOType io, uint32_t index)
    {
        return ResourceUse{.handle = ResourceHandle(handle),
                           .io = io,
                           .usage = ResourceUsage::STORAGE_BUFFER,
                           .kind = ResourceKind::BUFFER,
                           .descriptorBinding = binding(index)};
    };

    RenderGraphNodeCreateInfo gatherPass = {
        .nodeName = "EmbedToken",
        .queueType = QueueType::COMPUTE,
        .resourceUses = {use("TokenEmbedding", ResourceIOType::READ, llama_add_bindings::A),
                         use("TokenIndex", ResourceIOType::READ, llama_add_bindings::B),
                         use("TokenVector", ResourceIOType::READ_WRITE, llama_add_bindings::D)},
        .shaderNames = {isF16 ? "get_rows_f16_f32.comp" : "get_rows_f32.comp"},
        .execute = [embeddingDim, vocabSize](RenderGraphNodeContext& ctx)
        {
            LlamaAddPushConstants pushConstants{};
            pushConstants.ne = vocabSize * embeddingDim;
            pushConstants.ne00 = embeddingDim;
            pushConstants.ne01 = vocabSize;
            pushConstants.ne02 = 1;
            pushConstants.ne03 = 1;
            pushConstants.nb00 = 1;
            pushConstants.nb01 = embeddingDim;
            pushConstants.nb02 = vocabSize * embeddingDim;
            pushConstants.nb03 = vocabSize * embeddingDim;
            pushConstants.ne10 = 1;
            pushConstants.ne11 = 1;
            pushConstants.ne12 = 1;
            pushConstants.ne13 = 1;
            pushConstants.nb10 = 1;
            pushConstants.nb11 = 1;
            pushConstants.nb12 = 1;
            pushConstants.nb13 = 1;
            pushConstants.ne20 = embeddingDim;
            pushConstants.ne21 = 1;
            pushConstants.ne22 = 1;
            pushConstants.ne23 = 1;
            pushConstants.nb20 = 1;
            pushConstants.nb21 = embeddingDim;
            pushConstants.nb22 = embeddingDim;
            pushConstants.nb23 = embeddingDim;
            vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(pushConstants), &pushConstants);
            vkCmdDispatch(ctx.commandBuffer, (embeddingDim + 511) / 512, 1, 1);

            // The compute-write to host-read barrier (docs/Findings.md F4): without it the gather's
            // result stays in cache and the mapped read sees zeros.
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

    auto* pEmbedding = GetRenderResourceManager()->GetResource<BufferResource>("TokenEmbedding");
    auto* pIndex = GetRenderResourceManager()->GetResource<BufferResource>("TokenIndex");
    auto* pVector = GetRenderResourceManager()->GetResource<BufferResource>("TokenVector");
    pEmbedding->SetData(bytes.data(), bytes.size());
    pIndex->SetData(&token, sizeof(int32_t));

    builder.Execute();

    const auto* result = static_cast<const float*>(pVector->Map());
    uint32_t nWrong = 0;
    for (uint32_t d = 0; d < embeddingDim; ++d)
    {
        if (result[d] != expected[d])
        {
            if (nWrong < 4)
            {
                std::fprintf(stderr, "[fail] emb[%u] = %g, expected %g\n", d, result[d], expected[d]);
            }
            ++nWrong;
        }
    }
    pVector->Unmap();

    if (nWrong != 0)
    {
        std::fprintf(stderr, "[fail] %u of %u embedding values wrong\n", nWrong, embeddingDim);
        return 1;
    }

    std::printf("[embed] OK: token %d's %u-value embedding loaded from %s matches the file\n", token, embeddingDim,
                path.c_str());
    return 0;
}
}  // namespace muyo_llm
