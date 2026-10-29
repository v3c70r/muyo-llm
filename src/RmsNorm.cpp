#include "RmsNorm.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "GraphServices.h"
#include "HeapIndices.h"
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
constexpr uint32_t K_ROWS = 8;
constexpr uint32_t K_COLS = 512;
constexpr float K_EPSILON = 1e-6F;

/// `rms_norm.comp`'s `do_multiply` specialization constant (generic_binary_head's shader, id 1):
/// true multiplies by the weight, which is what a transformer's RMSNorm does.
constexpr uint32_t RMS_NORM_MULTIPLY_CONSTANT = 1;

/// Relative tolerance: the shader uses `inversesqrt`, which is not bit-identical to `1/sqrt`. The
/// inputs are small integers so the sum and mean are exact, and the difference is confined to the
/// reciprocal square root.
bool NearlyEqual(float actual, float expected)
{
    return std::fabs(actual - expected) <= 1e-4F * (1.0F + std::fabs(expected));
}
}  // namespace

int RunRmsNorm()
{
    std::vector<float> hostA(K_ROWS * K_COLS);
    std::vector<float> hostWeight(K_COLS);
    for (uint32_t row = 0; row < K_ROWS; ++row)
    {
        for (uint32_t column = 0; column < K_COLS; ++column)
        {
            hostA[row * K_COLS + column] = static_cast<float>(static_cast<int>((row * K_COLS + column) % 13) - 6);
        }
    }
    for (uint32_t column = 0; column < K_COLS; ++column)
    {
        hostWeight[column] = 0.5F + 0.001F * static_cast<float>(column % 11);
    }

    RenderGraphBuilder builder(GraphServices());

    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    const auto buffer = [](uint32_t count, uint32_t stride, VkBufferUsageFlags usage)
    {
        return BufferResourceDesc{.count = count,
                                  .stride = stride,
                                  .usage = usage,
                                  .memoryProperties = VMA_MEMORY_USAGE_CPU_TO_GPU,
                                  .lifetime = ResourceLifetime::Persistent};
    };
    builder.AddResource("RmsInput", buffer(K_ROWS * K_COLS, sizeof(float), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    builder.AddResource("RmsWeight", buffer(K_COLS, sizeof(float), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    builder.AddResource("RmsOutput", buffer(K_ROWS * K_COLS, sizeof(float), storage));

    RenderGraphNodeCreateInfo normPass = {
        .nodeName = "LlamaRmsNorm",
        .queueType = QueueType::COMPUTE,
        .resourceUses =
            {
                ResourceUse{.handle = ResourceHandle("RmsInput"),
                            .io = ResourceIOType::READ,
                            .usage = ResourceUsage::STORAGE_BUFFER,
                            .kind = ResourceKind::BUFFER},
                ResourceUse{.handle = ResourceHandle("RmsWeight"),
                            .io = ResourceIOType::READ,
                            .usage = ResourceUsage::STORAGE_BUFFER,
                            .kind = ResourceKind::BUFFER},
                ResourceUse{.handle = ResourceHandle("RmsOutput"),
                            .io = ResourceIOType::READ_WRITE,
                            .usage = ResourceUsage::STORAGE_BUFFER,
                            .kind = ResourceKind::BUFFER},
            },
        .shaderNames = {"rms_norm_f32.comp"},
        .specializationConstants = {{.id = RMS_NORM_MULTIPLY_CONSTANT, .value = 1}},
        .execute = [](RenderGraphNodeContext& ctx)
        {
            // ggml's `ggml_vk_rms_norm_push_constants`: src0 and dst are [rows, cols]
            // transposed in ggml's order - ne[0] = cols, nb[0] = 1 (elements, again).
            // The weight is 1-D, so its outer strides are 0 and every row reads the
            // same `ne10 = cols` elements.
            LlamaAddPushConstants pushConstants{};
            pushConstants.ne = K_ROWS * K_COLS;
            pushConstants.ne00 = K_COLS;
            pushConstants.ne01 = K_ROWS;
            pushConstants.ne02 = 1;
            pushConstants.ne03 = 1;
            pushConstants.nb00 = 1;
            pushConstants.nb01 = K_COLS;
            pushConstants.nb02 = K_ROWS * K_COLS;
            pushConstants.nb03 = K_ROWS * K_COLS;
            pushConstants.ne10 = K_COLS;
            pushConstants.ne11 = 1;
            pushConstants.ne12 = 1;
            pushConstants.ne13 = 1;
            pushConstants.nb10 = 1;
            pushConstants.nb11 = 0;
            pushConstants.nb12 = 0;
            pushConstants.nb13 = 0;
            pushConstants.ne20 = K_COLS;
            pushConstants.ne21 = K_ROWS;
            pushConstants.ne22 = 1;
            pushConstants.ne23 = 1;
            pushConstants.nb20 = 1;
            pushConstants.nb21 = K_COLS;
            pushConstants.nb22 = K_ROWS * K_COLS;
            pushConstants.nb23 = K_ROWS * K_COLS;
            pushConstants.misalignOffsets = 0;
            pushConstants.param1 = K_EPSILON;
            pushConstants.param2 = 0.0F;
            pushConstants.param3 = 0;
            SetHeapIndex(pushConstants.heap, llama_add_bindings::A, ctx, "RmsInput", ResourceUsage::STORAGE_BUFFER);
            SetHeapIndex(pushConstants.heap, llama_add_bindings::B, ctx, "RmsWeight", ResourceUsage::STORAGE_BUFFER);
            SetHeapIndex(pushConstants.heap, llama_add_bindings::D, ctx, "RmsOutput", ResourceUsage::STORAGE_BUFFER);
            PushData(ctx, pushConstants);

            // One workgroup per row (ggml dispatches `{ne[1], ne[2], ne[3]}`).
            vkCmdDispatch(ctx.commandBuffer, K_ROWS, 1, 1);

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

    builder.AddNode(normPass);
    builder.Build();

    std::printf("[graph] execution order:");
    for (const std::string& nodeName : builder.GetExecutionOrder())
    {
        std::printf(" %s", nodeName.c_str());
    }
    std::printf("\n");

    auto* pInput = GetRenderResourceManager()->GetResource<BufferResource>("RmsInput");
    auto* pWeight = GetRenderResourceManager()->GetResource<BufferResource>("RmsWeight");
    auto* pOutput = GetRenderResourceManager()->GetResource<BufferResource>("RmsOutput");
    if (pInput == nullptr || pWeight == nullptr || pOutput == nullptr)
    {
        std::fprintf(stderr, "[fail] the graph did not allocate the rms_norm buffers\n");
        return 1;
    }
    pInput->SetData(hostA.data(), hostA.size() * sizeof(float));
    pWeight->SetData(hostWeight.data(), hostWeight.size() * sizeof(float));

    builder.Execute();

    const auto* pResult = static_cast<const float*>(pOutput->Map());
    uint32_t nWrong = 0;
    for (uint32_t row = 0; row < K_ROWS; ++row)
    {
        double sumSquares = 0.0;
        for (uint32_t column = 0; column < K_COLS; ++column)
        {
            const double value = hostA[row * K_COLS + column];
            sumSquares += value * value;
        }
        const float scale = static_cast<float>(1.0 / std::sqrt(sumSquares / K_COLS + K_EPSILON));
        for (uint32_t column = 0; column < K_COLS; ++column)
        {
            const float expected = hostA[row * K_COLS + column] * scale * hostWeight[column];
            const float actual = pResult[row * K_COLS + column];
            if (!NearlyEqual(actual, expected))
            {
                if (nWrong < 4)
                {
                    std::fprintf(stderr, "[fail] D[%u][%u] = %g, expected %g\n", row, column, actual, expected);
                }
                ++nWrong;
            }
        }
    }
    pOutput->Unmap();

    if (nWrong != 0)
    {
        std::fprintf(stderr, "[fail] %u of %u elements wrong\n", nWrong, K_ROWS * K_COLS);
        return 1;
    }

    std::printf("[rmsnorm] OK: llama.cpp rms_norm over %ux%u matches the CPU reference\n", K_ROWS, K_COLS);
    return 0;
}
}  // namespace muyo_llm
