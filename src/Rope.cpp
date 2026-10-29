#include "Rope.h"

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
constexpr uint32_t K_N_DIMS = 8;
constexpr uint32_t K_ROWS = 1;
constexpr float K_FREQ_BASE = 10000.0F;
constexpr int32_t K_POSITION = 3;

/// The rotation the shader applies, for one pair `j`: `theta = pos * thetaScale^j`, mixing channel
/// `j` with channel `j + nDims/2`.
float ThetaForPair(uint32_t j, float thetaScale)
{
    return static_cast<float>(K_POSITION) * std::pow(thetaScale, static_cast<float>(j));
}
}  // namespace

int RunRope()
{
    std::vector<float> hostInput(K_ROWS * K_N_DIMS);
    for (uint32_t row = 0; row < K_ROWS; ++row)
    {
        for (uint32_t channel = 0; channel < K_N_DIMS; ++channel)
        {
            hostInput[row * K_N_DIMS + channel] = static_cast<float>(static_cast<int>(channel) - 3);
        }
    }
    std::vector<int32_t> hostPositions(K_ROWS, K_POSITION);

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
    builder.AddResource("RopeInput",
                        buffer(K_ROWS * K_N_DIMS, sizeof(float), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    builder.AddResource("RopePos", buffer(K_ROWS, sizeof(int32_t), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    // Bindings 2 and 4 are declared unconditionally; they are read only under `hasFf` and the fused
    // set-rows path, neither of which is used here, but the descriptors must still be valid.
    builder.AddResource("RopeFreqFactors", buffer(K_N_DIMS / 2, sizeof(float), storage));
    builder.AddResource("RopeOutput", buffer(K_ROWS * K_N_DIMS, sizeof(float), storage));
    builder.AddResource("RopeSetRows", buffer(K_ROWS, sizeof(uint32_t) * 2, storage));

    const auto readUse = [](const char* handle, uint32_t /*bindingSlot*/)
    {
        return ResourceUse{.handle = ResourceHandle(handle),
                           .io = ResourceIOType::READ,
                           .usage = ResourceUsage::STORAGE_BUFFER,
                           .kind = ResourceKind::BUFFER};
    };

    RenderGraphNodeCreateInfo ropePass = {
        .nodeName = "LlamaRope",
        .queueType = QueueType::COMPUTE,
        .resourceUses =
            {
                readUse("RopeInput", llama_rope_bindings::X),
                readUse("RopePos", llama_rope_bindings::Y),
                readUse("RopeFreqFactors", llama_rope_bindings::Z),
                ResourceUse{.handle = ResourceHandle("RopeOutput"),
                            .io = ResourceIOType::READ_WRITE,
                            .usage = ResourceUsage::STORAGE_BUFFER,
                            .kind = ResourceKind::BUFFER},
                readUse("RopeSetRows", llama_rope_bindings::I),
            },
        .shaderNames = {"rope_neox_f32.comp"},
        .execute = [](RenderGraphNodeContext& ctx)
        {
            // ggml's `ggml_vk_make_rope_constants`: mode NEOX, no extrapolation (extFactor 0,
            // attnFactor 1, freqScale 1), theta_scale = freqBase^(-2/nDims). Strides are in
            // elements, as everywhere else. corr_dims is only read when extFactor is non-zero.
            LlamaRopePushConstants pushConstants{};
            pushConstants.ropeMode = llama_rope_modes::NEOX;
            pushConstants.nrows = K_ROWS;
            pushConstants.nDims = K_N_DIMS;
            pushConstants.nOffs = 0;
            pushConstants.freqScale = 1.0F;
            pushConstants.freqBase = K_FREQ_BASE;
            pushConstants.extFactor = 0.0F;
            pushConstants.attnFactor = 1.0F;
            pushConstants.corrDims[0] = 0.0F;
            pushConstants.corrDims[1] = 0.0F;
            pushConstants.thetaScale = std::pow(K_FREQ_BASE, -2.0F / static_cast<float>(K_N_DIMS));
            pushConstants.hasFf = 0;
            for (int32_t& section : pushConstants.sections)
            {
                section = 0;
            }
            pushConstants.isImrope = 0;
            pushConstants.isBack = 0;
            pushConstants.setRowsStride = 0;
            pushConstants.ne00 = K_N_DIMS;
            pushConstants.ne01 = 1;
            pushConstants.ne02 = 1;
            pushConstants.nb01 = K_N_DIMS;
            pushConstants.nb02 = K_N_DIMS;
            pushConstants.nb03 = K_N_DIMS;
            pushConstants.nb11 = K_N_DIMS;
            pushConstants.nb12 = K_N_DIMS;
            pushConstants.nb13 = K_N_DIMS;
            pushConstants.aOffset = 0;
            pushConstants.dOffset = 0;
            SetHeapIndex(pushConstants.heap, llama_rope_bindings::X, ctx, "RopeInput", ResourceUsage::STORAGE_BUFFER);
            SetHeapIndex(pushConstants.heap, llama_rope_bindings::Y, ctx, "RopePos", ResourceUsage::STORAGE_BUFFER);
            SetHeapIndex(pushConstants.heap, llama_rope_bindings::Z, ctx, "RopeFreqFactors",
                         ResourceUsage::STORAGE_BUFFER);
            SetHeapIndex(pushConstants.heap, llama_rope_bindings::D, ctx, "RopeOutput", ResourceUsage::STORAGE_BUFFER);
            SetHeapIndex(pushConstants.heap, llama_rope_bindings::I, ctx, "RopeSetRows", ResourceUsage::STORAGE_BUFFER);
            PushData(ctx, pushConstants);

            // ggml dispatches `{nrows, ne00, z}` against wg_denoms `{1, 512, 1}`; one row and
            // nDims <= 512 is one workgroup.
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

    builder.AddNode(ropePass);
    builder.Build();

    std::printf("[graph] execution order:");
    for (const std::string& nodeName : builder.GetExecutionOrder())
    {
        std::printf(" %s", nodeName.c_str());
    }
    std::printf("\n");

    auto* pInput = GetRenderResourceManager()->GetResource<BufferResource>("RopeInput");
    auto* pPos = GetRenderResourceManager()->GetResource<BufferResource>("RopePos");
    auto* pOutput = GetRenderResourceManager()->GetResource<BufferResource>("RopeOutput");
    if (pInput == nullptr || pPos == nullptr || pOutput == nullptr)
    {
        std::fprintf(stderr, "[fail] the graph did not allocate the rope buffers\n");
        return 1;
    }
    pInput->SetData(hostInput.data(), hostInput.size() * sizeof(float));
    pPos->SetData(hostPositions.data(), hostPositions.size() * sizeof(int32_t));

    builder.Execute();

    const float thetaScale = std::pow(K_FREQ_BASE, -2.0F / static_cast<float>(K_N_DIMS));
    const auto* pResult = static_cast<const float*>(pOutput->Map());
    uint32_t nWrong = 0;
    const uint32_t half = K_N_DIMS / 2;
    for (uint32_t j = 0; j < half; ++j)
    {
        const float theta = ThetaForPair(j, thetaScale);
        const float cosTheta = std::cos(theta);
        const float sinTheta = std::sin(theta);
        const float x0 = hostInput[j];
        const float x1 = hostInput[j + half];
        const float expected0 = x0 * cosTheta - x1 * sinTheta;
        const float expected1 = x0 * sinTheta + x1 * cosTheta;
        if (std::fabs(pResult[j] - expected0) > 1e-5F * (1.0F + std::fabs(expected0)) ||
            std::fabs(pResult[j + half] - expected1) > 1e-5F * (1.0F + std::fabs(expected1)))
        {
            if (nWrong < 4)
            {
                std::fprintf(stderr, "[fail] pair %u: (%g, %g) expected (%g, %g)\n", j, pResult[j], pResult[j + half],
                             expected0, expected1);
            }
            ++nWrong;
        }
    }
    pOutput->Unmap();

    if (nWrong != 0)
    {
        std::fprintf(stderr, "[fail] %u of %u rotation pairs wrong\n", nWrong, half);
        return 1;
    }

    std::printf("[rope] OK: llama.cpp rope_neox over %u channels matches the CPU reference\n", K_N_DIMS);
    return 0;
}
}  // namespace muyo_llm
