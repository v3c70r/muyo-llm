#include "MatVec.h"

#include <cstdint>
#include <cstdio>
#include <glm/gtc/packing.hpp>
#include <glm/vec2.hpp>
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
constexpr uint32_t K_ROWS = 128;
constexpr uint32_t K_COLS = 256;
constexpr uint32_t K_BLOCK_SIZE = 32;

/// Pack a float into its f16 bit pattern. The inputs are small integers, which f16 represents
/// exactly, so the weight buffer carries the values the CPU reference uses, bit for bit.
uint16_t ToHalf(float value) { return static_cast<uint16_t>(glm::packHalf2x16(glm::vec2(value, 0.0F)) & 0xFFFFu); }

/// The same sum `mul_mat_vec.comp` computes. The inputs are small integers, whose products summed
/// over 256 columns are exact in f32 - so the comparison has no tolerance to hide a wrong index in.
float HostMatVec(const std::vector<uint16_t>& a, const std::vector<float>& b, uint32_t row)
{
    float sum = 0.0F;
    for (uint32_t column = 0; column < K_COLS; ++column)
    {
        sum += glm::unpackHalf2x16(a[row * K_COLS + column]).x * b[column];
    }
    return sum;
}
}  // namespace

int RunMatVec()
{
    // F16 weights, F32 activations: llama.cpp's `mul_mat_vec_f16_f32_f32`. The F16 storage buffer
    // needs `storageBuffer16BitAccess`, which a consumer could not enable until Muyo #78 landed
    // (docs/Findings.md F8); that is why this variant is only reachable now.
    std::vector<uint16_t> hostA(K_ROWS * K_COLS);
    std::vector<float> hostB(K_COLS);
    for (uint32_t row = 0; row < K_ROWS; ++row)
    {
        for (uint32_t column = 0; column < K_COLS; ++column)
        {
            hostA[row * K_COLS + column] = ToHalf(static_cast<float>(static_cast<int>(column % 7) - 3));
        }
    }
    for (uint32_t column = 0; column < K_COLS; ++column)
    {
        hostB[column] = static_cast<float>(static_cast<int>(column % 5) - 2);
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
    builder.AddResource("WeightsA",
                        buffer(K_ROWS * K_COLS, sizeof(uint16_t), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    builder.AddResource("InputB", buffer(K_COLS, sizeof(float), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    builder.AddResource("OutputD", buffer(K_ROWS, sizeof(float), storage));
    // `mul_mat_vec.comp` declares bindings 3 and 4 unconditionally; they are read only under the
    // matching fusion flag, which is 0 here, but the descriptor must still be valid.
    builder.AddResource("Fuse0", buffer(K_ROWS, sizeof(float), storage));
    builder.AddResource("Fuse1", buffer(K_ROWS, sizeof(float), storage));

    const auto binding = [](uint32_t index) { return DescriptorBinding{.set = 0, .binding = index}; };
    const auto readUse = [&binding](const char* handle, uint32_t index)
    {
        return ResourceUse{.handle = ResourceHandle(handle),
                           .io = ResourceIOType::READ,
                           .usage = ResourceUsage::STORAGE_BUFFER,
                           .kind = ResourceKind::BUFFER,
                           .descriptorBinding = binding(index)};
    };

    RenderGraphNodeCreateInfo gemvPass = {
        .nodeName = "LlamaMatVec",
        .queueType = QueueType::COMPUTE,
        .resourceUses =
            {
                readUse("WeightsA", llama_mat_vec_bindings::A),
                readUse("InputB", llama_mat_vec_bindings::B),
                // READ_WRITE, not WRITE: `{STORAGE_BUFFER, WRITE}` is a *host* write in the usage
                // table (docs/Findings.md F3, Muyo #12).
                ResourceUse{.handle = ResourceHandle("OutputD"),
                            .io = ResourceIOType::READ_WRITE,
                            .usage = ResourceUsage::STORAGE_BUFFER,
                            .kind = ResourceKind::BUFFER,
                            .descriptorBinding = binding(llama_mat_vec_bindings::D)},
                readUse("Fuse0", llama_mat_vec_bindings::FUSE0),
                readUse("Fuse1", llama_mat_vec_bindings::FUSE1),
            },
        .shaderNames = {"mul_mat_vec_f16_f32_f32.comp"},
        .specializationConstants =
            {
                {.id = llama_mat_vec_spec_constants::BLOCK_SIZE, .value = K_BLOCK_SIZE},
                {.id = llama_mat_vec_spec_constants::NUM_ROWS, .value = 1},
                {.id = llama_mat_vec_spec_constants::NUM_COLS, .value = 1},
            },
        .execute = [](RenderGraphNodeContext& ctx)
        {
            // ggml's `ggml_vk_mul_mat_vec` fill: ncols = ne00 (K), stride_a = ne10 (K),
            // stride_b = ne10 (K), stride_d = ne01 (rows), the batch fields zero, ne02 = ne12 =
            // broadcast2 = broadcast3 = 1.
            LlamaMatVecPushConstants pushConstants{};
            pushConstants.ncols = K_COLS;
            pushConstants.strideA = K_COLS;
            pushConstants.strideB = K_COLS;
            pushConstants.strideD = K_ROWS;
            pushConstants.batchStrideA = 0;
            pushConstants.batchStrideB = 0;
            pushConstants.batchStrideD = 0;
            pushConstants.fusionFlags = 0;
            pushConstants.baseWorkGroupY = 0;
            pushConstants.ne02 = 1;
            pushConstants.ne12 = 1;
            pushConstants.broadcast2 = 1;
            pushConstants.broadcast3 = 1;
            vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(pushConstants), &pushConstants);

            // NUM_ROWS rows per workgroup; one workgroup per row here.
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

    builder.AddNode(gemvPass);
    builder.Build();

    std::printf("[graph] execution order:");
    for (const std::string& nodeName : builder.GetExecutionOrder())
    {
        std::printf(" %s", nodeName.c_str());
    }
    std::printf("\n");

    auto* pA = GetRenderResourceManager()->GetResource<BufferResource>("WeightsA");
    auto* pB = GetRenderResourceManager()->GetResource<BufferResource>("InputB");
    auto* pD = GetRenderResourceManager()->GetResource<BufferResource>("OutputD");
    if (pA == nullptr || pB == nullptr || pD == nullptr)
    {
        std::fprintf(stderr, "[fail] the graph did not allocate the matvec buffers\n");
        return 1;
    }
    pA->SetData(hostA.data(), hostA.size() * sizeof(uint16_t));
    pB->SetData(hostB.data(), hostB.size() * sizeof(float));

    builder.Execute();

    const auto* pResult = static_cast<const float*>(pD->Map());
    uint32_t nWrong = 0;
    for (uint32_t row = 0; row < K_ROWS; ++row)
    {
        const float expected = HostMatVec(hostA, hostB, row);
        if (pResult[row] != expected)
        {
            if (nWrong < 4)
            {
                std::fprintf(stderr, "[fail] D[%u] = %g, expected %g\n", row, pResult[row], expected);
            }
            ++nWrong;
        }
    }
    pD->Unmap();

    if (nWrong != 0)
    {
        std::fprintf(stderr, "[fail] %u of %u rows wrong\n", nWrong, K_ROWS);
        return 1;
    }

    std::printf("[matvec] OK: llama.cpp mul_mat_vec (F16 weights) over %ux%u matches the CPU reference\n", K_ROWS,
                K_COLS);
    return 0;
}
}  // namespace muyo_llm
