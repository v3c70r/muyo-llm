#include "Attention.h"

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
constexpr uint32_t K_HEAD_DIM = 8;    // HSK == HSV for this test.
constexpr uint32_t K_KEY_COUNT = 16;  // KV.
constexpr uint32_t K_BLOCK_SIZE = 32;

/// The attention scale the soft_max shader applies.
const float K_SCALE = 1.0F / std::sqrt(static_cast<float>(K_HEAD_DIM));

/// Fill a GEMV node's execute callback. `rows` output rows, each a dot product over `reduction`.
auto MakeGemvExecute(uint32_t reduction, uint32_t rows)
{
    return [reduction, rows](RenderGraphNodeContext& ctx)
    {
        LlamaMatVecPushConstants pushConstants{};
        pushConstants.ncols = reduction;
        pushConstants.strideA = reduction;
        pushConstants.strideB = reduction;
        pushConstants.strideD = rows;
        pushConstants.batchStrideA = 0;
        pushConstants.batchStrideB = 0;
        pushConstants.batchStrideD = 0;
        pushConstants.fusionFlags = 0;
        pushConstants.baseWorkGroupY = 0;
        pushConstants.ne02 = 1;
        pushConstants.ne12 = 1;
        pushConstants.broadcast2 = 1;
        pushConstants.broadcast3 = 1;
        vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                           &pushConstants);
        vkCmdDispatch(ctx.commandBuffer, rows, 1, 1);
    };
}
}  // namespace

int RunAttention()
{
    // Small integers: the Q.K scores and the P.V output are exact in f32, and only the softmax's
    // exp introduces a tolerance.
    std::vector<float> hostQuery(K_HEAD_DIM);
    std::vector<float> hostKeys(static_cast<size_t>(K_KEY_COUNT) * K_HEAD_DIM);
    std::vector<float> hostValues(static_cast<size_t>(K_KEY_COUNT) * K_HEAD_DIM);
    std::vector<float> hostValuesTransposed(static_cast<size_t>(K_HEAD_DIM) * K_KEY_COUNT);
    for (uint32_t d = 0; d < K_HEAD_DIM; ++d)
    {
        hostQuery[d] = static_cast<float>(static_cast<int>(d % 5) - 2);
    }
    for (uint32_t key = 0; key < K_KEY_COUNT; ++key)
    {
        for (uint32_t d = 0; d < K_HEAD_DIM; ++d)
        {
            hostKeys[static_cast<size_t>(key) * K_HEAD_DIM + d] =
                static_cast<float>(static_cast<int>((key + d) % 3) - 1);
            hostValues[static_cast<size_t>(key) * K_HEAD_DIM + d] =
                static_cast<float>(static_cast<int>((key * 2 + d) % 7) - 3);
        }
    }
    // The output GEMV is `out = Vt . probs`, so V is stored transposed: one row per output channel.
    for (uint32_t d = 0; d < K_HEAD_DIM; ++d)
    {
        for (uint32_t key = 0; key < K_KEY_COUNT; ++key)
        {
            hostValuesTransposed[static_cast<size_t>(d) * K_KEY_COUNT + key] =
                hostValues[static_cast<size_t>(key) * K_HEAD_DIM + d];
        }
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

    builder.AddResource("Query", buffer(K_HEAD_DIM, sizeof(float), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    builder.AddResource("Keys",
                        buffer(K_KEY_COUNT * K_HEAD_DIM, sizeof(float), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    builder.AddResource("ValuesTransposed",
                        buffer(K_HEAD_DIM * K_KEY_COUNT, sizeof(float), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    builder.AddResource("Scores", buffer(K_KEY_COUNT, sizeof(float), storage));
    builder.AddResource("Probabilities", buffer(K_KEY_COUNT, sizeof(float), storage));
    builder.AddResource("AttnOutput", buffer(K_HEAD_DIM, sizeof(float), storage));
    // GEMV bindings 3/4 and the softmax bindings 1/2 are declared unconditionally; unused here.
    const uint32_t fuseSize = K_KEY_COUNT > K_HEAD_DIM ? K_KEY_COUNT : K_HEAD_DIM;
    builder.AddResource("Fuse0", buffer(fuseSize, sizeof(float), storage));
    builder.AddResource("Fuse1", buffer(fuseSize, sizeof(float), storage));
    builder.AddResource("SoftMaxMask", buffer(K_KEY_COUNT, sizeof(float), storage));
    builder.AddResource("SoftMaxSinks", buffer(1, sizeof(float), storage));

    const auto binding = [](uint32_t index) { return DescriptorBinding{.set = 0, .binding = index}; };
    const auto gemvUse = [&binding](const char* handle, ResourceIOType io, uint32_t index)
    {
        return ResourceUse{.handle = ResourceHandle(handle),
                           .io = io,
                           .usage = ResourceUsage::STORAGE_BUFFER,
                           .kind = ResourceKind::BUFFER,
                           .descriptorBinding = binding(index)};
    };

    // Node 1: scores = K . q. K is [KV, HSK], so one row per key.
    RenderGraphNodeCreateInfo scoresPass = {
        .nodeName = "AttnScores",
        .queueType = QueueType::COMPUTE,
        .resourceUses =
            {
                gemvUse("Keys", ResourceIOType::READ, llama_mat_vec_bindings::A),
                gemvUse("Query", ResourceIOType::READ, llama_mat_vec_bindings::B),
                gemvUse("Scores", ResourceIOType::READ_WRITE, llama_mat_vec_bindings::D),
                gemvUse("Fuse0", ResourceIOType::READ, llama_mat_vec_bindings::FUSE0),
                gemvUse("Fuse1", ResourceIOType::READ, llama_mat_vec_bindings::FUSE1),
            },
        .shaderNames = {"mul_mat_vec_f32_f32_f32.comp"},
        .specializationConstants =
            {
                {.id = llama_mat_vec_spec_constants::BLOCK_SIZE, .value = K_BLOCK_SIZE},
                {.id = llama_mat_vec_spec_constants::NUM_ROWS, .value = 1},
                {.id = llama_mat_vec_spec_constants::NUM_COLS, .value = 1},
            },
        .execute = MakeGemvExecute(K_HEAD_DIM, K_KEY_COUNT)};

    // Node 2: probs = softmax(scores * scale).
    RenderGraphNodeCreateInfo weightsPass = {
        .nodeName = "AttnWeights",
        .queueType = QueueType::COMPUTE,
        .resourceUses =
            {
                gemvUse("Scores", ResourceIOType::READ, llama_soft_max_bindings::X),
                gemvUse("SoftMaxMask", ResourceIOType::READ, llama_soft_max_bindings::Y),
                gemvUse("SoftMaxSinks", ResourceIOType::READ, llama_soft_max_bindings::Z),
                gemvUse("Probabilities", ResourceIOType::READ_WRITE, llama_soft_max_bindings::D),
            },
        .shaderNames = {"soft_max_f32.comp"},
        .specializationConstants = {{.id = llama_soft_max_spec_constants::BLOCK_SIZE, .value = K_BLOCK_SIZE}},
        .execute = [](RenderGraphNodeContext& ctx)
        {
            LlamaSoftMaxPushConstants pushConstants{};
            pushConstants.KX = K_KEY_COUNT;
            pushConstants.KY = 0;
            pushConstants.ne00 = K_KEY_COUNT;
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
            vkCmdDispatch(ctx.commandBuffer, 1, 1, 1);
        }};

    // Node 3: out = Vt . probs. Vt is [HSV, KV], so reduction is over KV.
    RenderGraphNodeCreateInfo outputPass = {
        .nodeName = "AttnOutput",
        .queueType = QueueType::COMPUTE,
        .resourceUses =
            {
                gemvUse("ValuesTransposed", ResourceIOType::READ, llama_mat_vec_bindings::A),
                gemvUse("Probabilities", ResourceIOType::READ, llama_mat_vec_bindings::B),
                gemvUse("AttnOutput", ResourceIOType::READ_WRITE, llama_mat_vec_bindings::D),
                gemvUse("Fuse0", ResourceIOType::READ, llama_mat_vec_bindings::FUSE0),
                gemvUse("Fuse1", ResourceIOType::READ, llama_mat_vec_bindings::FUSE1),
            },
        .shaderNames = {"mul_mat_vec_f32_f32_f32.comp"},
        .specializationConstants =
            {
                {.id = llama_mat_vec_spec_constants::BLOCK_SIZE, .value = K_BLOCK_SIZE},
                {.id = llama_mat_vec_spec_constants::NUM_ROWS, .value = 1},
                {.id = llama_mat_vec_spec_constants::NUM_COLS, .value = 1},
            },
        .execute = MakeGemvExecute(K_KEY_COUNT, K_HEAD_DIM)};

    builder.AddNode(scoresPass);
    builder.AddNode(weightsPass);
    builder.AddNode(outputPass);
    // Ordering is explicit: sharing a resource does not create an edge, so the chain is declared.
    builder.AddDependency(scoresPass.nodeName, weightsPass.nodeName);
    builder.AddDependency(weightsPass.nodeName, outputPass.nodeName);
    builder.Build();

    std::printf("[graph] execution order:");
    for (const std::string& nodeName : builder.GetExecutionOrder())
    {
        std::printf(" %s", nodeName.c_str());
    }
    std::printf("\n");

    auto* pQuery = GetRenderResourceManager()->GetResource<BufferResource>("Query");
    auto* pKeys = GetRenderResourceManager()->GetResource<BufferResource>("Keys");
    auto* pValues = GetRenderResourceManager()->GetResource<BufferResource>("ValuesTransposed");
    auto* pOutput = GetRenderResourceManager()->GetResource<BufferResource>("AttnOutput");
    if (pQuery == nullptr || pKeys == nullptr || pValues == nullptr || pOutput == nullptr)
    {
        std::fprintf(stderr, "[fail] the graph did not allocate the attention buffers\n");
        return 1;
    }
    pQuery->SetData(hostQuery.data(), hostQuery.size() * sizeof(float));
    pKeys->SetData(hostKeys.data(), hostKeys.size() * sizeof(float));
    pValues->SetData(hostValuesTransposed.data(), hostValuesTransposed.size() * sizeof(float));

    builder.Execute();

    // Reference: scores, softmax, weighted sum over V.
    std::vector<float> scores(K_KEY_COUNT);
    for (uint32_t key = 0; key < K_KEY_COUNT; ++key)
    {
        float dot = 0.0F;
        for (uint32_t d = 0; d < K_HEAD_DIM; ++d)
        {
            dot += hostKeys[static_cast<size_t>(key) * K_HEAD_DIM + d] * hostQuery[d];
        }
        scores[key] = dot * K_SCALE;
    }
    const float maxScore = *std::max_element(scores.begin(), scores.end());
    std::vector<float> probabilities(K_KEY_COUNT);
    double sumExp = 0.0;
    for (uint32_t key = 0; key < K_KEY_COUNT; ++key)
    {
        probabilities[key] = std::exp(scores[key] - maxScore);
        sumExp += probabilities[key];
    }
    for (uint32_t key = 0; key < K_KEY_COUNT; ++key)
    {
        probabilities[key] = static_cast<float>(probabilities[key] / sumExp);
    }

    const auto* pResult = static_cast<const float*>(pOutput->Map());
    uint32_t nWrong = 0;
    for (uint32_t d = 0; d < K_HEAD_DIM; ++d)
    {
        float expected = 0.0F;
        for (uint32_t key = 0; key < K_KEY_COUNT; ++key)
        {
            expected += probabilities[key] * hostValues[static_cast<size_t>(key) * K_HEAD_DIM + d];
        }
        if (std::fabs(pResult[d] - expected) > 1e-4F * (1.0F + std::fabs(expected)))
        {
            if (nWrong < 4)
            {
                std::fprintf(stderr, "[fail] out[%u] = %g, expected %g\n", d, pResult[d], expected);
            }
            ++nWrong;
        }
    }
    pOutput->Unmap();

    if (nWrong != 0)
    {
        std::fprintf(stderr, "[fail] %u of %u attention outputs wrong\n", nWrong, K_HEAD_DIM);
        return 1;
    }

    std::printf("[attention] OK: 3-node QK^T -> softmax -> V chain over %u keys matches the CPU reference\n",
                K_KEY_COUNT);
    return 0;
}
}  // namespace muyo_llm
