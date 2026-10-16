#include "KvCache.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "GraphServices.h"
#include "LlamaContracts.h"
#include "RenderGraph/RenderGraphBuilder.h"
#include "RenderGraph/RenderGraphNodeContext.h"
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
constexpr uint32_t K_KV_DIM = 16;  // One head, so the key/value row is head_dim wide.
constexpr uint32_t K_MAX_SEQ = 4;  // Cache capacity.
constexpr uint32_t K_TOKENS = 2;   // Positions 0 and 1 are filled.
constexpr uint32_t K_BLOCK_SIZE = 32;
constexpr float K_FREQ_BASE = 10000.0F;

/// `copy.comp`'s push-constant block (`generic_unary_head.glsl`), 29 four-byte words.
struct CopyPushConstants
{
    uint32_t ne;
    uint32_t ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03;
    uint32_t ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13;
    uint32_t misalignOffsets;
    float param1, param2, param3, param4;
    uint32_t ne0_012mp, ne0_01mp, ne0_0mp, ne0_Ls;
    uint32_t ne1_012mp, ne1_01mp, ne1_0mp, ne1_Ls;
};

/// 30 four-byte words, matching `vk_op_unary_push_constants` in ggml-vulkan-push-constants.h.
static_assert(sizeof(CopyPushConstants) == 30 * sizeof(uint32_t),
              "CopyPushConstants no longer matches generic_unary_head.glsl's `parameter` block");

/// ggml's `pack_fastdiv_L`: three shift amounts packed into one word.
uint32_t PackL(uint32_t l0, uint32_t l1, uint32_t l2) { return l0 | (l1 << 8) | (l2 << 16); }

/// The packed-L form `vk_op_unary_push_constants` carries: three shift amounts in one word.
uint32_t PackFastDivL(uint32_t d, uint32_t& mp)
{
    uint32_t L = 0;
    while (L < 32 && (uint32_t{1} << L) < d)
    {
        ++L;
    }
    mp = static_cast<uint32_t>((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
    return L;
}

/// The same rotation `rope_neox` applies, for the CPU reference.
std::vector<float> ReferenceRope(const std::vector<float>& key, uint32_t position)
{
    std::vector<float> out(key.size());
    const float thetaScale = std::pow(K_FREQ_BASE, -2.0F / static_cast<float>(K_KV_DIM));
    for (uint32_t j = 0; j < K_KV_DIM / 2; ++j)
    {
        const float theta = static_cast<float>(position) * std::pow(thetaScale, static_cast<float>(j));
        const float c = std::cos(theta);
        const float s = std::sin(theta);
        out[j] = key[j] * c - key[j + K_KV_DIM / 2] * s;
        out[j + K_KV_DIM / 2] = key[j] * s + key[j + K_KV_DIM / 2] * c;
    }
    return out;
}
}  // namespace

int RunKvCache()
{
    // Query is fixed across tokens; keys and values differ per token. Small integers so the score
    // and the weighted sum are exact, and only the softmax introduces tolerance.
    std::vector<float> query(K_KV_DIM);
    for (uint32_t d = 0; d < K_KV_DIM; ++d)
    {
        query[d] = static_cast<float>(static_cast<int>(d % 5) - 2);
    }

    uint32_t position = 0;  ///< Updated between executions; read by the record-time callbacks.

    RenderGraphBuilder builder(GraphServices());
    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const auto buffer = [&](uint32_t count, uint32_t stride, VkBufferUsageFlags usage, ResourceLifetime lifetime)
    {
        return BufferResourceDesc{.count = count,
                                  .stride = stride,
                                  .usage = usage,
                                  .memoryProperties = VMA_MEMORY_USAGE_CPU_TO_GPU,
                                  .lifetime = lifetime};
    };

    // The caches outlive the graph's executions: this is the "long-lived resource" case, and the
    // reason the append matters at all.
    builder.AddResource("KeyCache", buffer(K_MAX_SEQ * K_KV_DIM, sizeof(float),
                                           storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, ResourceLifetime::Persistent));
    builder.AddResource("ValueCacheT",
                        buffer(K_KV_DIM * K_MAX_SEQ, sizeof(float), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               ResourceLifetime::Persistent));
    builder.AddResource("KeyVector", buffer(K_KV_DIM, sizeof(float), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                            ResourceLifetime::Persistent));
    builder.AddResource("ValueVector", buffer(K_KV_DIM, sizeof(float), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                              ResourceLifetime::Persistent));
    builder.AddResource("Query", buffer(K_KV_DIM, sizeof(float), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        ResourceLifetime::Persistent));
    builder.AddResource("Scores", buffer(K_MAX_SEQ, sizeof(float), storage, ResourceLifetime::Persistent));
    builder.AddResource("Probs", buffer(K_MAX_SEQ, sizeof(float), storage, ResourceLifetime::Persistent));
    builder.AddResource("AttnOut", buffer(K_KV_DIM, sizeof(float), storage, ResourceLifetime::Persistent));
    builder.AddResource("SoftMaxMask", buffer(K_MAX_SEQ, sizeof(float), storage, ResourceLifetime::Persistent));
    builder.AddResource("SoftMaxSinks", buffer(1, sizeof(float), storage, ResourceLifetime::Persistent));
    builder.AddResource("CopyScratch", buffer(K_KV_DIM, sizeof(float), storage, ResourceLifetime::Persistent));
    builder.AddResource("RopePositions", buffer(1, sizeof(int32_t), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                ResourceLifetime::Persistent));

    const auto binding = [](uint32_t index) { return DescriptorBinding{.set = 0, .binding = index}; };
    const auto use = [&binding](const char* handle, ResourceIOType io, uint32_t index)
    {
        return ResourceUse{.handle = ResourceHandle(handle),
                           .io = io,
                           .usage = ResourceUsage::STORAGE_BUFFER,
                           .kind = ResourceKind::BUFFER,
                           .descriptorBinding = binding(index)};
    };

    // Node 1: rope the key **into the cache row** for the current position. rope's destination base
    // is a full 32-bit push-constant field, so `d_offset = position * kvDim` addresses the row.
    RenderGraphNodeCreateInfo ropeKeyPass = {
        .nodeName = "RopeKeyIntoCache",
        .queueType = QueueType::COMPUTE,
        .resourceUses = {use("KeyVector", ResourceIOType::READ, llama_rope_bindings::X),
                         use("RopePositions", ResourceIOType::READ, llama_rope_bindings::Y),
                         use("CopyScratch", ResourceIOType::READ, llama_rope_bindings::Z),
                         use("KeyCache", ResourceIOType::READ_WRITE, llama_rope_bindings::D),
                         use("CopyScratch", ResourceIOType::READ, llama_rope_bindings::I)},
        .shaderNames = {"rope_neox_f32.comp"},
        .execute = [&position](RenderGraphNodeContext& ctx)
        {
            LlamaRopePushConstants pushConstants{};
            pushConstants.ropeMode = llama_rope_modes::NEOX;
            pushConstants.nrows = 1;
            pushConstants.nDims = K_KV_DIM;
            pushConstants.freqScale = 1.0F;
            pushConstants.freqBase = K_FREQ_BASE;
            pushConstants.attnFactor = 1.0F;
            pushConstants.thetaScale = std::pow(K_FREQ_BASE, -2.0F / static_cast<float>(K_KV_DIM));
            pushConstants.ne00 = K_KV_DIM;
            pushConstants.ne01 = 1;
            pushConstants.ne02 = 1;
            pushConstants.nb01 = K_KV_DIM;
            pushConstants.nb11 = K_KV_DIM;
            pushConstants.dOffset = position * K_KV_DIM;  // The cache row for this token.
            vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(pushConstants), &pushConstants);
            vkCmdDispatch(ctx.commandBuffer, 1, 1, 1);
        }};

    // Node 2: scatter the value into the **transposed** cache column, with `copy.comp`. The
    // destination is [kvDim, seq] (channel-major), so `dst_idx = position + channel * seq` and the
    // position rides in `misalign_offsets`' low 16 bits.
    RenderGraphNodeCreateInfo copyValuePass = {
        .nodeName = "CopyValueIntoCache",
        .queueType = QueueType::COMPUTE,
        .resourceUses = {use("ValueVector", ResourceIOType::READ, 0),
                         use("ValueCacheT", ResourceIOType::READ_WRITE, 1)},
        .shaderNames = {"cpy_f32_f32.comp"},
        .execute = [&position](RenderGraphNodeContext& ctx)
        {
            // `vk_op_unary_push_constants`, matching generic_unary_head.glsl. The destination is
            // [channels][positions] with the channel stride = seq, the position stride = 1; the
            // source is 1-D. `misalign_offsets` carries the destination row in its low 16 bits.
            CopyPushConstants pushConstants{};
            pushConstants.ne = K_KV_DIM;
            pushConstants.ne00 = K_KV_DIM;
            pushConstants.ne01 = 1;
            pushConstants.ne02 = 1;
            pushConstants.ne03 = 1;
            pushConstants.nb00 = 1;
            pushConstants.ne10 = K_KV_DIM;   // Destination rows = channels.
            pushConstants.ne11 = K_MAX_SEQ;  // Destination columns = positions.
            pushConstants.ne12 = 1;
            pushConstants.ne13 = 1;
            pushConstants.nb10 = K_MAX_SEQ;  // Channel stride in the transposed cache.
            pushConstants.nb11 = 1;          // Position stride.
            pushConstants.misalignOffsets = position;
            // Both halves carry the shift amounts packed into one word (`ne*_Ls`), because
            // generic_unary_head.glsl packs the three L values to stay under 128 bytes. The
            // divisors are the products *including* the innermost dimension, matching
            // `init_pushconst_fastdiv(vk_op_unary_push_constants&)` in ggml-vulkan-push-constants.h.
            const uint32_t l12 =
                PackFastDivL(pushConstants.ne12 * pushConstants.ne11 * pushConstants.ne10, pushConstants.ne1_012mp);
            const uint32_t l1 = PackFastDivL(pushConstants.ne11 * pushConstants.ne10, pushConstants.ne1_01mp);
            const uint32_t l0 = PackFastDivL(pushConstants.ne10, pushConstants.ne1_0mp);
            pushConstants.ne1_Ls = PackL(l12, l1, l0);
            const uint32_t s12 =
                PackFastDivL(pushConstants.ne02 * pushConstants.ne01 * pushConstants.ne00, pushConstants.ne0_012mp);
            const uint32_t s1 = PackFastDivL(pushConstants.ne01 * pushConstants.ne00, pushConstants.ne0_01mp);
            const uint32_t s0 = PackFastDivL(pushConstants.ne00, pushConstants.ne0_0mp);
            pushConstants.ne0_Ls = PackL(s12, s1, s0);
            vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(pushConstants), &pushConstants);
            vkCmdDispatch(ctx.commandBuffer, 1, 1, 1);
        }};

    // Node 3: scores = KeyCache[0..position] . query. The reduction and the dispatch are limited to
    // the rows written so far, which is what makes the cache "grow".
    RenderGraphNodeCreateInfo scoresPass = {
        .nodeName = "AttnScores",
        .queueType = QueueType::COMPUTE,
        .resourceUses = {use("KeyCache", ResourceIOType::READ, llama_mat_vec_bindings::A),
                         use("Query", ResourceIOType::READ, llama_mat_vec_bindings::B),
                         use("Scores", ResourceIOType::READ_WRITE, llama_mat_vec_bindings::D),
                         use("CopyScratch", ResourceIOType::READ, llama_mat_vec_bindings::FUSE0),
                         use("CopyScratch", ResourceIOType::READ, llama_mat_vec_bindings::FUSE1)},
        .shaderNames = {"mul_mat_vec_f32_f32_f32.comp"},
        .specializationConstants =
            {
                {.id = llama_mat_vec_spec_constants::BLOCK_SIZE, .value = K_BLOCK_SIZE},
                {.id = llama_mat_vec_spec_constants::NUM_ROWS, .value = 1},
                {.id = llama_mat_vec_spec_constants::NUM_COLS, .value = 1},
            },
        .execute = [&position](RenderGraphNodeContext& ctx)
        {
            const uint32_t keysSoFar = position + 1;
            LlamaMatVecPushConstants pushConstants{};
            pushConstants.ncols = K_KV_DIM;
            pushConstants.strideA = K_KV_DIM;
            pushConstants.strideB = K_KV_DIM;
            pushConstants.strideD = keysSoFar;
            vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(pushConstants), &pushConstants);
            vkCmdDispatch(ctx.commandBuffer, keysSoFar, 1, 1);
        }};

    // Node 4: probs = softmax(scores).
    RenderGraphNodeCreateInfo weightsPass = {
        .nodeName = "AttnWeights",
        .queueType = QueueType::COMPUTE,
        .resourceUses = {use("Scores", ResourceIOType::READ, llama_soft_max_bindings::X),
                         use("SoftMaxMask", ResourceIOType::READ, llama_soft_max_bindings::Y),
                         use("SoftMaxSinks", ResourceIOType::READ, llama_soft_max_bindings::Z),
                         use("Probs", ResourceIOType::READ_WRITE, llama_soft_max_bindings::D)},
        .shaderNames = {"soft_max_f32.comp"},
        .specializationConstants = {{.id = llama_soft_max_spec_constants::BLOCK_SIZE, .value = K_BLOCK_SIZE}},
        .execute = [&position](RenderGraphNodeContext& ctx)
        {
            LlamaSoftMaxPushConstants pushConstants{};
            pushConstants.KX = position + 1;
            pushConstants.ne00 = position + 1;
            pushConstants.ne01 = 1;
            pushConstants.ne02 = 1;
            pushConstants.scale = 1.0F / std::sqrt(static_cast<float>(K_KV_DIM));
            pushConstants.nrowsX = 1;
            vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(pushConstants), &pushConstants);
            vkCmdDispatch(ctx.commandBuffer, 1, 1, 1);
        }};

    // Node 5: out = ValueCacheT . probs, reduced over the keys written so far.
    RenderGraphNodeCreateInfo outputPass = {
        .nodeName = "AttnOut",
        .queueType = QueueType::COMPUTE,
        .resourceUses = {use("ValueCacheT", ResourceIOType::READ, llama_mat_vec_bindings::A),
                         use("Probs", ResourceIOType::READ, llama_mat_vec_bindings::B),
                         use("AttnOut", ResourceIOType::READ_WRITE, llama_mat_vec_bindings::D),
                         use("CopyScratch", ResourceIOType::READ, llama_mat_vec_bindings::FUSE0),
                         use("CopyScratch", ResourceIOType::READ, llama_mat_vec_bindings::FUSE1)},
        .shaderNames = {"mul_mat_vec_f32_f32_f32.comp"},
        .specializationConstants =
            {
                {.id = llama_mat_vec_spec_constants::BLOCK_SIZE, .value = K_BLOCK_SIZE},
                {.id = llama_mat_vec_spec_constants::NUM_ROWS, .value = 1},
                {.id = llama_mat_vec_spec_constants::NUM_COLS, .value = 1},
            },
        .execute = [&position](RenderGraphNodeContext& ctx)
        {
            // `mul_mat_vec` hardcodes A's row stride to `ncols` (`ibi = first_row*p.ncols` in the
            // shader), so A must be row-contiguous: the reduction must span the whole cache row
            // (seq columns), not just the keys written so far. Columns past `position` are zero in
            // ValueCacheT, so they contribute nothing regardless of the probabilities.
            LlamaMatVecPushConstants pushConstants{};
            pushConstants.ncols = K_MAX_SEQ;
            pushConstants.strideA = K_MAX_SEQ;
            pushConstants.strideB = K_MAX_SEQ;
            pushConstants.strideD = K_KV_DIM;
            vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(pushConstants), &pushConstants);
            vkCmdDispatch(ctx.commandBuffer, K_KV_DIM, 1, 1);
        }};

    builder.AddNode(ropeKeyPass);
    builder.AddNode(copyValuePass);
    builder.AddNode(scoresPass);
    builder.AddNode(weightsPass);
    builder.AddNode(outputPass);
    builder.AddDependency(ropeKeyPass.nodeName, scoresPass.nodeName);
    builder.AddDependency(copyValuePass.nodeName, scoresPass.nodeName);
    builder.AddDependency(scoresPass.nodeName, weightsPass.nodeName);
    builder.AddDependency(weightsPass.nodeName, outputPass.nodeName);
    builder.Build();

    std::printf("[graph] execution order:");
    for (const std::string& nodeName : builder.GetExecutionOrder())
    {
        std::printf(" %s", nodeName.c_str());
    }
    std::printf("\n");

    auto* pKeyCache = GetRenderResourceManager()->GetResource<BufferResource>("KeyCache");
    auto* pValueCache = GetRenderResourceManager()->GetResource<BufferResource>("ValueCacheT");
    auto* pKey = GetRenderResourceManager()->GetResource<BufferResource>("KeyVector");
    auto* pValue = GetRenderResourceManager()->GetResource<BufferResource>("ValueVector");
    auto* pRopePos = GetRenderResourceManager()->GetResource<BufferResource>("RopePositions");
    auto* pQuery = GetRenderResourceManager()->GetResource<BufferResource>("Query");
    auto* pOut = GetRenderResourceManager()->GetResource<BufferResource>("AttnOut");

    std::vector<float> hostQuery(K_KV_DIM);
    for (uint32_t d = 0; d < K_KV_DIM; ++d)
    {
        hostQuery[d] = query[d];
    }
    pQuery->SetData(hostQuery.data(), hostQuery.size() * sizeof(float));
    const std::vector<int32_t> hostRopePos = {0};
    pRopePos->SetData(hostRopePos.data(), hostRopePos.size() * sizeof(int32_t));
    // Zero the caches once; the appends then fill them in place across executions.
    std::vector<float> zeros(K_MAX_SEQ * K_KV_DIM, 0.0F);
    pKeyCache->SetData(zeros.data(), zeros.size() * sizeof(float));
    pValueCache->SetData(zeros.data(), zeros.size() * sizeof(float));

    uint32_t nWrongTotal = 0;
    std::vector<std::vector<float>> allKeys;
    std::vector<std::vector<float>> allValues;
    for (uint32_t token = 0; token < K_TOKENS; ++token)
    {
        std::vector<float> key(K_KV_DIM);
        std::vector<float> value(K_KV_DIM);
        for (uint32_t d = 0; d < K_KV_DIM; ++d)
        {
            key[d] = static_cast<float>(static_cast<int>((token * 3 + d) % 5) - 2);
            value[d] = static_cast<float>(static_cast<int>((token + d) % 5) - 2);
        }
        pKey->SetData(key.data(), key.size() * sizeof(float));
        pValue->SetData(value.data(), value.size() * sizeof(float));
        allKeys.push_back(key);
        allValues.push_back(value);

        position = token;  // Read by every record-time callback.
        builder.Execute();

        // Reference: attention over the cache grown to `token + 1` keys. Each cached key is the
        // key that was appended at its own position and roped at that position; likewise the cached
        // value at column `j` is the value appended then, so the reference must replay them.
        std::vector<float> expected(K_KV_DIM, 0.0F);
        std::vector<double> weights(token + 1);
        double sumExp = 0.0;
        for (uint32_t j = 0; j <= token; ++j)
        {
            const std::vector<float> ropedKey = ReferenceRope(allKeys[j], j);
            float dot = 0.0F;
            for (uint32_t d = 0; d < K_KV_DIM; ++d)
            {
                dot += ropedKey[d] * query[d];
            }
            weights[j] = std::exp(dot / std::sqrt(static_cast<float>(K_KV_DIM)));
            sumExp += weights[j];
        }
        for (uint32_t j = 0; j <= token; ++j)
        {
            weights[j] /= sumExp;
        }
        for (uint32_t j = 0; j <= token; ++j)
        {
            for (uint32_t d = 0; d < K_KV_DIM; ++d)
            {
                expected[d] += static_cast<float>(weights[j]) * allValues[j][d];
            }
        }

        const auto* result = static_cast<const float*>(pOut->Map());
        uint32_t nWrong = 0;
        for (uint32_t d = 0; d < K_KV_DIM; ++d)
        {
            if (std::fabs(result[d] - expected[d]) > 1e-4F * (1.0F + std::fabs(expected[d])))
            {
                if (nWrong < 4)
                {
                    std::fprintf(stderr, "[fail] token %u out[%u] = %g, expected %g\n", token, d, result[d],
                                 expected[d]);
                }
                ++nWrong;
            }
        }
        pOut->Unmap();
        if (nWrong != 0)
        {
            std::fprintf(stderr, "[fail] token %u: %u of %u outputs wrong\n", token, nWrong, K_KV_DIM);
            nWrongTotal += nWrong;
        }
        else
        {
            std::printf("[kvcache] OK: token %u at position %u over %u cached keys matches the CPU reference\n", token,
                        position, token + 1);
        }
    }

    return nWrongTotal == 0 ? 0 : 1;
}
}  // namespace muyo_llm
