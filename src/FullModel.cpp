#include "FullModel.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ChatTemplate.h"
#include "Gguf.h"
#include "LlamaContracts.h"
#include "RenderGraph/RenderGraphBuilder.h"
#include "RenderGraph/RenderGraphNodeContext.h"
#include "RenderGraph/RenderGraphNodeResource.h"
#include "RenderGraph/RenderGraphResourceDesc.h"
#include "RenderResource.h"
#include "RenderResourceManager.h"
#include "Sampler.h"
#include "Tokenizer.h"
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
/// Cache length. The value GEMV reduces the **whole** cache row every step, because `mul_mat_vec`
/// hardcodes A's row stride to `ncols` (Finding F11): a short cache does not shrink the *work*,
/// only the score GEMV's dispatch. The transposing value append carries its position in
/// `misalign_offsets`' 16 bits, so the ceiling is 65535; 2048 is a few hundred MB of cache on this
/// model and leaves room for a prompt and a reply.
constexpr uint32_t K_MAX_SEQ = 2048;
constexpr uint32_t K_GEMV_BLOCK_SIZE = 32;
constexpr uint32_t K_SOFT_MAX_BLOCK_SIZE = 32;
constexpr uint32_t K_ARGMAX_BLOCK_SIZE = 256;

/// `exp(-1e30)` is zero in F32, so a pre-filled score tail never wins the softmax maximum and never
/// contributes to the normalizer. The tail is written once and never again: the score GEMV only
/// dispatches rows up to the current position.
constexpr float K_SCORE_MASK = -1e30F;

/// Qwen2's shape, read from the GGUF metadata.
struct Qwen2Config
{
    uint32_t layers = 0;
    uint32_t hidden = 0;
    uint32_t feedForward = 0;
    uint32_t heads = 0;
    uint32_t kvHeads = 0;
    uint32_t headDim = 0;
    uint32_t queryDim = 0;
    uint32_t kvDim = 0;
    uint32_t gqa = 0;
    uint32_t vocab = 0;
    float rmsEps = 0.0F;
    float ropeBase = 0.0F;
};

Qwen2Config ReadQwen2Config(const GgufReader& reader)
{
    Qwen2Config config;
    config.layers = static_cast<uint32_t>(reader.UintOr("qwen2.block_count", 0));
    config.hidden = static_cast<uint32_t>(reader.UintOr("qwen2.embedding_length", 0));
    config.feedForward = static_cast<uint32_t>(reader.UintOr("qwen2.feed_forward_length", 0));
    config.heads = static_cast<uint32_t>(reader.UintOr("qwen2.attention.head_count", 0));
    config.kvHeads = static_cast<uint32_t>(reader.UintOr("qwen2.attention.head_count_kv", 0));
    config.rmsEps = static_cast<float>(reader.FloatOr("qwen2.attention.layer_norm_rms_epsilon", 0.0));
    config.ropeBase = static_cast<float>(reader.FloatOr("qwen2.rope.freq_base", 10000.0));
    config.vocab = static_cast<uint32_t>(reader.StringArraySize("tokenizer.ggml.tokens"));
    if (config.hidden == 0 || config.heads == 0 || config.kvHeads == 0 || config.vocab == 0)
    {
        throw std::runtime_error("the model is missing the qwen2 dimensions or the token list");
    }
    config.headDim = config.hidden / config.heads;
    config.queryDim = config.heads * config.headDim;
    config.kvDim = config.kvHeads * config.headDim;
    config.gqa = config.heads / config.kvHeads;
    return config;
}

/// Resource name for one layer's tensor, e.g. `b7.wq`. A name per layer keeps every node's binding
/// unambiguous at Build time; the graph allocates one buffer each (it does not alias transients).
std::string LayerName(uint32_t layer, const char* suffix) { return "b" + std::to_string(layer) + "." + suffix; }

uint64_t ElementCount(const GgufTensorInfo& tensor)
{
    uint64_t count = 1;
    for (uint64_t dimension : tensor.dimensions)
    {
        count *= dimension;
    }
    return count;
}

/// Zero base for a resource use.
ResourceUse MakeUse(const std::string& handle, ResourceIOType io, uint32_t binding)
{
    return ResourceUse{.handle = ResourceHandle(handle),
                       .io = io,
                       .usage = ResourceUsage::STORAGE_BUFFER,
                       .kind = ResourceKind::BUFFER,
                       .descriptorBinding = DescriptorBinding{.set = 0, .binding = binding}};
}

/// `mul_mat_vec`'s full push-constant state. The dispatch is fixed unless `keysSoFar` is set, in
/// which case the x dimension is the number of cache keys written so far (`position + 1`).
struct MatVecSpec
{
    uint32_t ncols = 0;
    uint32_t rows = 0;  ///< Output rows; also `strideD`, which the shader bounds against.
    uint32_t strideA = 0;
    uint32_t strideB = 0;
    uint32_t batchStrideA = 0;
    uint32_t batchStrideB = 0;
    uint32_t batchStrideD = 0;
    uint32_t fusionFlags = 0;
    uint32_t ne02 = 0;
    uint32_t ne12 = 0;
    uint32_t broadcast2 = 0;
    uint32_t broadcast3 = 0;
    uint32_t dispatchX = 0;
    uint32_t dispatchY = 1;
    const uint32_t* keysSoFar = nullptr;  ///< When set, `dispatchX = *keysSoFar + 1` (keys written so far).
};

auto MatVecExecute(const MatVecSpec& spec)
{
    return [spec](RenderGraphNodeContext& ctx)
    {
        LlamaMatVecPushConstants pushConstants{};
        pushConstants.ncols = spec.ncols;
        pushConstants.strideA = spec.strideA == 0 ? spec.ncols : spec.strideA;
        pushConstants.strideB = spec.strideB == 0 ? spec.ncols : spec.strideB;
        pushConstants.strideD = spec.rows;
        pushConstants.batchStrideA = spec.batchStrideA;
        pushConstants.batchStrideB = spec.batchStrideB;
        pushConstants.batchStrideD = spec.batchStrideD;
        pushConstants.fusionFlags = spec.fusionFlags;
        pushConstants.baseWorkGroupY = 0;
        pushConstants.ne02 = spec.ne02;
        pushConstants.ne12 = spec.ne12;
        pushConstants.broadcast2 = spec.broadcast2;
        pushConstants.broadcast3 = spec.broadcast3;
        vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                           &pushConstants);
        const uint32_t x = spec.keysSoFar != nullptr ? *spec.keysSoFar + 1 : spec.dispatchX;
        vkCmdDispatch(ctx.commandBuffer, x, spec.dispatchY, 1);
    };
}

/// `rope_neox.comp` for `rows` heads of `headDim` channels each. When `position` is set, the
/// destination base is `*position * headDim` - the cache row of the current token.
struct RopeSpec
{
    uint32_t rows = 1;
    uint32_t headDim = 0;
    uint32_t inputRowStride = 0;     ///< `nb01`.
    uint32_t outputRowStride = 0;    ///< `nb11`.
    uint32_t destinationOffset = 0;  ///< `dOffset`, used when `position` is null.
    float freqBase = 10000.0F;
    const uint32_t* position = nullptr;
};

auto RopeExecute(const RopeSpec& spec)
{
    return [spec](RenderGraphNodeContext& ctx)
    {
        LlamaRopePushConstants pushConstants{};
        pushConstants.ropeMode = llama_rope_modes::NEOX;
        pushConstants.nrows = spec.rows;
        pushConstants.nDims = spec.headDim;
        pushConstants.freqScale = 1.0F;
        pushConstants.freqBase = spec.freqBase;
        pushConstants.attnFactor = 1.0F;
        pushConstants.thetaScale = std::pow(spec.freqBase, -2.0F / static_cast<float>(spec.headDim));
        pushConstants.ne00 = spec.headDim;
        pushConstants.ne01 = spec.rows;
        pushConstants.ne02 = 1;
        pushConstants.nb01 = spec.inputRowStride;
        pushConstants.nb11 = spec.outputRowStride;
        pushConstants.dOffset = spec.position != nullptr ? *spec.position * spec.headDim : spec.destinationOffset;
        vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                           &pushConstants);
        vkCmdDispatch(ctx.commandBuffer, spec.rows, 1, 1);
    };
}

/// `rms_norm.comp` over one `length`-channel row.
auto RmsNormExecute(uint32_t length, float epsilon)
{
    return [length, epsilon](RenderGraphNodeContext& ctx)
    {
        LlamaAddPushConstants pushConstants{};
        pushConstants.ne = length;
        pushConstants.ne00 = length;
        pushConstants.ne01 = 1;
        pushConstants.ne02 = 1;
        pushConstants.ne03 = 1;
        pushConstants.nb00 = 1;
        pushConstants.nb01 = length;
        pushConstants.nb02 = length;
        pushConstants.nb03 = length;
        pushConstants.ne10 = length;
        pushConstants.ne11 = 1;
        pushConstants.ne12 = 1;
        pushConstants.ne13 = 1;
        pushConstants.nb10 = 1;
        pushConstants.ne20 = length;
        pushConstants.ne21 = 1;
        pushConstants.ne22 = 1;
        pushConstants.ne23 = 1;
        pushConstants.nb20 = 1;
        pushConstants.nb21 = length;
        pushConstants.nb22 = length;
        pushConstants.nb23 = length;
        pushConstants.param1 = epsilon;
        vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                           &pushConstants);
        vkCmdDispatch(ctx.commandBuffer, 1, 1, 1);
    };
}

/// `add.comp` for two equal-length F32 operands.
auto AddExecute(uint32_t length)
{
    return [length](RenderGraphNodeContext& ctx)
    {
        LlamaAddPushConstants pushConstants{};
        pushConstants.ne = length;
        pushConstants.ne00 = length;
        pushConstants.ne01 = 1;
        pushConstants.ne02 = 1;
        pushConstants.ne03 = 1;
        pushConstants.nb00 = 1;
        pushConstants.nb01 = length;
        pushConstants.nb02 = length;
        pushConstants.nb03 = length;
        pushConstants.ne10 = length;
        pushConstants.ne11 = 1;
        pushConstants.ne12 = 1;
        pushConstants.ne13 = 1;
        pushConstants.nb10 = 1;
        pushConstants.nb11 = length;
        pushConstants.nb12 = length;
        pushConstants.nb13 = length;
        pushConstants.ne20 = length;
        pushConstants.ne21 = 1;
        pushConstants.ne22 = 1;
        pushConstants.ne23 = 1;
        pushConstants.nb20 = 1;
        pushConstants.nb21 = length;
        pushConstants.nb22 = length;
        pushConstants.nb23 = length;
        vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                           &pushConstants);
        vkCmdDispatch(ctx.commandBuffer, 1, (length + 511) / 512, 1);
    };
}

/// `swiglu.comp` in split mode (separate gate and up operands) over `count` elements.
auto SwigluExecute(uint32_t count)
{
    return [count](RenderGraphNodeContext& ctx)
    {
        LlamaGluPushConstants pushConstants{};
        pushConstants.N = count;
        pushConstants.ne00 = count;
        pushConstants.ne20 = count;
        pushConstants.mode = llama_glu_modes::SPLIT;
        pushConstants.nb00 = 1;
        pushConstants.nb01 = count;
        pushConstants.nb02 = count;
        pushConstants.nb03 = count;
        pushConstants.nb10 = 1;
        pushConstants.nb11 = count;
        pushConstants.nb12 = count;
        pushConstants.nb13 = count;
        pushConstants.nb20 = 1;
        pushConstants.nb21 = count;
        pushConstants.nb22 = count;
        pushConstants.nb23 = count;
        pushConstants.ne21 = 1;
        pushConstants.ne22 = 1;
        const auto fastDiv = [](uint32_t d, uint32_t& mp, uint32_t& l)
        {
            l = 0;
            while (l < 32 && (uint32_t{1} << l) < d)
            {
                ++l;
            }
            mp = static_cast<uint32_t>((uint64_t{1} << 32) * ((uint64_t{1} << l) - d) / d + 1);
        };
        fastDiv(pushConstants.ne22 * pushConstants.ne21 * pushConstants.ne20, pushConstants.ne2_012mp,
                pushConstants.ne2_012L);
        fastDiv(pushConstants.ne21 * pushConstants.ne20, pushConstants.ne2_01mp, pushConstants.ne2_01L);
        fastDiv(pushConstants.ne20, pushConstants.ne2_0mp, pushConstants.ne2_0L);
        vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                           &pushConstants);
        vkCmdDispatch(ctx.commandBuffer, 1, (count + 511) / 512, 1);
    };
}

/// `soft_max.comp` over `rows` rows of `columns` with no mask. `columns` is the full cache length,
/// so the pre-masked tail contributes `exp(-1e30 - max) = 0`.
auto SoftMaxExecute(uint32_t rows, uint32_t columns, float scale)
{
    return [rows, columns, scale](RenderGraphNodeContext& ctx)
    {
        LlamaSoftMaxPushConstants pushConstants{};
        pushConstants.KX = columns;
        pushConstants.KY = 0;
        pushConstants.ne00 = columns;
        pushConstants.ne01 = rows;
        pushConstants.ne02 = 1;
        pushConstants.ne12 = 1;
        pushConstants.ne13 = 1;
        pushConstants.scale = scale;
        pushConstants.nrowsX = rows;
        vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                           &pushConstants);
        vkCmdDispatch(ctx.commandBuffer, rows, 1, 1);
    };
}

/// `copy.comp`'s `vk_op_unary_push_constants`, with the fastdiv magic built the way
/// `init_pushconst_fastdiv` builds it (products **including** the innermost dimension - Finding F12).
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
static_assert(sizeof(CopyPushConstants) == 30 * sizeof(uint32_t),
              "CopyPushConstants no longer matches generic_unary_head.glsl's `parameter` block");

void InitFastDiv(uint32_t d, uint32_t& mp, uint32_t& l)
{
    l = 0;
    while (l < 32 && (uint32_t{1} << l) < d)
    {
        ++l;
    }
    mp = static_cast<uint32_t>((uint64_t{1} << 32) * ((uint64_t{1} << l) - d) / d + 1);
}

/// A transposing column append: `value[kvHead][channel]` -> `cache[kvHead][channel][position]`.
///
/// The source is the flat value vector. The destination is described as `[headDim][maxSeq][kvHeads]`
/// with strides `[maxSeq][headDim*maxSeq][0]`, so `dst_idx(idx) = (idx/headDim)*(headDim*maxSeq) +
/// (idx%headDim)*maxSeq`, and `misalign_offsets` carries the position.
auto CopyValueColumnExecute(uint32_t kvDim, uint32_t headDim, uint32_t maxSeq, const uint32_t* position)
{
    return [kvDim, headDim, maxSeq, position](RenderGraphNodeContext& ctx)
    {
        CopyPushConstants pushConstants{};
        pushConstants.ne = kvDim;
        pushConstants.ne00 = kvDim;
        pushConstants.ne01 = 1;
        pushConstants.ne02 = 1;
        pushConstants.ne03 = 1;
        pushConstants.nb00 = 1;
        pushConstants.ne10 = headDim;
        pushConstants.ne11 = maxSeq;
        pushConstants.ne12 = 1;
        pushConstants.ne13 = 1;
        pushConstants.nb10 = maxSeq;
        pushConstants.nb11 = headDim * maxSeq;
        pushConstants.misalignOffsets = *position;

        uint32_t l0 = 0;
        InitFastDiv(pushConstants.ne11, pushConstants.ne1_0mp, l0);
        uint32_t l1 = 0;
        InitFastDiv(pushConstants.ne11 * pushConstants.ne10, pushConstants.ne1_01mp, l1);
        uint32_t l2 = 0;
        InitFastDiv(pushConstants.ne12 * pushConstants.ne11 * pushConstants.ne10, pushConstants.ne1_012mp, l2);
        // ggml's `pack_fastdiv_L(l012, l01, l0)` packs the *outer* divisor's shift first.
        pushConstants.ne1_Ls = l2 | (l1 << 8) | (l0 << 16);
        uint32_t s0 = 0;
        InitFastDiv(pushConstants.ne00, pushConstants.ne0_0mp, s0);
        uint32_t s1 = 0;
        InitFastDiv(pushConstants.ne01 * pushConstants.ne00, pushConstants.ne0_01mp, s1);
        uint32_t s2 = 0;
        InitFastDiv(pushConstants.ne02 * pushConstants.ne01 * pushConstants.ne00, pushConstants.ne0_012mp, s2);
        pushConstants.ne0_Ls = s2 | (s1 << 8) | (s0 << 16);

        vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                           &pushConstants);
        vkCmdDispatch(ctx.commandBuffer, 1, 1, 1);
    };
}

/// `get_rows.comp`: gather one `embedDim`-channel row `indices[0]` out of a `[vocab][embedDim]`
/// weight, as an F32 row.
auto GetRowsExecute(uint32_t embedDim, uint32_t vocab)
{
    return [embedDim, vocab](RenderGraphNodeContext& ctx)
    {
        LlamaAddPushConstants pushConstants{};
        pushConstants.ne = embedDim * vocab;
        pushConstants.ne00 = embedDim;
        pushConstants.ne01 = vocab;
        pushConstants.ne02 = 1;
        pushConstants.ne03 = 1;
        pushConstants.nb00 = 1;
        pushConstants.nb01 = embedDim;
        pushConstants.nb02 = embedDim * vocab;
        pushConstants.nb03 = embedDim * vocab;
        pushConstants.ne10 = 1;
        pushConstants.ne11 = 1;
        pushConstants.ne12 = 1;
        pushConstants.ne13 = 1;
        pushConstants.nb10 = 1;
        pushConstants.nb11 = 1;
        pushConstants.nb12 = 1;
        pushConstants.nb13 = 1;
        pushConstants.ne20 = embedDim;
        pushConstants.ne21 = 1;
        pushConstants.ne22 = 1;
        pushConstants.ne23 = 1;
        pushConstants.nb20 = 1;
        pushConstants.nb21 = embedDim;
        pushConstants.nb22 = embedDim;
        pushConstants.nb23 = embedDim;
        vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                           &pushConstants);
        vkCmdDispatch(ctx.commandBuffer, (embedDim + 511) / 512, 1, 1);
    };
}

/// `argmax.comp` over the logit row, then a compute-to-host barrier so the caller can map the token.
auto ArgmaxExecute(uint32_t vocab)
{
    return [vocab](RenderGraphNodeContext& ctx)
    {
        LlamaGenericHeadPushConstants pushConstants{};
        pushConstants.KX = vocab;
        pushConstants.KY = 1;
        vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                           &pushConstants);
        vkCmdDispatch(ctx.commandBuffer, 1, 1, 1);

        VkMemoryBarrier2 toHost{};
        toHost.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        toHost.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        toHost.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        toHost.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        toHost.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        VkDependencyInfo dependencyInfo{};
        dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependencyInfo.memoryBarrierCount = 1;
        dependencyInfo.pMemoryBarriers = &toHost;
        vkCmdPipelineBarrier2(ctx.commandBuffer, &dependencyInfo);
    };
}
}  // namespace

int RunFullModel(const std::string& ggufPath, const ForwardOptions& options)
{
    try
    {
        const GgufReader reader(ggufPath);
        const Qwen2Config config = ReadQwen2Config(reader);
        const Tokenizer tokenizer(reader);
        // In chat mode the prompt is the user's raw text and the ChatML framing is applied later, so
        // it must not be tokenized as a completion here.
        const std::vector<int32_t> prompt =
            options.chat ? std::vector<int32_t>{}
                         : (options.prompt.empty() ? options.tokens : tokenizer.Encode(options.prompt));
        if (prompt.size() >= K_MAX_SEQ)
        {
            std::fprintf(stderr, "[fail] prompt is %zu tokens, the cache holds %u\n", prompt.size(), K_MAX_SEQ);
            return 1;
        }
        std::fprintf(stderr, "[forward] %s\n", ggufPath.c_str());
        std::fprintf(stderr, "[forward] layers=%u hidden=%u ffn=%u heads=%u kv_heads=%u head_dim=%u vocab=%u gqa=%u\n",
                     config.layers, config.hidden, config.feedForward, config.heads, config.kvHeads, config.headDim,
                     config.vocab, config.gqa);
        std::fprintf(stderr, "[forward] max_seq=%u prompt=%zu n_predict=%u\n", K_MAX_SEQ, prompt.size(),
                     options.nPredict);

        RenderGraphBuilder builder(GetRenderDevice());

        const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        // When MUYO_DUMP is set the intermediates are made host-visible so they can be mapped for a
        // comparison; the weights stay device-local either way.
        const bool hostVisible = std::getenv("MUYO_DUMP") != nullptr;
        const auto weightDesc = [](uint64_t count, uint64_t stride)
        {
            return BufferResourceDesc{.count = count,
                                      .stride = stride,
                                      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      .memoryProperties = VMA_MEMORY_USAGE_GPU_ONLY,
                                      .lifetime = ResourceLifetime::Transient};
        };
        const auto scratchDesc = [storage, hostVisible](uint64_t count, uint64_t stride, ResourceLifetime lifetime)
        {
            return BufferResourceDesc{
                .count = count,
                .stride = stride,
                .usage = storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .memoryProperties = hostVisible ? VMA_MEMORY_USAGE_CPU_TO_GPU : VMA_MEMORY_USAGE_GPU_ONLY,
                .lifetime = lifetime};
        };
        const auto hostDesc = [storage](uint64_t count, uint64_t stride)
        {
            return BufferResourceDesc{.count = count,
                                      .stride = stride,
                                      .usage = storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      .memoryProperties = VMA_MEMORY_USAGE_CPU_TO_GPU,
                                      .lifetime = ResourceLifetime::Transient};
        };

        // Weight resources, remembered so they can be uploaded after Build().
        std::vector<std::pair<std::string, std::string>> weightUploads;
        const auto addWeight = [&](const std::string& name, const std::string& tensor)
        {
            const GgufTensorInfo* info = reader.FindTensor(tensor);
            if (info == nullptr)
            {
                throw std::runtime_error("the model is missing tensor " + tensor);
            }
            const uint64_t stride = info->type == 1 ? 2 : 4;  // F16, F32
            builder.AddResource(name, weightDesc(ElementCount(*info), stride));
            weightUploads.emplace_back(name, tensor);
        };

        addWeight("token_embd", "token_embd.weight");
        addWeight("output", "output.weight");
        addWeight("output_norm", "output_norm.weight");

        for (uint32_t layer = 0; layer < config.layers; ++layer)
        {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            addWeight(LayerName(layer, "attn_norm"), prefix + "attn_norm.weight");
            addWeight(LayerName(layer, "wq"), prefix + "attn_q.weight");
            addWeight(LayerName(layer, "bq"), prefix + "attn_q.bias");
            addWeight(LayerName(layer, "wk"), prefix + "attn_k.weight");
            addWeight(LayerName(layer, "bk"), prefix + "attn_k.bias");
            addWeight(LayerName(layer, "wv"), prefix + "attn_v.weight");
            addWeight(LayerName(layer, "bv"), prefix + "attn_v.bias");
            addWeight(LayerName(layer, "wo"), prefix + "attn_output.weight");
            addWeight(LayerName(layer, "ffn_norm"), prefix + "ffn_norm.weight");
            addWeight(LayerName(layer, "wgate"), prefix + "ffn_gate.weight");
            addWeight(LayerName(layer, "wup"), prefix + "ffn_up.weight");
            addWeight(LayerName(layer, "wdown"), prefix + "ffn_down.weight");
        }

        // Global scratch and I/O.
        builder.AddResource("embedding", scratchDesc(config.hidden, sizeof(float), ResourceLifetime::Transient));
        builder.AddResource("final_norm", scratchDesc(config.hidden, sizeof(float), ResourceLifetime::Transient));
        builder.AddResource("logits", hostDesc(config.vocab, sizeof(float)));
        builder.AddResource("token_id", hostDesc(1, sizeof(int32_t)));
        builder.AddResource("next_token", hostDesc(1, sizeof(int32_t)));
        builder.AddResource("rope_pos", hostDesc(1, sizeof(int32_t)));
        builder.AddResource("partial", hostDesc(2, sizeof(float)));
        builder.AddResource("rope_scratch", scratchDesc(4, sizeof(float), ResourceLifetime::Transient));
        builder.AddResource("softmax_mask", scratchDesc(K_MAX_SEQ, sizeof(float), ResourceLifetime::Transient));
        builder.AddResource("softmax_sinks", scratchDesc(1, sizeof(float), ResourceLifetime::Transient));

        // Per-layer caches and intermediates.
        for (uint32_t layer = 0; layer < config.layers; ++layer)
        {
            builder.AddResource(LayerName(layer, "kcache"), scratchDesc(static_cast<uint64_t>(config.kvDim) * K_MAX_SEQ,
                                                                        sizeof(float), ResourceLifetime::Persistent));
            builder.AddResource(LayerName(layer, "vcache"), scratchDesc(static_cast<uint64_t>(config.kvDim) * K_MAX_SEQ,
                                                                        sizeof(float), ResourceLifetime::Persistent));
            builder.AddResource(LayerName(layer, "xn"),
                                scratchDesc(config.hidden, sizeof(float), ResourceLifetime::Transient));
            builder.AddResource(LayerName(layer, "kvec"),
                                scratchDesc(config.kvDim, sizeof(float), ResourceLifetime::Transient));
            builder.AddResource(LayerName(layer, "vvec"),
                                scratchDesc(config.kvDim, sizeof(float), ResourceLifetime::Transient));
            builder.AddResource(LayerName(layer, "q"),
                                scratchDesc(config.queryDim, sizeof(float), ResourceLifetime::Transient));
            builder.AddResource(LayerName(layer, "qr"),
                                scratchDesc(config.queryDim, sizeof(float), ResourceLifetime::Transient));
            builder.AddResource(LayerName(layer, "scores"), scratchDesc(static_cast<uint64_t>(config.heads) * K_MAX_SEQ,
                                                                        sizeof(float), ResourceLifetime::Transient));
            builder.AddResource(LayerName(layer, "probs"), scratchDesc(static_cast<uint64_t>(config.heads) * K_MAX_SEQ,
                                                                       sizeof(float), ResourceLifetime::Transient));
            builder.AddResource(LayerName(layer, "attn_out"),
                                scratchDesc(config.queryDim, sizeof(float), ResourceLifetime::Transient));
            builder.AddResource(LayerName(layer, "o"),
                                scratchDesc(config.hidden, sizeof(float), ResourceLifetime::Transient));
            builder.AddResource(LayerName(layer, "x1"),
                                scratchDesc(config.hidden, sizeof(float), ResourceLifetime::Transient));
            builder.AddResource(LayerName(layer, "xn2"),
                                scratchDesc(config.hidden, sizeof(float), ResourceLifetime::Transient));
            builder.AddResource(LayerName(layer, "gate"),
                                scratchDesc(config.feedForward, sizeof(float), ResourceLifetime::Transient));
            builder.AddResource(LayerName(layer, "up"),
                                scratchDesc(config.feedForward, sizeof(float), ResourceLifetime::Transient));
            builder.AddResource(LayerName(layer, "act"),
                                scratchDesc(config.feedForward, sizeof(float), ResourceLifetime::Transient));
            builder.AddResource(LayerName(layer, "mlp_out"),
                                scratchDesc(config.hidden, sizeof(float), ResourceLifetime::Transient));
            builder.AddResource(LayerName(layer, "out"),
                                scratchDesc(config.hidden, sizeof(float), ResourceLifetime::Transient));
        }

        const std::vector<SpecializationValue> gemvSpec = {
            {.id = llama_mat_vec_spec_constants::BLOCK_SIZE, .value = K_GEMV_BLOCK_SIZE},
            {.id = llama_mat_vec_spec_constants::NUM_ROWS, .value = 1},
            {.id = llama_mat_vec_spec_constants::NUM_COLS, .value = 1}};

        const auto gemvUses = [](const std::string& a, const std::string& b, const std::string& d,
                                 const std::string& fuse0, const std::string& fuse1)
        {
            return std::vector<ResourceUse>{MakeUse(a, ResourceIOType::READ, llama_mat_vec_bindings::A),
                                            MakeUse(b, ResourceIOType::READ, llama_mat_vec_bindings::B),
                                            MakeUse(d, ResourceIOType::READ_WRITE, llama_mat_vec_bindings::D),
                                            MakeUse(fuse0, ResourceIOType::READ, llama_mat_vec_bindings::FUSE0),
                                            MakeUse(fuse1, ResourceIOType::READ, llama_mat_vec_bindings::FUSE1)};
        };

        // The position of the token being decoded. Every callback that needs it captures this by
        // reference; the callbacks are re-invoked by each Execute(), so the value is current.
        uint32_t position = 0;

        const float attentionScale = 1.0F / std::sqrt(static_cast<float>(config.headDim));

        for (uint32_t layer = 0; layer < config.layers; ++layer)
        {
            const std::string xn = LayerName(layer, "xn");
            const std::string kvec = LayerName(layer, "kvec");
            const std::string vvec = LayerName(layer, "vvec");
            const std::string q = LayerName(layer, "q");
            const std::string qr = LayerName(layer, "qr");
            const std::string kcache = LayerName(layer, "kcache");
            const std::string vcache = LayerName(layer, "vcache");
            const std::string scores = LayerName(layer, "scores");
            const std::string probs = LayerName(layer, "probs");
            const std::string attnOut = LayerName(layer, "attn_out");
            const std::string o = LayerName(layer, "o");
            const std::string x1 = LayerName(layer, "x1");
            const std::string xn2 = LayerName(layer, "xn2");
            const std::string gate = LayerName(layer, "gate");
            const std::string up = LayerName(layer, "up");
            const std::string act = LayerName(layer, "act");
            const std::string mlpOut = LayerName(layer, "mlp_out");
            const std::string out = LayerName(layer, "out");
            const std::string input = layer == 0 ? "embedding" : LayerName(layer - 1, "out");

            const std::string preNorm = LayerName(layer, "pre_norm");
            const std::string qProj = LayerName(layer, "q_proj");
            const std::string kProj = LayerName(layer, "k_proj");
            const std::string vProj = LayerName(layer, "v_proj");
            const std::string qRope = LayerName(layer, "q_rope");
            const std::string kRope = LayerName(layer, "k_rope");
            const std::string vCopy = LayerName(layer, "v_copy");
            const std::string attnScores = LayerName(layer, "attn_scores");
            const std::string attnWeights = LayerName(layer, "attn_weights");
            const std::string attnOutput = LayerName(layer, "attn_output");
            const std::string oProj = LayerName(layer, "o_proj");
            const std::string residual1 = LayerName(layer, "residual1");
            const std::string postNorm = LayerName(layer, "post_norm");
            const std::string gateProj = LayerName(layer, "gate_proj");
            const std::string upProj = LayerName(layer, "up_proj");
            const std::string swiglu = LayerName(layer, "swiglu");
            const std::string downProj = LayerName(layer, "down_proj");
            const std::string residual2 = LayerName(layer, "residual2");

            builder.AddNode(
                {.nodeName = preNorm,
                 .queueType = QueueType::COMPUTE,
                 .resourceUses = {MakeUse(input, ResourceIOType::READ, llama_add_bindings::A),
                                  MakeUse(LayerName(layer, "attn_norm"), ResourceIOType::READ, llama_add_bindings::B),
                                  MakeUse(xn, ResourceIOType::READ_WRITE, llama_add_bindings::D)},
                 .shaderNames = {"rms_norm_f32.comp"},
                 .specializationConstants = {{.id = 1, .value = 1}},
                 .execute = RmsNormExecute(config.hidden, config.rmsEps)});

            const auto projection = [&](const std::string& node, const std::string& weight, const std::string& bias,
                                        const std::string& destination, uint32_t rows)
            {
                MatVecSpec spec;
                spec.ncols = config.hidden;
                spec.rows = rows;
                spec.strideA = config.hidden;
                spec.strideB = config.hidden;
                spec.dispatchX = rows;
                spec.fusionFlags = llama_mat_vec_fusion::BIAS0;
                builder.AddNode({.nodeName = node,
                                 .queueType = QueueType::COMPUTE,
                                 .resourceUses = gemvUses(weight, xn, destination, bias, "partial"),
                                 .shaderNames = {"mul_mat_vec_f16_f32_f32.comp"},
                                 .specializationConstants = gemvSpec,
                                 .execute = MatVecExecute(spec)});
            };
            projection(qProj, LayerName(layer, "wq"), LayerName(layer, "bq"), q, config.queryDim);
            projection(kProj, LayerName(layer, "wk"), LayerName(layer, "bk"), kvec, config.kvDim);
            projection(vProj, LayerName(layer, "wv"), LayerName(layer, "bv"), vvec, config.kvDim);

            builder.AddNode({.nodeName = qRope,
                             .queueType = QueueType::COMPUTE,
                             .resourceUses = {MakeUse(q, ResourceIOType::READ, llama_rope_bindings::X),
                                              MakeUse("rope_pos", ResourceIOType::READ, llama_rope_bindings::Y),
                                              MakeUse("rope_scratch", ResourceIOType::READ, llama_rope_bindings::Z),
                                              MakeUse(qr, ResourceIOType::READ_WRITE, llama_rope_bindings::D),
                                              MakeUse("rope_scratch", ResourceIOType::READ, llama_rope_bindings::I)},
                             .shaderNames = {"rope_neox_f32.comp"},
                             .execute = RopeExecute(RopeSpec{.rows = config.heads,
                                                             .headDim = config.headDim,
                                                             .inputRowStride = config.headDim,
                                                             .outputRowStride = config.headDim,
                                                             .freqBase = config.ropeBase})});

            builder.AddNode({.nodeName = kRope,
                             .queueType = QueueType::COMPUTE,
                             .resourceUses = {MakeUse(kvec, ResourceIOType::READ, llama_rope_bindings::X),
                                              MakeUse("rope_pos", ResourceIOType::READ, llama_rope_bindings::Y),
                                              MakeUse("rope_scratch", ResourceIOType::READ, llama_rope_bindings::Z),
                                              MakeUse(kcache, ResourceIOType::READ_WRITE, llama_rope_bindings::D),
                                              MakeUse("rope_scratch", ResourceIOType::READ, llama_rope_bindings::I)},
                             .shaderNames = {"rope_neox_f32.comp"},
                             .execute = RopeExecute(RopeSpec{.rows = config.kvHeads,
                                                             .headDim = config.headDim,
                                                             .inputRowStride = config.headDim,
                                                             .outputRowStride = config.headDim * K_MAX_SEQ,
                                                             .freqBase = config.ropeBase,
                                                             .position = &position})});

            builder.AddNode({.nodeName = vCopy,
                             .queueType = QueueType::COMPUTE,
                             .resourceUses = {MakeUse(vvec, ResourceIOType::READ, 0),
                                              MakeUse(vcache, ResourceIOType::READ_WRITE, 1)},
                             .shaderNames = {"cpy_f32_f32.comp"},
                             .execute = CopyValueColumnExecute(config.kvDim, config.headDim, K_MAX_SEQ, &position)});

            MatVecSpec scoresSpec;
            scoresSpec.ncols = config.headDim;
            scoresSpec.rows = K_MAX_SEQ;
            scoresSpec.strideA = config.headDim;
            scoresSpec.strideB = config.headDim;
            scoresSpec.batchStrideA = config.headDim * K_MAX_SEQ;
            scoresSpec.batchStrideB = config.headDim;
            scoresSpec.batchStrideD = K_MAX_SEQ;
            scoresSpec.ne02 = 1;
            scoresSpec.ne12 = config.gqa;
            scoresSpec.broadcast2 = config.gqa;
            scoresSpec.broadcast3 = 1;
            scoresSpec.dispatchY = config.heads;
            scoresSpec.keysSoFar = &position;  // Dispatch grows with the cache.
            builder.AddNode({.nodeName = attnScores,
                             .queueType = QueueType::COMPUTE,
                             .resourceUses = gemvUses(kcache, qr, scores, "partial", "partial"),
                             .shaderNames = {"mul_mat_vec_f32_f32_f32.comp"},
                             .specializationConstants = gemvSpec,
                             .execute = MatVecExecute(scoresSpec)});

            builder.AddNode(
                {.nodeName = attnWeights,
                 .queueType = QueueType::COMPUTE,
                 .resourceUses = {MakeUse(scores, ResourceIOType::READ, llama_soft_max_bindings::X),
                                  MakeUse("softmax_mask", ResourceIOType::READ, llama_soft_max_bindings::Y),
                                  MakeUse("softmax_sinks", ResourceIOType::READ, llama_soft_max_bindings::Z),
                                  MakeUse(probs, ResourceIOType::READ_WRITE, llama_soft_max_bindings::D)},
                 .shaderNames = {"soft_max_f32.comp"},
                 .specializationConstants = {{.id = llama_soft_max_spec_constants::BLOCK_SIZE,
                                              .value = K_SOFT_MAX_BLOCK_SIZE}},
                 .execute = SoftMaxExecute(config.heads, K_MAX_SEQ, attentionScale)});

            MatVecSpec valueSpec;
            valueSpec.ncols = K_MAX_SEQ;
            valueSpec.rows = config.headDim;
            valueSpec.strideA = K_MAX_SEQ;
            valueSpec.strideB = K_MAX_SEQ;
            valueSpec.batchStrideA = config.headDim * K_MAX_SEQ;
            valueSpec.batchStrideB = K_MAX_SEQ;
            valueSpec.batchStrideD = config.headDim;
            valueSpec.ne02 = 1;
            valueSpec.ne12 = config.gqa;
            valueSpec.broadcast2 = config.gqa;
            valueSpec.broadcast3 = 1;
            valueSpec.dispatchX = config.headDim;
            valueSpec.dispatchY = config.heads;
            builder.AddNode({.nodeName = attnOutput,
                             .queueType = QueueType::COMPUTE,
                             .resourceUses = gemvUses(vcache, probs, attnOut, "partial", "partial"),
                             .shaderNames = {"mul_mat_vec_f32_f32_f32.comp"},
                             .specializationConstants = gemvSpec,
                             .execute = MatVecExecute(valueSpec)});

            MatVecSpec outputSpec;
            outputSpec.ncols = config.queryDim;
            outputSpec.rows = config.hidden;
            outputSpec.strideA = config.queryDim;
            outputSpec.strideB = config.queryDim;
            outputSpec.dispatchX = config.hidden;
            builder.AddNode({.nodeName = oProj,
                             .queueType = QueueType::COMPUTE,
                             .resourceUses = gemvUses(LayerName(layer, "wo"), attnOut, o, "partial", "partial"),
                             .shaderNames = {"mul_mat_vec_f16_f32_f32.comp"},
                             .specializationConstants = gemvSpec,
                             .execute = MatVecExecute(outputSpec)});

            builder.AddNode(
                {.nodeName = residual1,
                 .queueType = QueueType::COMPUTE,
                 .resourceUses = {MakeUse(input, ResourceIOType::READ, llama_add_bindings::A),
                                  MakeUse(o, ResourceIOType::READ, llama_add_bindings::B),
                                  MakeUse(x1, ResourceIOType::READ_WRITE, llama_add_bindings::D),
                                  MakeUse("partial", ResourceIOType::READ_WRITE, llama_add_bindings::PARTIAL_SUMS)},
                 .shaderNames = {"add_f32.comp"},
                 .specializationConstants = {{.id = llama_add_spec_constants::NOREPEAT, .value = 1}},
                 .execute = AddExecute(config.hidden)});

            builder.AddNode(
                {.nodeName = postNorm,
                 .queueType = QueueType::COMPUTE,
                 .resourceUses = {MakeUse(x1, ResourceIOType::READ, llama_add_bindings::A),
                                  MakeUse(LayerName(layer, "ffn_norm"), ResourceIOType::READ, llama_add_bindings::B),
                                  MakeUse(xn2, ResourceIOType::READ_WRITE, llama_add_bindings::D)},
                 .shaderNames = {"rms_norm_f32.comp"},
                 .specializationConstants = {{.id = 1, .value = 1}},
                 .execute = RmsNormExecute(config.hidden, config.rmsEps)});

            MatVecSpec mlpSpec;
            mlpSpec.ncols = config.hidden;
            mlpSpec.rows = config.feedForward;
            mlpSpec.strideA = config.hidden;
            mlpSpec.strideB = config.hidden;
            mlpSpec.dispatchX = config.feedForward;
            builder.AddNode({.nodeName = gateProj,
                             .queueType = QueueType::COMPUTE,
                             .resourceUses = gemvUses(LayerName(layer, "wgate"), xn2, gate, "partial", "partial"),
                             .shaderNames = {"mul_mat_vec_f16_f32_f32.comp"},
                             .specializationConstants = gemvSpec,
                             .execute = MatVecExecute(mlpSpec)});
            builder.AddNode({.nodeName = upProj,
                             .queueType = QueueType::COMPUTE,
                             .resourceUses = gemvUses(LayerName(layer, "wup"), xn2, up, "partial", "partial"),
                             .shaderNames = {"mul_mat_vec_f16_f32_f32.comp"},
                             .specializationConstants = gemvSpec,
                             .execute = MatVecExecute(mlpSpec)});

            builder.AddNode({.nodeName = swiglu,
                             .queueType = QueueType::COMPUTE,
                             .resourceUses = {MakeUse(gate, ResourceIOType::READ, llama_glu_bindings::A),
                                              MakeUse(up, ResourceIOType::READ, llama_glu_bindings::B),
                                              MakeUse(act, ResourceIOType::READ_WRITE, llama_glu_bindings::D)},
                             .shaderNames = {"swiglu_f32.comp"},
                             .execute = SwigluExecute(config.feedForward)});

            MatVecSpec downSpec;
            downSpec.ncols = config.feedForward;
            downSpec.rows = config.hidden;
            downSpec.strideA = config.feedForward;
            downSpec.strideB = config.feedForward;
            downSpec.dispatchX = config.hidden;
            builder.AddNode({.nodeName = downProj,
                             .queueType = QueueType::COMPUTE,
                             .resourceUses = gemvUses(LayerName(layer, "wdown"), act, mlpOut, "partial", "partial"),
                             .shaderNames = {"mul_mat_vec_f16_f32_f32.comp"},
                             .specializationConstants = gemvSpec,
                             .execute = MatVecExecute(downSpec)});

            builder.AddNode(
                {.nodeName = residual2,
                 .queueType = QueueType::COMPUTE,
                 .resourceUses = {MakeUse(x1, ResourceIOType::READ, llama_add_bindings::A),
                                  MakeUse(mlpOut, ResourceIOType::READ, llama_add_bindings::B),
                                  MakeUse(out, ResourceIOType::READ_WRITE, llama_add_bindings::D),
                                  MakeUse("partial", ResourceIOType::READ_WRITE, llama_add_bindings::PARTIAL_SUMS)},
                 .shaderNames = {"add_f32.comp"},
                 .specializationConstants = {{.id = llama_add_spec_constants::NOREPEAT, .value = 1}},
                 .execute = AddExecute(config.hidden)});

            builder.AddDependency(preNorm, qProj);
            builder.AddDependency(preNorm, kProj);
            builder.AddDependency(preNorm, vProj);
            builder.AddDependency(qProj, qRope);
            builder.AddDependency(kProj, kRope);
            builder.AddDependency(vProj, vCopy);
            builder.AddDependency(qRope, attnScores);
            builder.AddDependency(kRope, attnScores);
            builder.AddDependency(vCopy, attnScores);
            builder.AddDependency(attnScores, attnWeights);
            builder.AddDependency(attnWeights, attnOutput);
            builder.AddDependency(attnOutput, oProj);
            builder.AddDependency(oProj, residual1);
            builder.AddDependency(residual1, postNorm);
            builder.AddDependency(postNorm, gateProj);
            builder.AddDependency(postNorm, upProj);
            builder.AddDependency(gateProj, swiglu);
            builder.AddDependency(upProj, swiglu);
            builder.AddDependency(swiglu, downProj);
            builder.AddDependency(downProj, residual2);
            if (layer > 0)
            {
                builder.AddDependency(LayerName(layer - 1, "residual2"), preNorm);
            }
        }

        builder.AddNode({.nodeName = "embed",
                         .queueType = QueueType::COMPUTE,
                         .resourceUses = {MakeUse("token_embd", ResourceIOType::READ, llama_add_bindings::A),
                                          MakeUse("token_id", ResourceIOType::READ, llama_add_bindings::B),
                                          MakeUse("embedding", ResourceIOType::READ_WRITE, llama_add_bindings::D)},
                         .shaderNames = {"get_rows_f16_f32.comp"},
                         .execute = GetRowsExecute(config.hidden, config.vocab)});

        builder.AddNode(
            {.nodeName = "final_norm",
             .queueType = QueueType::COMPUTE,
             .resourceUses = {MakeUse(LayerName(config.layers - 1, "out"), ResourceIOType::READ, llama_add_bindings::A),
                              MakeUse("output_norm", ResourceIOType::READ, llama_add_bindings::B),
                              MakeUse("final_norm", ResourceIOType::READ_WRITE, llama_add_bindings::D)},
             .shaderNames = {"rms_norm_f32.comp"},
             .specializationConstants = {{.id = 1, .value = 1}},
             .execute = RmsNormExecute(config.hidden, config.rmsEps)});

        MatVecSpec lmSpec;
        lmSpec.ncols = config.hidden;
        lmSpec.rows = config.vocab;
        lmSpec.strideA = config.hidden;
        lmSpec.strideB = config.hidden;
        lmSpec.dispatchX = config.vocab;
        builder.AddNode({.nodeName = "lm_head",
                         .queueType = QueueType::COMPUTE,
                         .resourceUses = gemvUses("output", "final_norm", "logits", "partial", "partial"),
                         .shaderNames = {"mul_mat_vec_f16_f32_f32.comp"},
                         .specializationConstants = gemvSpec,
                         .execute = MatVecExecute(lmSpec)});

        builder.AddNode(
            {.nodeName = "argmax",
             .queueType = QueueType::COMPUTE,
             .resourceUses = {MakeUse("logits", ResourceIOType::READ, llama_argmax_bindings::A),
                              MakeUse("next_token", ResourceIOType::READ_WRITE, llama_argmax_bindings::D)},
             .shaderNames = {"argmax_f32.comp"},
             .specializationConstants = {{.id = llama_argmax_spec_constants::BLOCK_SIZE, .value = K_ARGMAX_BLOCK_SIZE}},
             .execute = ArgmaxExecute(config.vocab)});

        builder.AddDependency("embed", LayerName(0, "pre_norm"));
        builder.AddDependency(LayerName(config.layers - 1, "residual2"), "final_norm");
        builder.AddDependency("final_norm", "lm_head");
        builder.AddDependency("lm_head", "argmax");

        builder.Build();

        std::fprintf(stderr, "[graph] %zu nodes, %zu weight tensors\n", builder.GetExecutionOrder().size(),
                     weightUploads.size());

        // Upload. The weights are GPU-only, so SetData stages and copies each one.
        for (const auto& [name, tensor] : weightUploads)
        {
            const std::vector<uint8_t> bytes = reader.ReadTensorData(tensor);
            auto* resource = GetRenderResourceManager()->GetResource<BufferResource>(name);
            if (resource == nullptr)
            {
                std::fprintf(stderr, "[fail] the graph did not allocate %s\n", name.c_str());
                return 1;
            }
            resource->SetData(bytes.data(), bytes.size());
        }

        const std::vector<float> zeroCache(static_cast<size_t>(config.kvDim) * K_MAX_SEQ, 0.0F);
        const std::vector<float> maskedScores(static_cast<size_t>(config.heads) * K_MAX_SEQ, K_SCORE_MASK);
        for (uint32_t layer = 0; layer < config.layers; ++layer)
        {
            GetRenderResourceManager()
                ->GetResource<BufferResource>(LayerName(layer, "kcache"))
                ->SetData(zeroCache.data(), zeroCache.size() * sizeof(float));
            GetRenderResourceManager()
                ->GetResource<BufferResource>(LayerName(layer, "vcache"))
                ->SetData(zeroCache.data(), zeroCache.size() * sizeof(float));
            GetRenderResourceManager()
                ->GetResource<BufferResource>(LayerName(layer, "scores"))
                ->SetData(maskedScores.data(), maskedScores.size() * sizeof(float));
        }

        auto* pTokenId = GetRenderResourceManager()->GetResource<BufferResource>("token_id");
        auto* pRopePos = GetRenderResourceManager()->GetResource<BufferResource>("rope_pos");
        auto* pLogitsResource = GetRenderResourceManager()->GetResource<BufferResource>("logits");
        bool wroteLogits = false;

        const int32_t eos = static_cast<int32_t>(reader.UintOr("tokenizer.ggml.eos_token_id", UINT64_MAX));
        const auto startTime = std::chrono::steady_clock::now();

        Sampler sampler(options.sampler);
        std::vector<int32_t> fed;        // every token in the cache, in order
        std::vector<int32_t> generated;  // the reply of the current turn
        std::string pending;             // bytes held back until a UTF-8 character completes

        // Streaming output: a token is a byte piece, so a UTF-8 character can span two of them.
        // Hold back an incomplete trailing sequence rather than print a replacement character.
        const auto flushText = [&pending](bool final)
        {
            size_t complete = 0;
            while (complete < pending.size())
            {
                const auto lead = static_cast<unsigned char>(pending[complete]);
                size_t length = 1;
                if ((lead & 0xE0) == 0xC0)
                {
                    length = 2;
                }
                else if ((lead & 0xF0) == 0xE0)
                {
                    length = 3;
                }
                else if ((lead & 0xF8) == 0xF0)
                {
                    length = 4;
                }
                if (!final && complete + length > pending.size())
                {
                    break;
                }
                complete += length;
            }
            std::fwrite(pending.data(), 1, complete, stdout);
            pending.erase(0, complete);
        };

        // Feed one token at the next cache position and re-record the graph, which refreshes the
        // logits. False when the cache is full.
        const auto feedToken = [&](int32_t id) -> bool
        {
            if (fed.size() >= K_MAX_SEQ)
            {
                return false;
            }
            const uint32_t pos = static_cast<uint32_t>(fed.size());
            position = pos;
            pTokenId->SetData(&id, sizeof(int32_t));
            pRopePos->SetData(&pos, sizeof(int32_t));
            builder.Execute();
            fed.push_back(id);
            return true;
        };

        const auto sampleNext = [&]()
        {
            auto* pLogits = static_cast<float*>(pLogitsResource->Map());
            const int32_t next = sampler.Sample(pLogits, static_cast<int32_t>(config.vocab), fed);
            pLogitsResource->Unmap();
            return next;
        };

        // First-step dumps, gated by env vars (see README).
        const auto dumpFirstStep = [&]()
        {
            if (const char* dumpDir = std::getenv("MUYO_DUMP"))
            {
                const std::vector<std::pair<std::string, uint32_t>> dumpList = {{"embedding", config.hidden},
                                                                                {"b0.xn", config.hidden},
                                                                                {"b0.q", config.queryDim},
                                                                                {"b0.qr", config.queryDim},
                                                                                {"b0.kvec", config.kvDim},
                                                                                {"b0.vvec", config.kvDim},
                                                                                {"b0.vcache", config.kvDim * K_MAX_SEQ},
                                                                                {"b0.scores", config.heads * K_MAX_SEQ},
                                                                                {"b0.probs", config.heads * K_MAX_SEQ},
                                                                                {"b0.attn_out", config.queryDim},
                                                                                {"b0.o", config.hidden},
                                                                                {"b0.x1", config.hidden},
                                                                                {"b0.gate", config.feedForward},
                                                                                {"b0.up", config.feedForward},
                                                                                {"b0.act", config.feedForward},
                                                                                {"b0.mlp_out", config.hidden},
                                                                                {"b0.out", config.hidden},
                                                                                {"final_norm", config.hidden},
                                                                                {"logits", config.vocab}};
                for (const auto& [name, count] : dumpList)
                {
                    auto* resource = GetRenderResourceManager()->GetResource<BufferResource>(name);
                    if (resource == nullptr)
                    {
                        continue;
                    }
                    const auto* values = static_cast<const float*>(resource->Map());
                    const std::string file = std::string(dumpDir) + "/" + name + ".bin";
                    std::ofstream out(file, std::ios::binary);
                    out.write(reinterpret_cast<const char*>(values), sizeof(float) * count);
                    resource->Unmap();
                }
                std::fprintf(stderr, "[forward] dumped intermediates to %s\n", dumpDir);
            }
            if (const char* dumpPath = std::getenv("MUYO_LOGITS"))
            {
                const auto* pLogits = static_cast<const float*>(pLogitsResource->Map());
                std::ofstream out(dumpPath, std::ios::binary);
                const int32_t n = static_cast<int32_t>(config.vocab);
                out.write(reinterpret_cast<const char*>(&n), sizeof(n));
                out.write(reinterpret_cast<const char*>(pLogits), sizeof(float) * config.vocab);
                pLogitsResource->Unmap();
                std::fprintf(stderr, "[forward] wrote logits to %s\n", dumpPath);
            }
        };

        // Feed `tokens`, then generate up to `budget` tokens, streaming the text. EOS is fed too, so
        // a chat turn leaves the cache ending at the assistant's `<|im_end|>` and the next turn can
        // be appended without re-prefilling.
        const auto runTurn = [&](const std::vector<int32_t>& tokens, uint32_t budget)
        {
            generated.clear();
            for (int32_t id : tokens)
            {
                if (!feedToken(id))
                {
                    std::fprintf(stderr, "[forward] the context is full (%u tokens)\n", K_MAX_SEQ);
                    return;
                }
            }
            for (uint32_t n = 0; n < budget; ++n)
            {
                if (!wroteLogits)
                {
                    dumpFirstStep();
                    wroteLogits = true;
                }
                const int32_t next = sampleNext();
                generated.push_back(next);
                if (!tokenizer.IsSpecial(next))
                {
                    pending += tokenizer.TokenBytes(next);
                    flushText(false);
                    std::fflush(stdout);
                }
                if (!feedToken(next))
                {
                    break;
                }
                if (eos != static_cast<int32_t>(UINT64_MAX) && next == eos)
                {
                    break;
                }
            }
        };

        if (options.chat)
        {
            const std::string system = options.system.empty() ? chat_template::QWEN2_DEFAULT_SYSTEM : options.system;
            // With no initial prompt in an interactive session, prime only the system turn and let
            // the first line from stdin be the first user turn.
            bool firstUserTurn = true;
            if (!options.prompt.empty() || !options.interactive)
            {
                const std::string user = options.prompt.empty() ? "Hello" : options.prompt;
                runTurn(tokenizer.Encode(chat_template::Qwen2ChatStart(system, user), true), options.nPredict);
                firstUserTurn = false;
            }
            else
            {
                for (int32_t id : tokenizer.Encode(chat_template::Qwen2SystemStart(system), true))
                {
                    feedToken(id);
                }
            }
            while (options.interactive)
            {
                std::fprintf(stderr, "\n[chat] you> ");
                std::fflush(stderr);
                std::string line;
                if (!std::getline(std::cin, line))
                {
                    break;
                }
                if (line.empty())
                {
                    continue;
                }
                if (line == "/exit" || line == "/quit")
                {
                    break;
                }
                // Every turn after the assistant's `<|im_end|>` starts with a newline.
                const std::string prefix = firstUserTurn ? "" : "\n";
                firstUserTurn = false;
                std::printf("\n");
                runTurn(tokenizer.Encode(prefix + chat_template::Qwen2UserTurn(line), true), options.nPredict);
            }
        }
        else
        {
            std::vector<int32_t> promptTokens = prompt;
            if (promptTokens.empty())
            {
                promptTokens.push_back(static_cast<int32_t>(reader.UintOr("tokenizer.ggml.bos_token_id", 0)));
            }
            std::fprintf(stderr, "[forward] prompt tokens:");
            for (int32_t token : promptTokens)
            {
                std::fprintf(stderr, " %d", token);
            }
            std::fprintf(stderr, "\n");
            runTurn(promptTokens, options.nPredict);
        }

        flushText(true);
        std::printf("\n");
        std::fflush(stdout);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
        std::fprintf(stderr, "[forward] generated %zu tokens:", generated.size());
        for (int32_t token : generated)
        {
            std::fprintf(stderr, " %d", token);
        }
        std::fprintf(stderr, "\n");
        if (seconds > 0.0)
        {
            std::fprintf(stderr, "[forward] %.2f tok/s (%.0f ms total)\n",
                         static_cast<double>(generated.size()) / seconds, seconds * 1000.0);
        }
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::fprintf(stderr, "[fail] %s\n", exception.what());
        return 1;
    }
}
}  // namespace muyo_llm
