#include "TransformerLayer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
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
constexpr uint32_t K_HIDDEN = 64;
constexpr uint32_t K_SEQ = 8;  // Cached keys/values, current token included.
constexpr uint32_t K_INTERMEDIATE = 128;
constexpr uint32_t K_BLOCK_SIZE = 32;
constexpr float K_EPSILON = 1e-6F;
constexpr float K_ATTN_SCALE = 1.0F / 8.0F;  // 1 / sqrt(K_HIDDEN)

constexpr uint32_t K_FUSE_SIZE = K_INTERMEDIATE;  // Largest output row count any node binds.

// ── Per-op execute callbacks ─────────────────────────────────────────────────

/// `mul_mat_vec`: `rows` output rows, each a dot product over `reduction`.
auto GemvExecute(uint32_t reduction, uint32_t rows)
{
    return [reduction, rows](RenderGraphNodeContext& ctx)
    {
        LlamaMatVecPushConstants pushConstants{};
        pushConstants.ncols = reduction;
        pushConstants.strideA = reduction;
        pushConstants.strideB = reduction;
        pushConstants.strideD = rows;
        pushConstants.ne02 = 1;
        pushConstants.ne12 = 1;
        pushConstants.broadcast2 = 1;
        pushConstants.broadcast3 = 1;
        vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                           &pushConstants);
        vkCmdDispatch(ctx.commandBuffer, rows, 1, 1);
    };
}

/// `rms_norm` for a single `length`-element row.
auto RmsNormExecute(uint32_t length)
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
        pushConstants.ne20 = length;
        pushConstants.ne21 = 1;
        pushConstants.ne22 = 1;
        pushConstants.ne23 = 1;
        pushConstants.nb20 = 1;
        pushConstants.nb21 = length;
        pushConstants.nb22 = length;
        pushConstants.nb23 = length;
        pushConstants.param1 = K_EPSILON;
        vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                           &pushConstants);
        vkCmdDispatch(ctx.commandBuffer, 1, 1, 1);
    };
}

/// `shaders/add.comp` for two equal-length operands (broadcast `norepeat = true`).
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

/// `swiglu` in split mode over N elements.
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
        auto fastDiv = [](uint32_t d, uint32_t& mp, uint32_t& L)
        {
            L = 0;
            while (L < 32 && (uint32_t{1} << L) < d)
            {
                ++L;
            }
            mp = static_cast<uint32_t>((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
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

/// `soft_max` without a mask over `columns`.
auto SoftMaxExecute(uint32_t columns)
{
    return [columns](RenderGraphNodeContext& ctx)
    {
        LlamaSoftMaxPushConstants pushConstants{};
        pushConstants.KX = columns;
        pushConstants.ne00 = columns;
        pushConstants.ne01 = 1;
        pushConstants.ne02 = 1;
        pushConstants.scale = K_ATTN_SCALE;
        pushConstants.nrowsX = 1;
        vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                           &pushConstants);
        vkCmdDispatch(ctx.commandBuffer, 1, 1, 1);
    };
}

// ── CPU helpers ──────────────────────────────────────────────────────────────

std::vector<float> Gemv(const std::vector<float>& weights, const std::vector<float>& input, uint32_t rows,
                        uint32_t reduction)
{
    std::vector<float> output(rows);
    for (uint32_t row = 0; row < rows; ++row)
    {
        float sum = 0.0F;
        for (uint32_t k = 0; k < reduction; ++k)
        {
            sum += weights[static_cast<size_t>(row) * reduction + k] * input[k];
        }
        output[row] = sum;
    }
    return output;
}

std::vector<float> RmsNorm(const std::vector<float>& input, const std::vector<float>& weight)
{
    double sumSquares = 0.0;
    for (float value : input)
    {
        sumSquares += static_cast<double>(value) * value;
    }
    const float scale = static_cast<float>(1.0 / std::sqrt(sumSquares / input.size() + K_EPSILON));
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i)
    {
        output[i] = input[i] * scale * weight[i];
    }
    return output;
}
}  // namespace

int RunTransformerLayer()
{
    // Small integers so the projections and residuals are exact; only the softmax and rsqrt
    // introduce tolerance.
    std::vector<float> hostHidden(K_HIDDEN);
    std::vector<float> hostNorm1(K_HIDDEN);
    std::vector<float> hostNorm2(K_HIDDEN);
    std::vector<float> hostWq(K_HIDDEN * K_HIDDEN);
    std::vector<float> hostWo(K_HIDDEN * K_HIDDEN);
    std::vector<float> hostWgate(K_INTERMEDIATE * K_HIDDEN);
    std::vector<float> hostWup(K_INTERMEDIATE * K_HIDDEN);
    std::vector<float> hostWdown(K_HIDDEN * K_INTERMEDIATE);
    std::vector<float> hostKeys(K_SEQ * K_HIDDEN);
    std::vector<float> hostValuesTransposed(K_HIDDEN * K_SEQ);

    for (uint32_t i = 0; i < K_HIDDEN; ++i)
    {
        hostHidden[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
        hostNorm1[i] = 0.5F + 0.01F * static_cast<float>(i % 5);
        hostNorm2[i] = 0.6F + 0.01F * static_cast<float>(i % 3);
    }
    for (uint32_t i = 0; i < K_HIDDEN * K_HIDDEN; ++i)
    {
        hostWq[i] = static_cast<float>(static_cast<int>(i % 3) - 1);
        hostWo[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.5F;
    }
    for (uint32_t i = 0; i < K_INTERMEDIATE * K_HIDDEN; ++i)
    {
        hostWgate[i] = static_cast<float>(static_cast<int>(i % 3) - 1);
        hostWup[i] = static_cast<float>(static_cast<int>(i % 4) - 1);
    }
    for (uint32_t i = 0; i < K_HIDDEN * K_INTERMEDIATE; ++i)
    {
        hostWdown[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.25F;
    }
    for (uint32_t key = 0; key < K_SEQ; ++key)
    {
        for (uint32_t d = 0; d < K_HIDDEN; ++d)
        {
            hostKeys[static_cast<size_t>(key) * K_HIDDEN + d] = static_cast<float>(static_cast<int>((key + d) % 3) - 1);
            // Transposed: value channel `d` of key `key` at [d][key].
            hostValuesTransposed[static_cast<size_t>(d) * K_SEQ + key] =
                static_cast<float>(static_cast<int>((key * 2 + d) % 7) - 3);
        }
    }

    RenderGraphBuilder builder(GraphServices());
    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkBufferUsageFlags inputUsage = storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const auto buffer = [](uint32_t count, VkBufferUsageFlags usage)
    {
        return BufferResourceDesc{.count = count,
                                  .stride = sizeof(float),
                                  .usage = usage,
                                  .memoryProperties = VMA_MEMORY_USAGE_CPU_TO_GPU,
                                  .lifetime = ResourceLifetime::Transient};
    };
    const auto addInput = [&](const char* name, uint32_t count)
    { builder.AddResource(name, buffer(count, inputUsage)); };
    const auto addOutput = [&](const char* name, uint32_t count) { builder.AddResource(name, buffer(count, storage)); };

    addInput("Hidden", K_HIDDEN);
    addInput("Norm1", K_HIDDEN);
    addInput("Norm2", K_HIDDEN);
    addInput("Wq", K_HIDDEN * K_HIDDEN);
    addInput("Wo", K_HIDDEN * K_HIDDEN);
    addInput("Wgate", K_INTERMEDIATE * K_HIDDEN);
    addInput("Wup", K_INTERMEDIATE * K_HIDDEN);
    addInput("Wdown", K_HIDDEN * K_INTERMEDIATE);
    addInput("KeyCache", K_SEQ * K_HIDDEN);
    addInput("ValueCacheT", K_HIDDEN * K_SEQ);
    // The RoPE position for the current token, read as `int` by rope_head.glsl.
    builder.AddResource("RopePositions", BufferResourceDesc{.count = 1,
                                                            .stride = sizeof(int32_t),
                                                            .usage = inputUsage,
                                                            .memoryProperties = VMA_MEMORY_USAGE_CPU_TO_GPU,
                                                            .lifetime = ResourceLifetime::Transient});

    addOutput("Xn", K_HIDDEN);
    addOutput("Q", K_HIDDEN);
    addOutput("Qr", K_HIDDEN);
    addOutput("Scores", K_SEQ);
    addOutput("Probs", K_SEQ);
    addOutput("AttnOut", K_HIDDEN);
    addOutput("O", K_HIDDEN);
    addOutput("X1", K_HIDDEN);
    addOutput("Xn2", K_HIDDEN);
    addOutput("Gate", K_INTERMEDIATE);
    addOutput("Up", K_INTERMEDIATE);
    addOutput("Act", K_INTERMEDIATE);
    addOutput("MlpOut", K_HIDDEN);
    addOutput("LayerOut", K_HIDDEN);
    addOutput("Fuse0", K_FUSE_SIZE);
    addOutput("Fuse1", K_FUSE_SIZE);
    addOutput("Partial", 2);
    addOutput("SoftMaxMask", K_SEQ);
    addOutput("SoftMaxSinks", 1);

    const auto binding = [](uint32_t index) { return DescriptorBinding{.set = 0, .binding = index}; };
    const auto use = [&binding](const char* handle, ResourceIOType io, uint32_t index)
    {
        return ResourceUse{.handle = ResourceHandle(handle),
                           .io = io,
                           .usage = ResourceUsage::STORAGE_BUFFER,
                           .kind = ResourceKind::BUFFER,
                           .descriptorBinding = binding(index)};
    };
    const auto gemvUses = [&use](const char* a, const char* b, const char* d)
    {
        return std::vector<ResourceUse>{use(a, ResourceIOType::READ, llama_mat_vec_bindings::A),
                                        use(b, ResourceIOType::READ, llama_mat_vec_bindings::B),
                                        use(d, ResourceIOType::READ_WRITE, llama_mat_vec_bindings::D),
                                        use("Fuse0", ResourceIOType::READ, llama_mat_vec_bindings::FUSE0),
                                        use("Fuse1", ResourceIOType::READ, llama_mat_vec_bindings::FUSE1)};
    };
    const std::vector<SpecializationValue> gemvSpec = {
        {.id = llama_mat_vec_spec_constants::BLOCK_SIZE, .value = K_BLOCK_SIZE},
        {.id = llama_mat_vec_spec_constants::NUM_ROWS, .value = 1},
        {.id = llama_mat_vec_spec_constants::NUM_COLS, .value = 1}};

    builder.AddNode({.nodeName = "PreNorm",
                     .queueType = QueueType::COMPUTE,
                     .resourceUses = {use("Hidden", ResourceIOType::READ, llama_add_bindings::A),
                                      use("Norm1", ResourceIOType::READ, llama_add_bindings::B),
                                      use("Xn", ResourceIOType::READ_WRITE, llama_add_bindings::D)},
                     .shaderNames = {"rms_norm_f32.comp"},
                     .specializationConstants = {{.id = 1, .value = 1}},
                     .execute = RmsNormExecute(K_HIDDEN)});

    builder.AddNode({.nodeName = "QProj",
                     .queueType = QueueType::COMPUTE,
                     .resourceUses = gemvUses("Wq", "Xn", "Q"),
                     .shaderNames = {"mul_mat_vec_f32_f32_f32.comp"},
                     .specializationConstants = gemvSpec,
                     .execute = GemvExecute(K_HIDDEN, K_HIDDEN)});

    builder.AddNode({.nodeName = "QRope",
                     .queueType = QueueType::COMPUTE,
                     .resourceUses = {use("Q", ResourceIOType::READ, llama_rope_bindings::X),
                                      use("RopePositions", ResourceIOType::READ, llama_rope_bindings::Y),
                                      use("Partial", ResourceIOType::READ, llama_rope_bindings::Z),
                                      use("Qr", ResourceIOType::READ_WRITE, llama_rope_bindings::D),
                                      use("Partial", ResourceIOType::READ, llama_rope_bindings::I)},
                     .shaderNames = {"rope_neox_f32.comp"},
                     .execute = [](RenderGraphNodeContext& ctx)
                     {
                         LlamaRopePushConstants pushConstants{};
                         pushConstants.ropeMode = llama_rope_modes::NEOX;
                         pushConstants.nrows = 1;
                         pushConstants.nDims = K_HIDDEN;
                         pushConstants.freqScale = 1.0F;
                         pushConstants.freqBase = 10000.0F;
                         pushConstants.attnFactor = 1.0F;
                         pushConstants.thetaScale = std::pow(10000.0F, -2.0F / static_cast<float>(K_HIDDEN));
                         pushConstants.ne00 = K_HIDDEN;
                         pushConstants.ne01 = 1;
                         pushConstants.ne02 = 1;
                         pushConstants.nb01 = K_HIDDEN;
                         pushConstants.nb11 = K_HIDDEN;
                         vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                            sizeof(pushConstants), &pushConstants);
                         vkCmdDispatch(ctx.commandBuffer, 1, 1, 1);
                     }});

    builder.AddNode({.nodeName = "AttnScores",
                     .queueType = QueueType::COMPUTE,
                     .resourceUses = gemvUses("KeyCache", "Qr", "Scores"),
                     .shaderNames = {"mul_mat_vec_f32_f32_f32.comp"},
                     .specializationConstants = gemvSpec,
                     .execute = GemvExecute(K_HIDDEN, K_SEQ)});

    builder.AddNode(
        {.nodeName = "AttnWeights",
         .queueType = QueueType::COMPUTE,
         .resourceUses = {use("Scores", ResourceIOType::READ, llama_soft_max_bindings::X),
                          use("SoftMaxMask", ResourceIOType::READ, llama_soft_max_bindings::Y),
                          use("SoftMaxSinks", ResourceIOType::READ, llama_soft_max_bindings::Z),
                          use("Probs", ResourceIOType::READ_WRITE, llama_soft_max_bindings::D)},
         .shaderNames = {"soft_max_f32.comp"},
         .specializationConstants = {{.id = llama_soft_max_spec_constants::BLOCK_SIZE, .value = K_BLOCK_SIZE}},
         .execute = SoftMaxExecute(K_SEQ)});

    builder.AddNode({.nodeName = "AttnOut",
                     .queueType = QueueType::COMPUTE,
                     .resourceUses = gemvUses("ValueCacheT", "Probs", "AttnOut"),
                     .shaderNames = {"mul_mat_vec_f32_f32_f32.comp"},
                     .specializationConstants = gemvSpec,
                     .execute = GemvExecute(K_SEQ, K_HIDDEN)});

    builder.AddNode({.nodeName = "OProj",
                     .queueType = QueueType::COMPUTE,
                     .resourceUses = gemvUses("Wo", "AttnOut", "O"),
                     .shaderNames = {"mul_mat_vec_f32_f32_f32.comp"},
                     .specializationConstants = gemvSpec,
                     .execute = GemvExecute(K_HIDDEN, K_HIDDEN)});

    builder.AddNode({.nodeName = "Residual1",
                     .queueType = QueueType::COMPUTE,
                     .resourceUses = {use("Hidden", ResourceIOType::READ, llama_add_bindings::A),
                                      use("O", ResourceIOType::READ, llama_add_bindings::B),
                                      use("X1", ResourceIOType::READ_WRITE, llama_add_bindings::D),
                                      use("Partial", ResourceIOType::READ_WRITE, 3)},
                     .shaderNames = {"add_f32.comp"},
                     .specializationConstants = {{.id = llama_add_spec_constants::NOREPEAT, .value = 1}},
                     .execute = AddExecute(K_HIDDEN)});

    builder.AddNode({.nodeName = "PostNorm",
                     .queueType = QueueType::COMPUTE,
                     .resourceUses = {use("X1", ResourceIOType::READ, llama_add_bindings::A),
                                      use("Norm2", ResourceIOType::READ, llama_add_bindings::B),
                                      use("Xn2", ResourceIOType::READ_WRITE, llama_add_bindings::D)},
                     .shaderNames = {"rms_norm_f32.comp"},
                     .specializationConstants = {{.id = 1, .value = 1}},
                     .execute = RmsNormExecute(K_HIDDEN)});

    builder.AddNode({.nodeName = "GateProj",
                     .queueType = QueueType::COMPUTE,
                     .resourceUses = gemvUses("Wgate", "Xn2", "Gate"),
                     .shaderNames = {"mul_mat_vec_f32_f32_f32.comp"},
                     .specializationConstants = gemvSpec,
                     .execute = GemvExecute(K_HIDDEN, K_INTERMEDIATE)});

    builder.AddNode({.nodeName = "UpProj",
                     .queueType = QueueType::COMPUTE,
                     .resourceUses = gemvUses("Wup", "Xn2", "Up"),
                     .shaderNames = {"mul_mat_vec_f32_f32_f32.comp"},
                     .specializationConstants = gemvSpec,
                     .execute = GemvExecute(K_HIDDEN, K_INTERMEDIATE)});

    builder.AddNode({.nodeName = "SwiGLU",
                     .queueType = QueueType::COMPUTE,
                     .resourceUses = {use("Gate", ResourceIOType::READ, llama_glu_bindings::A),
                                      use("Up", ResourceIOType::READ, llama_glu_bindings::B),
                                      use("Act", ResourceIOType::READ_WRITE, llama_glu_bindings::D)},
                     .shaderNames = {"swiglu_f32.comp"},
                     .execute = SwigluExecute(K_INTERMEDIATE)});

    builder.AddNode({.nodeName = "DownProj",
                     .queueType = QueueType::COMPUTE,
                     .resourceUses = gemvUses("Wdown", "Act", "MlpOut"),
                     .shaderNames = {"mul_mat_vec_f32_f32_f32.comp"},
                     .specializationConstants = gemvSpec,
                     .execute = GemvExecute(K_INTERMEDIATE, K_HIDDEN)});

    builder.AddNode({.nodeName = "Residual2",
                     .queueType = QueueType::COMPUTE,
                     .resourceUses = {use("X1", ResourceIOType::READ, llama_add_bindings::A),
                                      use("MlpOut", ResourceIOType::READ, llama_add_bindings::B),
                                      use("LayerOut", ResourceIOType::READ_WRITE, llama_add_bindings::D),
                                      use("Partial", ResourceIOType::READ_WRITE, 3)},
                     .shaderNames = {"add_f32.comp"},
                     .specializationConstants = {{.id = llama_add_spec_constants::NOREPEAT, .value = 1}},
                     .execute = AddExecute(K_HIDDEN)});

    // The ordering the graph does not infer: one edge per data dependency.
    const char* chain[] = {"PreNorm",   "QProj",    "QRope",  "AttnScores", "AttnWeights", "AttnOut",  "OProj",
                           "Residual1", "PostNorm", "UpProj", "SwiGLU",     "DownProj",    "Residual2"};
    for (size_t i = 0; i + 1 < std::size(chain); ++i)
    {
        builder.AddDependency(chain[i], chain[i + 1]);
    }
    builder.AddDependency("PostNorm", "GateProj");

    builder.Build();

    std::printf("[graph] %zu nodes in execution order:", builder.GetExecutionOrder().size());
    for (const std::string& nodeName : builder.GetExecutionOrder())
    {
        std::printf(" %s", nodeName.c_str());
    }
    std::printf("\n");

    const auto upload = [&](const char* name, const std::vector<float>& data)
    {
        auto* resource = GetRenderResourceManager()->GetResource<BufferResource>(name);
        resource->SetData(data.data(), data.size() * sizeof(float));
    };
    upload("Hidden", hostHidden);
    upload("Norm1", hostNorm1);
    upload("Norm2", hostNorm2);
    upload("Wq", hostWq);
    upload("Wo", hostWo);
    upload("Wgate", hostWgate);
    upload("Wup", hostWup);
    upload("Wdown", hostWdown);
    upload("KeyCache", hostKeys);
    upload("ValueCacheT", hostValuesTransposed);
    const std::vector<int32_t> hostPositions = {0};
    GetRenderResourceManager()
        ->GetResource<BufferResource>("RopePositions")
        ->SetData(hostPositions.data(), hostPositions.size() * sizeof(int32_t));

    builder.Execute();

    // CPU reference, the same sequence.
    const std::vector<float> xn = RmsNorm(hostHidden, hostNorm1);
    const std::vector<float> q = Gemv(hostWq, xn, K_HIDDEN, K_HIDDEN);
    std::vector<float> qr(K_HIDDEN);
    {
        const float thetaScale = static_cast<float>(std::pow(10000.0, -2.0 / static_cast<double>(K_HIDDEN)));
        for (uint32_t j = 0; j < K_HIDDEN / 2; ++j)
        {
            const float theta = 0.0F * std::pow(thetaScale, static_cast<float>(j));  // position 0
            const float c = std::cos(theta);
            const float s = std::sin(theta);
            qr[j] = q[j] * c - q[j + K_HIDDEN / 2] * s;
            qr[j + K_HIDDEN / 2] = q[j] * s + q[j + K_HIDDEN / 2] * c;
        }
    }
    const std::vector<float> scores = Gemv(hostKeys, qr, K_SEQ, K_HIDDEN);
    std::vector<float> scaledScores(K_SEQ);
    float maxScore = -1e30F;
    for (uint32_t key = 0; key < K_SEQ; ++key)
    {
        scaledScores[key] = scores[key] * K_ATTN_SCALE;
        maxScore = std::max(maxScore, scaledScores[key]);
    }
    double sumExp = 0.0;
    std::vector<float> probs(K_SEQ);
    for (uint32_t key = 0; key < K_SEQ; ++key)
    {
        probs[key] = std::exp(scaledScores[key] - maxScore);
        sumExp += probs[key];
    }
    for (float& probability : probs)
    {
        probability = static_cast<float>(probability / sumExp);
    }
    const std::vector<float> attnOut = Gemv(hostValuesTransposed, probs, K_HIDDEN, K_SEQ);
    const std::vector<float> o = Gemv(hostWo, attnOut, K_HIDDEN, K_HIDDEN);
    std::vector<float> x1(K_HIDDEN);
    for (uint32_t i = 0; i < K_HIDDEN; ++i)
    {
        x1[i] = hostHidden[i] + o[i];
    }
    const std::vector<float> xn2 = RmsNorm(x1, hostNorm2);
    const std::vector<float> gate = Gemv(hostWgate, xn2, K_INTERMEDIATE, K_HIDDEN);
    const std::vector<float> up = Gemv(hostWup, xn2, K_INTERMEDIATE, K_HIDDEN);
    std::vector<float> act(K_INTERMEDIATE);
    for (uint32_t i = 0; i < K_INTERMEDIATE; ++i)
    {
        act[i] = gate[i] / (1.0F + std::exp(-gate[i])) * up[i];
    }
    const std::vector<float> mlpOut = Gemv(hostWdown, act, K_HIDDEN, K_INTERMEDIATE);
    std::vector<float> expected(K_HIDDEN);
    for (uint32_t i = 0; i < K_HIDDEN; ++i)
    {
        expected[i] = x1[i] + mlpOut[i];
    }

    auto* pLayerOut = GetRenderResourceManager()->GetResource<BufferResource>("LayerOut");
    if (pLayerOut == nullptr)
    {
        std::fprintf(stderr, "[fail] the graph did not allocate LayerOut\n");
        return 1;
    }
    const auto* result = static_cast<const float*>(pLayerOut->Map());
    uint32_t nWrong = 0;
    for (uint32_t i = 0; i < K_HIDDEN; ++i)
    {
        if (std::fabs(result[i] - expected[i]) > 1e-3F * (1.0F + std::fabs(expected[i])))
        {
            if (nWrong < 4)
            {
                std::fprintf(stderr, "[fail] out[%u] = %g, expected %g\n", i, result[i], expected[i]);
            }
            ++nWrong;
        }
    }
    pLayerOut->Unmap();

    if (nWrong != 0)
    {
        std::fprintf(stderr, "[fail] %u of %u layer outputs wrong\n", nWrong, K_HIDDEN);
        return 1;
    }

    std::printf(
        "[layer] OK: %zu-node transformer layer (hidden %u, seq %u, intermediate %u) matches the CPU reference\n",
        builder.GetExecutionOrder().size(), K_HIDDEN, K_SEQ, K_INTERMEDIATE);
    return 0;
}
}  // namespace muyo_llm
