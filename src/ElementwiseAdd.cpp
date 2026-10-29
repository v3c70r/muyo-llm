#include "ElementwiseAdd.h"

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
/// Deliberately not a multiple of the shader's 512-element coverage per workgroup, so the
/// `idx >= p.ne` guard in `add.comp` is exercised rather than assumed.
constexpr uint32_t K_ELEMENT_COUNT = 1000;

/// `add.comp` is dispatched with `wg_denoms = {512, 1, 1}` and `elements = {512, ceil(ne/512), 1}`
/// (ggml-vulkan-common.h `ggml_vk_dispatch_pipeline`, `ggml_vk_op_f32`): workgroup counts
/// `(ceil(512/512), ceil(ceil(ne/512)/1), 1) = (1, ceil(ne/512), 1)`. The single x workgroup of
/// 256 threads covers a 512-element `y`-block because each thread writes `idx` and `idx + 256`.
/// Dispatching x by `ceil(ne/512)` instead - the intuitive spelling - covers only the first 768
/// elements, which is how this was found.
constexpr uint32_t K_ELEMENTS_PER_Y_BLOCK = 512;

/// `misalign_offsets` packs the three operands' byte offsets into three bytes; each heap descriptor
/// binds its whole buffer at offset 0.
constexpr uint32_t K_MISALIGN_OFFSETS = 0;

/// Fill the ggml-shaped push-constant block for a flat, single-dimension `A op B -> D`.
/// @param elementCount Number of elements in the flat tensor.
/// @return The block to push, matching `generic_binary_head.glsl`.
LlamaAddPushConstants MakeAddPushConstants(uint32_t elementCount)
{
    // ne0x/ne1x/ne2x: one dimension of `elementCount` elements. nb0x/nb1x/nb2x: strides in
    // *elements*, so a contiguous tensor is nb[0] = 1 and the outer strides = ne[0]. ggml passes
    // `tensor->nb[i] / type_size` (see `ggml_vk_add`); byte strides here write each element to a
    // quarter of the intended positions instead.
    constexpr uint32_t elementStride = 1;

    LlamaAddPushConstants pushConstants{};
    pushConstants.ne = elementCount;
    for (uint32_t* ne : {&pushConstants.ne00, &pushConstants.ne10, &pushConstants.ne20})
    {
        ne[0] = elementCount;
        ne[1] = 1;
        ne[2] = 1;
        ne[3] = 1;
    }
    for (uint32_t* nb : {&pushConstants.nb00, &pushConstants.nb10, &pushConstants.nb20})
    {
        nb[0] = elementStride;
        nb[1] = elementCount;
        nb[2] = elementCount;
        nb[3] = elementCount;
    }
    pushConstants.misalignOffsets = K_MISALIGN_OFFSETS;
    pushConstants.param1 = 0.0F;
    pushConstants.param2 = 0.0F;
    pushConstants.param3 = 0;
    return pushConstants;
}
}  // namespace

int RunElementwiseAdd()
{
    std::vector<float> hostA(K_ELEMENT_COUNT);
    std::vector<float> hostB(K_ELEMENT_COUNT);
    for (uint32_t index = 0; index < K_ELEMENT_COUNT; ++index)
    {
        hostA[index] = 0.5F * static_cast<float>(index);
        hostB[index] = 1.0F + 0.25F * static_cast<float>(index);
    }

    RenderGraphBuilder builder(GraphServices());

    // A and B are uploaded from the host, so they carry TRANSFER_DST for `BufferResource::SetData`,
    // which stages the copy rather than assuming host-visible device memory. D is read back on the
    // host, so CPU_TO_GPU. `PartialSums` is heap slot 3, which `add.comp` declares and indexes
    // unconditionally even when ADD_RMS is off.
    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    const auto floatBuffer = [](uint32_t count, VkBufferUsageFlags usage)
    {
        return BufferResourceDesc{.count = count,
                                  .stride = sizeof(float),
                                  .usage = usage,
                                  .memoryProperties = VMA_MEMORY_USAGE_CPU_TO_GPU,
                                  .lifetime = ResourceLifetime::Persistent};
    };
    builder.AddResource("MatrixA", floatBuffer(K_ELEMENT_COUNT, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    builder.AddResource("MatrixB", floatBuffer(K_ELEMENT_COUNT, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    builder.AddResource("MatrixD", floatBuffer(K_ELEMENT_COUNT, storage));
    builder.AddResource("PartialSums", floatBuffer(2, storage));

    RenderGraphNodeCreateInfo addPass = {
        .nodeName = "LlamaAdd",
        .queueType = QueueType::COMPUTE,
        .resourceUses =
            {
                ResourceUse{.handle = ResourceHandle("MatrixA"),
                            .io = ResourceIOType::READ,
                            .usage = ResourceUsage::STORAGE_BUFFER,
                            .kind = ResourceKind::BUFFER},
                ResourceUse{.handle = ResourceHandle("MatrixB"),
                            .io = ResourceIOType::READ,
                            .usage = ResourceUsage::STORAGE_BUFFER,
                            .kind = ResourceKind::BUFFER},
                // READ_WRITE, not WRITE: the usage table's `{STORAGE_BUFFER, WRITE}` profile is a
                // *host* write, so a GPU writer has to spell READ_WRITE (docs/Findings.md F3,
                // Muyo #12).
                ResourceUse{.handle = ResourceHandle("MatrixD"),
                            .io = ResourceIOType::READ_WRITE,
                            .usage = ResourceUsage::STORAGE_BUFFER,
                            .kind = ResourceKind::BUFFER},
                ResourceUse{.handle = ResourceHandle("PartialSums"),
                            .io = ResourceIOType::READ_WRITE,
                            .usage = ResourceUsage::STORAGE_BUFFER,
                            .kind = ResourceKind::BUFFER},
            },
        .shaderNames = {"add_f32.comp"},
        // `norepeat` is true because A and B have the same shape, so B's indices need no modulo.
        .specializationConstants = {{.id = llama_add_spec_constants::NOREPEAT, .value = 1}},
        .execute = [](RenderGraphNodeContext& ctx)
        {
            LlamaAddPushConstants pushConstants = MakeAddPushConstants(K_ELEMENT_COUNT);
            SetHeapIndex(pushConstants.heap, llama_add_bindings::A, ctx, "MatrixA", ResourceUsage::STORAGE_BUFFER);
            SetHeapIndex(pushConstants.heap, llama_add_bindings::B, ctx, "MatrixB", ResourceUsage::STORAGE_BUFFER);
            SetHeapIndex(pushConstants.heap, llama_add_bindings::D, ctx, "MatrixD", ResourceUsage::STORAGE_BUFFER);
            SetHeapIndex(pushConstants.heap, llama_add_bindings::PARTIAL_SUMS, ctx, "PartialSums",
                         ResourceUsage::STORAGE_BUFFER);
            PushData(ctx, pushConstants);

            const uint32_t yBlocks = (K_ELEMENT_COUNT + K_ELEMENTS_PER_Y_BLOCK - 1) / K_ELEMENTS_PER_Y_BLOCK;
            vkCmdDispatch(ctx.commandBuffer, 1, yBlocks, 1);

            // The graph plans barriers into a node, not out of the last one, so the compute-write to
            // host-read pair is ours (docs/Findings.md F4).
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

    builder.AddNode(addPass);
    builder.Build();

    std::printf("[graph] execution order:");
    for (const std::string& nodeName : builder.GetExecutionOrder())
    {
        std::printf(" %s", nodeName.c_str());
    }
    std::printf("\n");

    // Upload after Build(): graph-owned buffers are allocated there. SetData stages the copy and
    // waits for it, so the graph's first barrier sees the data in device memory.
    auto* pA = GetRenderResourceManager()->GetResource<BufferResource>("MatrixA");
    auto* pB = GetRenderResourceManager()->GetResource<BufferResource>("MatrixB");
    auto* pD = GetRenderResourceManager()->GetResource<BufferResource>("MatrixD");
    if (pA == nullptr || pB == nullptr || pD == nullptr)
    {
        std::fprintf(stderr, "[fail] the graph did not allocate the add buffers\n");
        return 1;
    }
    pA->SetData(hostA.data(), hostA.size() * sizeof(float));
    pB->SetData(hostB.data(), hostB.size() * sizeof(float));

    builder.Execute();

    const auto* pResult = static_cast<const float*>(pD->Map());
    uint32_t nWrong = 0;
    for (uint32_t index = 0; index < K_ELEMENT_COUNT; ++index)
    {
        const float expected = hostA[index] + hostB[index];
        if (pResult[index] != expected)
        {
            if (nWrong < 4)
            {
                std::fprintf(stderr, "[fail] D[%u] = %g, expected %g\n", index, pResult[index], expected);
            }
            ++nWrong;
        }
    }
    pD->Unmap();

    if (nWrong != 0)
    {
        std::fprintf(stderr, "[fail] %u of %u elements wrong\n", nWrong, K_ELEMENT_COUNT);
        return 1;
    }

    std::printf("[add] OK: llama.cpp add_f32 over %u elements matches the CPU reference\n", K_ELEMENT_COUNT);
    return 0;
}
}  // namespace muyo_llm
