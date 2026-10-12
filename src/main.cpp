// muyo-llm — M0 smoke test.
//
// The smallest end-to-end proof that a windowless, compute-only consumer can drive Muyo's
// RenderGraph: one graph-owned storage buffer, one compute node that writes a pattern into it,
// and a host-side check that reads the pattern back. Everything after this reuses the same
// skeleton, so the point is not the pattern, it is that the whole path runs without a surface.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "Argmax.h"
#include "Attention.h"
#include "DescriptorManager.h"
#include "ElementwiseAdd.h"
#include "Embedding.h"
#include "FullModel.h"
#include "GetRows.h"
#include "GgufSummary.h"
#include "HeadlessContext.h"
#include "KvCache.h"
#include "MatVec.h"
#include "ModelSummary.h"
#include "RenderGraph/RenderGraphBuilder.h"
#include "RenderGraph/RenderGraphNodeContext.h"
#include "RenderGraph/RenderGraphNodeResource.h"
#include "RenderGraph/RenderGraphResourceDesc.h"
#include "RenderResource.h"
#include "RenderResourceManager.h"
#include "RmsNorm.h"
#include "Rope.h"
#include "SoftMax.h"
#include "Swiglu.h"
#include "Tokenize.h"
#include "TransformerLayer.h"
#include "VkRenderDevice.h"
#include "vk_mem_alloc.h"

namespace
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

/// Elements the smoke node writes. One element per invocation, 64 invocations per workgroup.
constexpr uint32_t K_ELEMENT_COUNT = 1024;
constexpr uint32_t K_WORKGROUP_SIZE = 64;

/// Matches `shaders/smoke.comp`'s `PushConstants` block. The graph reflects the range and puts
/// it in the pipeline layout; the layout is ours to fill, so it is defined here.
struct SmokePushConstants
{
    uint32_t count;
};

/// The value `smoke.comp` writes at `index`.
constexpr uint32_t ExpectedValue(uint32_t index) { return index * 3u + 1u; }

void PrintDevice()
{
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(GetRenderDevice()->GetPhysicalDevice(), &properties);
    const bool bDedicatedCompute = GetRenderDevice()->IsComputeQueueDedicated();
    std::printf("[device] %s (Vulkan %u.%u.%u), %s\n", properties.deviceName, VK_VERSION_MAJOR(properties.apiVersion),
                VK_VERSION_MINOR(properties.apiVersion), VK_VERSION_PATCH(properties.apiVersion),
                bDedicatedCompute ? "dedicated compute family present"
                                  : "single queue family: the cross-queue path is not exercised");

    // Say it explicitly: this is a compute-only consumer, and whether the graph treats that as a
    // first-class profile is one of the things it exists to find out.
    std::printf("[device] cooperative matrix: %s\n",
                GetRenderDevice()->IsCooperativeMatrixSupported() ? "available" : "not available");
}

/// Build, run and verify the smoke graph.
/// @return 0 when the GPU-produced pattern matches the host-side expectation, 1 otherwise.
int RunSmoke()
{
    PrintDevice();

    RenderGraphBuilder builder(GetRenderDevice());

    // A graph-owned buffer. CPU_TO_GPU so the host can read it back without a staging copy; the
    // weights and KV cache later are imported/persistent rather than graph-owned, which is a
    // different lifetime and a different thing to get wrong.
    builder.AddResource("SmokeValues", BufferResourceDesc{.count = K_ELEMENT_COUNT,
                                                          .stride = sizeof(uint32_t),
                                                          .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                          .memoryProperties = VMA_MEMORY_USAGE_CPU_TO_GPU,
                                                          .lifetime = ResourceLifetime::Transient});

    RenderGraphNodeCreateInfo fillPass = {
        .nodeName = "SmokeFill",
        .queueType = QueueType::COMPUTE,
        .resourceUses = {ResourceUse{.handle = ResourceHandle("SmokeValues"),
                                     .io = ResourceIOType::READ_WRITE,
                                     .usage = ResourceUsage::STORAGE_BUFFER,
                                     .kind = ResourceKind::BUFFER,
                                     .descriptorBinding = DescriptorBinding{.set = 0, .binding = 0}}},
        .shaderNames = {"smoke.comp"},
        .execute = [](RenderGraphNodeContext& ctx)
        {
            const SmokePushConstants pushConstants{.count = K_ELEMENT_COUNT};
            vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(pushConstants), &pushConstants);
            vkCmdDispatch(ctx.commandBuffer, (K_ELEMENT_COUNT + K_WORKGROUP_SIZE - 1) / K_WORKGROUP_SIZE, 1, 1);

            // The graph plans the barriers *into* a node, not out of the last one, so the
            // compute-write-to-host-read pair is the caller's. Field-by-field per
            // docs/CodingConventions.md: positional sync2 initializers shuffle the masks.
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

    builder.AddNode(fillPass);
    builder.Build();

    std::printf("[graph] execution order:");
    for (const std::string& nodeName : builder.GetExecutionOrder())
    {
        std::printf(" %s", nodeName.c_str());
    }
    std::printf("\n");

    builder.Execute();

    auto* pValues = GetRenderResourceManager()->GetResource<BufferResource>("SmokeValues");
    if (pValues == nullptr)
    {
        std::fprintf(stderr, "[fail] the graph did not allocate SmokeValues\n");
        return 1;
    }

    const auto* pMapped = static_cast<const uint32_t*>(pValues->Map());
    uint32_t nWrong = 0;
    for (uint32_t index = 0; index < K_ELEMENT_COUNT; ++index)
    {
        if (pMapped[index] != ExpectedValue(index))
        {
            if (nWrong < 4)
            {
                std::fprintf(stderr, "[fail] values[%u] = %u, expected %u\n", index, pMapped[index],
                             ExpectedValue(index));
            }
            ++nWrong;
        }
    }
    pValues->Unmap();

    if (nWrong != 0)
    {
        std::fprintf(stderr, "[fail] %u of %u elements wrong\n", nWrong, K_ELEMENT_COUNT);
        return 1;
    }

    std::printf("[smoke] OK: %u elements written by the compute node and verified on the host\n", K_ELEMENT_COUNT);
    return 0;
}
}  // namespace

int main(int argc, char** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "--smoke";
    // --gguf reads a file and prints a summary; it needs no device, so it runs before the context.
    if (mode == "--gguf")
    {
        if (argc < 3)
        {
            std::fprintf(stderr, "usage: %s --gguf <path>\n", argv[0]);
            return 2;
        }
        return muyo_llm::PrintGgufSummary(argv[2]);
    }
    // --model validates the tensor manifest; no device needed.
    if (mode == "--model")
    {
        if (argc < 3)
        {
            std::fprintf(stderr, "usage: %s --model <gguf>\n", argv[0]);
            return 2;
        }
        return muyo_llm::PrintModelSummary(argv[2]);
    }
    // --tokenize uses the GGUF tokenizer; it also needs no device.
    if (mode == "--tokenize")
    {
        if (argc < 4)
        {
            std::fprintf(stderr, "usage: %s --tokenize <gguf> <text>\n", argv[0]);
            return 2;
        }
        try
        {
            return muyo_llm::RunTokenize(argv[2], argv[3]);
        }
        catch (const std::exception& exception)
        {
            std::fprintf(stderr, "[fail] %s\n", exception.what());
            return 1;
        }
    }
    // --embed loads a model's embedding from a GGUF file; it needs the device for the graph.
    if (mode == "--embed")
    {
        if (argc < 4)
        {
            std::fprintf(stderr, "usage: %s --embed <gguf> <token>\n", argv[0]);
            return 2;
        }
        try
        {
            const muyo_llm::HeadlessContext context;
            return muyo_llm::RunEmbedding(argv[2], std::stoi(argv[3]));
        }
        catch (const std::exception& exception)
        {
            std::fprintf(stderr, "[fail] %s\n", exception.what());
            return 1;
        }
    }
    const std::vector<std::string> modes = {"--smoke",   "--add",     "--matvec", "--rmsnorm", "--rope",
                                            "--swiglu",  "--getrows", "--argmax", "--softmax", "--attention",
                                            "--kvcache", "--layer",   "--forward"};
    if (std::find(modes.begin(), modes.end(), mode) == modes.end())
    {
        std::fprintf(stderr, "usage: %s", argv[0]);
        for (const std::string& name : modes)
        {
            std::fprintf(stderr, " %s", name.c_str());
        }
        std::fprintf(stderr, "\n");
        return 2;
    }

    try
    {
        // One context for the process: the Muyo subsystems are singletons, and every mode relies
        // on them already being initialised.
        const muyo_llm::HeadlessContext context;
        if (mode == "--add")
        {
            return muyo_llm::RunElementwiseAdd();
        }
        if (mode == "--matvec")
        {
            return muyo_llm::RunMatVec();
        }
        if (mode == "--rmsnorm")
        {
            return muyo_llm::RunRmsNorm();
        }
        if (mode == "--rope")
        {
            return muyo_llm::RunRope();
        }
        if (mode == "--swiglu")
        {
            return muyo_llm::RunSwiglu();
        }
        if (mode == "--getrows")
        {
            return muyo_llm::RunGetRows();
        }
        if (mode == "--argmax")
        {
            return muyo_llm::RunArgmax();
        }
        if (mode == "--softmax")
        {
            return muyo_llm::RunSoftMax();
        }
        if (mode == "--attention")
        {
            return muyo_llm::RunAttention();
        }
        if (mode == "--kvcache")
        {
            return muyo_llm::RunKvCache();
        }
        if (mode == "--layer")
        {
            return muyo_llm::RunTransformerLayer();
        }
        if (mode == "--forward")
        {
            if (argc < 4)
            {
                std::fprintf(stderr, "usage: %s --forward <gguf> [-p <text> | --tokens id,id] [-n N] [sampler flags]\n",
                             argv[0]);
                return 2;
            }
            muyo_llm::ForwardOptions options;
            bool chosePrompt = false;
            const auto value = [&](int& index, const char* name) -> std::string
            {
                if (index + 1 >= argc)
                {
                    std::fprintf(stderr, "%s needs a value\n", name);
                    throw std::runtime_error("missing value");
                }
                return argv[++index];
            };
            const auto parseTokens = [](const std::string& list)
            {
                std::vector<int32_t> tokens;
                std::string field;
                for (const char* cursor = list.c_str();; ++cursor)
                {
                    if (*cursor == ',' || *cursor == '\0')
                    {
                        if (!field.empty())
                        {
                            tokens.push_back(static_cast<int32_t>(std::stol(field)));
                            field.clear();
                        }
                        if (*cursor == '\0')
                        {
                            break;
                        }
                    }
                    else
                    {
                        field.push_back(*cursor);
                    }
                }
                return tokens;
            };
            for (int i = 3; i < argc; ++i)
            {
                const std::string arg = argv[i];
                if (arg == "-p" || arg == "--prompt")
                {
                    options.prompt = value(i, "--prompt");
                    options.tokens.clear();
                    chosePrompt = true;
                }
                else if (arg == "--tokens")
                {
                    options.tokens = parseTokens(value(i, "--tokens"));
                    options.prompt.clear();
                    chosePrompt = true;
                }
                else if (arg == "-n" || arg == "--n-predict")
                {
                    options.nPredict = static_cast<uint32_t>(std::stoul(value(i, "-n")));
                }
                else if (arg == "--temp" || arg == "--temperature")
                {
                    options.sampler.temperature = std::stof(value(i, "--temp"));
                }
                else if (arg == "--top-k")
                {
                    options.sampler.topK = std::stoi(value(i, "--top-k"));
                }
                else if (arg == "--top-p")
                {
                    options.sampler.topP = std::stof(value(i, "--top-p"));
                }
                else if (arg == "--repeat-penalty")
                {
                    options.sampler.repeatPenalty = std::stof(value(i, "--repeat-penalty"));
                }
                else if (arg == "--repeat-last-n")
                {
                    options.sampler.repeatLastN = std::stoi(value(i, "--repeat-last-n"));
                }
                else if (arg == "--seed")
                {
                    options.sampler.seed = std::stoull(value(i, "--seed"));
                }
                else if (arg == "--chat")
                {
                    options.chat = true;
                }
                else if (arg == "-i" || arg == "--interactive")
                {
                    options.chat = true;
                    options.interactive = true;
                }
                else if (arg == "--system")
                {
                    options.system = value(i, "--system");
                }
                else
                {
                    std::fprintf(stderr, "unknown --forward option: %s\n", arg.c_str());
                    return 2;
                }
            }
            if (!chosePrompt && !options.chat)
            {
                // A completion run needs a prompt; the default is the one the README uses. In chat
                // mode an empty prompt means "no initial turn": interactive starts from the system
                // turn, and a single turn falls back to "Hello" in RunFullModel.
                options.prompt = "Hello world";
                std::fprintf(stderr, "[forward] no -p/--tokens given, using -p \"%s\"\n", options.prompt.c_str());
            }
            return muyo_llm::RunFullModel(argv[2], options);
        }
        return RunSmoke();
    }
    catch (const std::exception& exception)
    {
        std::fprintf(stderr, "[fail] %s\n", exception.what());
        return 1;
    }
}
