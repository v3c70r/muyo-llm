#include "ModelSummary.h"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "Gguf.h"

namespace muyo_llm
{
namespace
{
int g_failures = 0;

/// Check one expected tensor: present, exact shape (GGUF order, fastest dimension first), and an
/// integer type the engine can upload today.
void Expect(const GgufReader& reader, const std::string& name, const std::vector<uint64_t>& dims)
{
    const GgufTensorInfo* tensor = reader.FindTensor(name);
    if (tensor == nullptr)
    {
        std::fprintf(stderr, "[model] missing %s\n", name.c_str());
        ++g_failures;
        return;
    }
    if (tensor->dimensions != dims)
    {
        std::fprintf(stderr, "[model] %s has dims", name.c_str());
        for (uint64_t d : tensor->dimensions)
        {
            std::fprintf(stderr, " %llu", static_cast<unsigned long long>(d));
        }
        std::fprintf(stderr, ", expected");
        for (uint64_t d : dims)
        {
            std::fprintf(stderr, " %llu", static_cast<unsigned long long>(d));
        }
        std::fprintf(stderr, "\n");
        ++g_failures;
        return;
    }
    if (tensor->type != 0 && tensor->type != 1)
    {
        std::fprintf(stderr, "[model] %s has type %u, expected F32 or F16\n", name.c_str(), tensor->type);
        ++g_failures;
    }
}
}  // namespace

int PrintModelSummary(const std::string& path)
{
    try
    {
        const GgufReader reader(path);
        const std::string arch = reader.StringOr("general.architecture", "(none)");
        const uint64_t blockCount = reader.UintOr("qwen2.block_count", 0);
        const uint64_t hidden = reader.UintOr("qwen2.embedding_length", 0);
        const uint64_t feedForward = reader.UintOr("qwen2.feed_forward_length", 0);
        const uint64_t heads = reader.UintOr("qwen2.attention.head_count", 0);
        const uint64_t kvHeads = reader.UintOr("qwen2.attention.head_count_kv", 0);

        std::printf("[model] %s\n", path.c_str());
        std::printf("[model] architecture=%s blocks=%llu hidden=%llu ffn=%llu heads=%llu kv_heads=%llu\n", arch.c_str(),
                    static_cast<unsigned long long>(blockCount), static_cast<unsigned long long>(hidden),
                    static_cast<unsigned long long>(feedForward), static_cast<unsigned long long>(heads),
                    static_cast<unsigned long long>(kvHeads));
        std::printf("[model] tensors=%zu\n", reader.Tensors().size());

        if (arch != "qwen2" || blockCount == 0 || hidden == 0 || heads == 0)
        {
            std::fprintf(stderr, "[model] this validator only knows the qwen2 layout\n");
            return 1;
        }

        const uint64_t headDim = hidden / heads;
        const uint64_t queryDim = heads * headDim;
        const uint64_t kvDim = kvHeads * headDim;
        const uint64_t vocab = reader.StringArraySize("tokenizer.ggml.tokens");

        Expect(reader, "token_embd.weight", {hidden, vocab});
        Expect(reader, "output_norm.weight", {hidden});
        Expect(reader, "output.weight", {hidden, vocab});

        for (uint64_t layer = 0; layer < blockCount; ++layer)
        {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            Expect(reader, prefix + "attn_norm.weight", {hidden});
            Expect(reader, prefix + "attn_q.weight", {hidden, queryDim});
            Expect(reader, prefix + "attn_q.bias", {queryDim});
            Expect(reader, prefix + "attn_k.weight", {hidden, kvDim});
            Expect(reader, prefix + "attn_k.bias", {kvDim});
            Expect(reader, prefix + "attn_v.weight", {hidden, kvDim});
            Expect(reader, prefix + "attn_v.bias", {kvDim});
            Expect(reader, prefix + "attn_output.weight", {queryDim, hidden});
            Expect(reader, prefix + "ffn_norm.weight", {hidden});
            Expect(reader, prefix + "ffn_gate.weight", {hidden, feedForward});
            Expect(reader, prefix + "ffn_up.weight", {hidden, feedForward});
            Expect(reader, prefix + "ffn_down.weight", {feedForward, hidden});
        }

        // The scalars the forward pass needs but that are easy to ignore until they bite.
        std::printf("[model] head_dim=%llu query_dim=%llu kv_dim=%llu gqa_ratio=%llu vocab=%llu\n",
                    static_cast<unsigned long long>(headDim), static_cast<unsigned long long>(queryDim),
                    static_cast<unsigned long long>(kvDim),
                    static_cast<unsigned long long>(kvHeads == 0 ? 0 : heads / kvHeads),
                    static_cast<unsigned long long>(vocab));
        std::printf("[model] rms_eps=%g rope_base=%g\n", reader.FloatOr("qwen2.attention.layer_norm_rms_epsilon", 0.0),
                    reader.FloatOr("qwen2.rope.freq_base", 0.0));

        if (g_failures != 0)
        {
            std::fprintf(stderr, "[fail] %d tensor(s) failed validation\n", g_failures);
            return 1;
        }
        std::printf("[model] OK: manifest matches the qwen2 layout for %llu layers\n",
                    static_cast<unsigned long long>(blockCount));
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::fprintf(stderr, "[fail] %s\n", exception.what());
        return 1;
    }
}
}  // namespace muyo_llm
