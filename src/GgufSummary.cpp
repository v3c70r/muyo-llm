#include "GgufSummary.h"

#include <cstdio>
#include <exception>

#include "Gguf.h"

namespace muyo_llm
{
int PrintGgufSummary(const std::string& path)
{
    try
    {
        const GgufReader reader(path);
        std::printf("[gguf] %s\n", path.c_str());
        std::printf("[gguf] version %u, %zu tensors, data starts at byte %llu\n", reader.Version(),
                    reader.Tensors().size(), static_cast<unsigned long long>(reader.DataOffset()));
        std::printf("[gguf] general.architecture = %s\n", reader.StringOr("general.architecture", "(none)").c_str());
        std::printf("[gguf] general.name         = %s\n", reader.StringOr("general.name", "(none)").c_str());
        std::printf("[gguf] tokenizer.ggml.model = %s\n", reader.StringOr("tokenizer.ggml.model", "(none)").c_str());

        const size_t tokenCount = reader.StringArraySize("tokenizer.ggml.tokens");
        const size_t mergeCount = reader.StringArraySize("tokenizer.ggml.merges");
        std::printf("[gguf] tokenizer tokens = %zu, merges = %zu\n", tokenCount, mergeCount);
        if (tokenCount == 0)
        {
            std::fprintf(stderr, "[fail] no tokenizer.ggml.tokens array\n");
            return 1;
        }
        std::printf("[gguf] first tokens:");
        for (size_t i = 0; i < 5 && i < tokenCount; ++i)
        {
            std::printf(" [%zu]='%s'", i, reader.StringArrayAt("tokenizer.ggml.tokens", i).c_str());
        }
        std::printf("\n");

        const auto& tensors = reader.Tensors();
        const size_t shown = tensors.size() < 5 ? tensors.size() : 5;
        for (size_t i = 0; i < shown; ++i)
        {
            std::printf("[gguf] tensor %-32s type=%u dims=", tensors[i].name.c_str(), tensors[i].type);
            for (size_t d = 0; d < tensors[i].dimensions.size(); ++d)
            {
                std::printf("%s%llu", d == 0 ? "" : "x", static_cast<unsigned long long>(tensors[i].dimensions[d]));
            }
            std::printf("\n");
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
