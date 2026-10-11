#include "Tokenize.h"

#include <cstdio>
#include <string>
#include <vector>

#include "Gguf.h"
#include "Tokenizer.h"

namespace muyo_llm
{
int RunTokenize(const std::string& path, const std::string& text)
{
    const GgufReader reader(path);
    const Tokenizer tokenizer(reader);
    std::printf("[tokenize] vocab %zu, model %s\n", tokenizer.VocabSize(),
                reader.StringOr("tokenizer.ggml.model", "(none)").c_str());

    const std::vector<int32_t> ids = tokenizer.Encode(text);
    std::printf("[tokenize] \"%s\" ->", text.c_str());
    for (int32_t id : ids)
    {
        std::printf(" %d", id);
    }
    std::printf("\n");

    const std::string roundTrip = tokenizer.Decode(ids);
    if (roundTrip != text)
    {
        std::fprintf(stderr, "[fail] round-trip differs: '%s' != '%s'\n", roundTrip.c_str(), text.c_str());
        return 1;
    }

    // External anchor: the Qwen2/GPT-2 vocab has token 0 = "!", and "!" byte-encodes to itself and
    // pre-tokenizes alone, so it must encode to exactly {0}.
    if (tokenizer.TokenFor(0) == "!")
    {
        const std::vector<int32_t> bang = tokenizer.Encode("!");
        if (bang.size() != 1 || bang[0] != 0)
        {
            std::fprintf(stderr, "[fail] Encode(\"!\") is not {0}\n");
            return 1;
        }
        std::printf("[tokenize] anchor OK: Encode(\"!\") == {0}\n");
    }

    std::printf("[tokenize] OK: %zu tokens, round-trip exact\n", ids.size());
    return 0;
}
}  // namespace muyo_llm
