#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Sampler.h"

namespace muyo_llm
{
/// What to feed the model and how to sample from it.
struct ForwardOptions
{
    std::vector<int32_t> tokens;  ///< Prompt token ids; used when `prompt` is empty.
    std::string prompt;           ///< Prompt text, tokenized with the model's tokenizer when set.
    uint32_t nPredict = 32;       ///< Number of tokens to generate.
    SamplerParams sampler;        ///< Greedy by default (`temperature = 0`).
};

/// Run the whole Qwen2 model through the graph.
///
/// Feeds the prompt one token at a time (each at the next cache position), then samples `nPredict`
/// tokens. The graph is built once; every step re-records it, so the only state that changes is the
/// KV cache, the three per-step push constants (the position, the rope destination offset and the
/// score dispatch) and the token id. The generated text is streamed to stdout; diagnostics go to
/// stderr.
///
/// @param ggufPath Path to a Qwen2 GGUF whose weights are F16/F32.
/// @param options  Prompt, generation length and sampler.
/// @return 0 on success, non-zero on a load or graph failure.
int RunFullModel(const std::string& ggufPath, const ForwardOptions& options);
}  // namespace muyo_llm
