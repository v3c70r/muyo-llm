#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace muyo_llm
{
/// Run the whole Qwen2 model through the graph.
///
/// Feeds `prompt` one token at a time (each at the next cache position), then greedily samples
/// `nPredict` tokens with the LM head and `argmax`. The graph is built once; every step re-records
/// it, so the only state that changes is the KV cache and the three per-step push constants
/// (the position, the rope destination offset and the score dispatch).
///
/// @param ggufPath Path to a Qwen2 GGUF whose weights are F16/F32.
/// @param prompt   Token ids to prime the cache with.
/// @param nPredict Number of tokens to sample.
/// @return 0 on success, non-zero on a load or graph failure.
int RunFullModel(const std::string& ggufPath, const std::vector<int32_t>& prompt, uint32_t nPredict);
}  // namespace muyo_llm
