#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace muyo_llm
{
/// Host-side sampling parameters.
///
/// `temperature <= 0` selects greedy, so the default is reproducible and the flags only matter when
/// a caller asks for them. `repeatPenalty` follows the usual convention: a logit that is already
/// positive is divided by the penalty, a negative one multiplied, which pushes both away from 0.
struct SamplerParams
{
    float temperature = 0.0F;
    int32_t topK = 40;
    float topP = 0.95F;
    float repeatPenalty = 1.0F;
    int32_t repeatLastN = 64;  ///< History window the penalty looks at; negative means all of it.
    uint64_t seed = 0;
};

/// Samples the next token from a logits row.
///
/// The RNG lives here rather than in the free function so a seeded run is reproducible: seeding per
/// call would draw the same quantile every step, which is deterministic but degenerate.
class Sampler
{
public:
    explicit Sampler(const SamplerParams& params) : m_params(params), m_rng(params.seed) {}

    /// @param logits  The model's logits for the current position. Modified in place by the penalty.
    /// @param vocab   Number of logits.
    /// @param history Tokens fed and generated so far, newest last; used by the repetition penalty.
    /// @return The sampled token id.
    int32_t Sample(float* logits, int32_t vocab, const std::vector<int32_t>& history);

private:
    SamplerParams m_params;
    std::mt19937_64 m_rng;
};
}  // namespace muyo_llm
