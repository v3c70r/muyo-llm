#include "Sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace muyo_llm
{
int32_t Sampler::Sample(float* logits, int32_t vocab, const std::vector<int32_t>& history)
{
    if (m_params.repeatPenalty != 1.0F && !history.empty())
    {
        const size_t historySize = history.size();
        size_t begin = 0;
        if (m_params.repeatLastN >= 0 && static_cast<size_t>(m_params.repeatLastN) < historySize)
        {
            begin = historySize - static_cast<size_t>(m_params.repeatLastN);
        }
        for (size_t i = begin; i < historySize; ++i)
        {
            const int32_t token = history[i];
            if (token < 0 || token >= vocab)
            {
                continue;
            }
            float& value = logits[token];
            value = value > 0.0F ? value / m_params.repeatPenalty : value * m_params.repeatPenalty;
        }
    }

    if (m_params.temperature <= 0.0F)
    {
        int32_t best = 0;
        for (int32_t i = 1; i < vocab; ++i)
        {
            if (logits[i] > logits[best])
            {
                best = i;
            }
        }
        return best;
    }

    for (int32_t i = 0; i < vocab; ++i)
    {
        logits[i] /= m_params.temperature;
    }

    const int32_t k = m_params.topK <= 0 ? vocab : std::min(m_params.topK, vocab);
    std::vector<int32_t> order(static_cast<size_t>(vocab));
    for (int32_t i = 0; i < vocab; ++i)
    {
        order[static_cast<size_t>(i)] = i;
    }
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });

    const float maxLogit = logits[order[0]];
    std::vector<double> weights(static_cast<size_t>(k));
    double total = 0.0;
    for (int32_t i = 0; i < k; ++i)
    {
        weights[static_cast<size_t>(i)] =
            std::exp(static_cast<double>(logits[order[static_cast<size_t>(i)]] - maxLogit));
        total += weights[static_cast<size_t>(i)];
    }

    // Nucleus: keep the shortest prefix whose mass reaches `topP`.
    int32_t kept = k;
    double cumulative = 0.0;
    for (int32_t i = 0; i < k; ++i)
    {
        cumulative += weights[static_cast<size_t>(i)] / total;
        if (cumulative >= static_cast<double>(m_params.topP))
        {
            kept = i + 1;
            break;
        }
    }
    double keptTotal = 0.0;
    for (int32_t i = 0; i < kept; ++i)
    {
        keptTotal += weights[static_cast<size_t>(i)];
    }

    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    const double target = distribution(m_rng) * keptTotal;
    double accumulated = 0.0;
    for (int32_t i = 0; i < kept; ++i)
    {
        accumulated += weights[static_cast<size_t>(i)];
        if (accumulated >= target)
        {
            return order[static_cast<size_t>(i)];
        }
    }
    return order[static_cast<size_t>(kept - 1)];
}
}  // namespace muyo_llm
