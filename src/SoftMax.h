#pragma once

namespace muyo_llm
{
/// Run llama.cpp's `soft_max.comp` through the graph and verify it against a CPU reference.
///
/// The attention-weights normalization, with the additive mask operand disabled (`KY = 0`). The
/// `scale` is applied inside the shader (`x * scale`), which is where the attention score's
/// `1/sqrt(headDim)` belongs. It is the first op that reads and writes the same buffer in place.
///
/// Assumes the Muyo subsystems are already initialised (see `HeadlessContext`).
/// @return 0 when the GPU result matches the CPU reference within tolerance, 1 otherwise.
int RunSoftMax();
}  // namespace muyo_llm
