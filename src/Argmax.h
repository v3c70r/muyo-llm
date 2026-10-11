#pragma once

namespace muyo_llm
{
/// Run llama.cpp's `argmax.comp` through the graph and verify it against a CPU reference.
///
/// Greedy sampling: the index of the largest logit. It is the last op in the decode chain and the
/// one whose output leaves the GPU as an index rather than a value - so off-by-one ties and a
/// wrong reduction both change which token is produced.
///
/// Assumes the Muyo subsystems are already initialised (see `HeadlessContext`).
/// @return 0 when the GPU result matches the CPU reference, 1 otherwise.
int RunArgmax();
}  // namespace muyo_llm
