#pragma once

namespace muyo_llm
{
/// Run llama.cpp's `swiglu.comp` through the graph and verify it against a CPU reference.
///
/// The gated MLP activation: `out = silu(gate) * up` with the gate and up projections as separate
/// inputs (split mode). It is the first op that reads two full-size operands and writes one, and
/// the first whose push constants carry host-computed `fastdiv` magic numbers.
///
/// Assumes the Muyo subsystems are already initialised (see `HeadlessContext`).
/// @return 0 when the GPU result matches the CPU reference within tolerance, 1 otherwise.
int RunSwiglu();
}  // namespace muyo_llm
