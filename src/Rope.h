#pragma once

namespace muyo_llm
{
/// Run llama.cpp's `rope_neox.comp` through the graph and verify it against a CPU reference.
///
/// Rotary position embedding rotates each head's channels in place by a position-dependent angle.
/// The NEOX (half-split) layout is what Qwen and most modern models use. This is the first op whose
/// input is not a per-element function of the same index - the pair `(j, j + nDims/2)` is mixed -
/// so a wrong stride or channel mapping shows as a wrong rotation rather than a shifted copy.
///
/// Assumes the Muyo subsystems are already initialised (see `HeadlessContext`).
/// @return 0 when the GPU result matches the CPU reference within tolerance, 1 otherwise.
int RunRope();
}  // namespace muyo_llm
