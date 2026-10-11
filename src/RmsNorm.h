#pragma once

namespace muyo_llm
{
/// Run llama.cpp's `rms_norm.comp` through the graph and verify it against a CPU reference.
///
/// RMSNorm is the transformer's normalization: `y = x * rsqrt(mean(x^2) + eps) * weight`. It is the
/// first op that reduces across a row rather than element-wise, so it exercises the shader's
/// workgroup reduction and a second input (the weight).
///
/// Assumes the Muyo subsystems are already initialised (see `HeadlessContext`).
/// @return 0 when the GPU result matches the CPU reference within tolerance, 1 otherwise.
int RunRmsNorm();
}  // namespace muyo_llm
