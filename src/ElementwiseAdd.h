#pragma once

namespace muyo_llm
{
/// Run llama.cpp's `add.comp` through the graph as `C[i] = A[i] + B[i]` and verify it against a
/// CPU reference.
///
/// Exercises the parts of a foreign shader's contract the graph does not own: the compile-time
/// define set, the reflected push-constant block, an explicit 4-binding descriptor set including a
/// binding the shader declares but does not use, and a specialization constant.
///
/// Assumes the Muyo subsystems are already initialised (see `HeadlessContext`).
/// @return 0 when the GPU result matches the CPU reference, 1 otherwise.
int RunElementwiseAdd();
}  // namespace muyo_llm
