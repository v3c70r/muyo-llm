#pragma once

namespace muyo_llm
{
/// Run llama.cpp's `mul_mat_vec.comp` (F16 weights x F32 activation) through the graph as
/// `d[row] = sum_k A[row, k] * b[k]`, and verify it against a CPU reference.
///
/// This is the decode workhorse: every projection in a transformer layer is a matrix-vector
/// product once the batch is a single token.
///
/// Assumes the Muyo subsystems are already initialised (see `HeadlessContext`).
/// @return 0 when the GPU result matches the CPU reference, 1 otherwise.
int RunMatVec();
}  // namespace muyo_llm
