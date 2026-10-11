#pragma once

namespace muyo_llm
{
/// Run llama.cpp's `get_rows.comp` through the graph and verify it against a CPU reference.
///
/// The embedding lookup: gather one row of the token-embedding matrix. It is the first op whose
/// output row is selected by an *index* operand rather than by the invocation id, so a wrong index
/// stride reads a different token's vector - a plausible-looking wrong answer, not a crash.
///
/// Assumes the Muyo subsystems are already initialised (see `HeadlessContext`).
/// @return 0 when the GPU result matches the CPU reference, 1 otherwise.
int RunGetRows();
}  // namespace muyo_llm
