#pragma once

#include <string>

namespace muyo_llm
{
/// Read a model GGUF and validate its tensor manifest against the Qwen2 architecture the plan
/// targets, printing the structure and the metadata the forward pass will need.
///
/// Needs no device. This is the M4 preparation: it pins the exact tensor names, shapes and dtypes a
/// full forward pass has to bind, and fails loudly on a missing or mis-shaped tensor rather than
/// discovering it at the first dispatch.
///
/// @param path GGUF model to read.
/// @return 0 when the manifest validates, 1 otherwise.
int PrintModelSummary(const std::string& path);
}  // namespace muyo_llm
