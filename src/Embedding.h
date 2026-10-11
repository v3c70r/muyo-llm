#pragma once

#include <cstdint>
#include <string>

namespace muyo_llm
{
/// Load a model's token embedding from a GGUF file and run the `get_rows` lookup for one token
/// through the graph, verifying the GPU result against the tensor's bytes in the file.
///
/// This closes the loop between the two halves of model loading: the GGUF reader supplies the real
/// weights, the graph uploads them and gathers one row, and the comparison is against the file, not
/// against synthetic data.
///
/// Assumes the Muyo subsystems are already initialised (see `HeadlessContext`).
/// @param path GGUF file to read.
/// @param token Token id to embed.
/// @return 0 on success, 1 on failure.
int RunEmbedding(const std::string& path, int32_t token);
}  // namespace muyo_llm
