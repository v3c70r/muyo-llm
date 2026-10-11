#pragma once

#include <string>

namespace muyo_llm
{
/// Read a GGUF file and print its header, key metadata and tensor table.
///
/// Does not touch the GPU, so it runs without a device. Used to check the reader against a known
/// file (a tokenizer-only GGUF, whose token count and a few tokens can be spot-checked) before
/// anything is built on it.
/// @param path GGUF file to read.
/// @return 0 on success, 1 on a parse failure.
int PrintGgufSummary(const std::string& path);
}  // namespace muyo_llm
