#pragma once

#include <string>

namespace muyo_llm
{
/// Tokenize `text` with a GGUF file's tokenizer and check the encode/decode round-trip.
///
/// Needs no device. The round-trip is the primary check; the anchor is that `Encode("!")` is
/// `{0}` for the Qwen2/GPT-2 vocab, whose token 0 is `!`.
///
/// @param path GGUF file whose tokenizer metadata to use.
/// @param text Text to tokenize.
/// @return 0 on success, 1 on failure.
int RunTokenize(const std::string& path, const std::string& text);
}  // namespace muyo_llm
