#pragma once

#include <string>

namespace muyo_llm
{
/// Qwen2.5's ChatML framing.
///
/// Kept as literal pieces rather than a Jinja engine: the model's non-tool path in
/// `tokenizer.chat_template` renders to exactly these strings, and the alternative is a template
/// language for four lines. The pieces are fed to the tokenizer with special-token parsing on, so
/// `<|im_start|>`/`<|im_end|>` become their own ids (151644/151645 for Qwen2.5).
///
/// The ids this renders were checked against llama.cpp (its `--tokenize` and this repo's produce the
/// same sequence for a rendered turn).
namespace chat_template
{
/// The system prompt Qwen2.5's template falls back to when none is supplied.
inline constexpr char QWEN2_DEFAULT_SYSTEM[] = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

/// `<|im_start|>system\n{system}<|im_end|>\n` - the conversation prefix.
std::string Qwen2SystemStart(const std::string& system);

/// `<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n` - one user turn and the assistant
/// generation prompt.
std::string Qwen2UserTurn(const std::string& user);

/// `Qwen2SystemStart(system) + Qwen2UserTurn(user)`, the single-turn prompt.
std::string Qwen2ChatStart(const std::string& system, const std::string& user);
}  // namespace chat_template
}  // namespace muyo_llm
