#include "ChatTemplate.h"

#include <string>

namespace muyo_llm
{
namespace chat_template
{
std::string Qwen2SystemStart(const std::string& system)
{
    std::string text;
    text += "<|im_start|>system\n";
    text += system;
    text += "<|im_end|>\n";
    return text;
}

std::string Qwen2UserTurn(const std::string& user)
{
    std::string text;
    text += "<|im_start|>user\n";
    text += user;
    text += "<|im_end|>\n";
    text += "<|im_start|>assistant\n";
    return text;
}

std::string Qwen2ChatStart(const std::string& system, const std::string& user)
{
    return Qwen2SystemStart(system) + Qwen2UserTurn(user);
}
}  // namespace chat_template
}  // namespace muyo_llm
