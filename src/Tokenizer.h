#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace muyo_llm
{
class GgufReader;

/// A GPT-2/Qwen2-style byte-level BPE tokenizer, built from a GGUF file's tokenizer metadata.
///
/// The pipeline is: pre-tokenize the text into words, byte-encode each word with GPT-2's
/// byte-to-unicode map (so every byte is a printable code point, and the space byte is `Ġ`), then
/// apply the vocab's ranked merges, then look each piece up in the vocab.
///
/// **Scope:** the pre-tokenizer is llama.cpp's, called through the vendored `unicode_regex_split`
/// (`src/llama/`), so it is correct for any UTF-8 script and matches llama.cpp's split exactly. The
/// two regexes a GGUF's `tokenizer.ggml.pre` can select (`qwen2` and the GPT-2 default) are the
/// ones this repo's models use; an unknown value falls back to GPT-2.
class Tokenizer
{
public:
    /// Build the token and merge tables from `reader`'s `tokenizer.ggml.*` metadata.
    /// @param reader A GGUF file already parsed.
    /// @throws std::runtime_error when the tokens or merges arrays are missing.
    explicit Tokenizer(const GgufReader& reader);

    /// @return The vocabulary size.
    size_t VocabSize() const { return m_idToToken.size(); }
    /// @return The token string for an id, or an empty string when out of range.
    const std::string& TokenFor(size_t id) const;

    /// @return The raw bytes token `id` expands to, for streaming output. A vocabulary string that
    /// is GPT-2-byte-encoded is decoded back to bytes; a printable special token such as
    /// `<|im_end|>` maps to itself through the same table.
    std::string TokenBytes(int32_t id) const;

    /// @return True for a CONTROL or USER_DEFINED token (`<|im_start|>`, `<|im_end|>`). Such a token
    /// is framing, not text, and llama.cpp's default `token_to_piece` renders it as nothing.
    bool IsSpecial(int32_t id) const;

    /// Tokenize `text`.
    /// @param text Input text (any UTF-8 script).
    /// @param parseSpecial When true (the default) a vocabulary special token such as `<|im_start|>`
    ///        is recognized as a unit and emitted as its id, instead of being split by BPE.
    /// @return The token ids.
    std::vector<int32_t> Encode(const std::string& text, bool parseSpecial = true) const;

    /// Reconstruct text from token ids.
    /// @param ids Token ids.
    /// @return The decoded text.
    std::string Decode(const std::vector<int32_t>& ids) const;

private:
    /// One token the tokenizer emits as a unit and never merges (ChatML's `<|im_start|>`).
    struct SpecialToken
    {
        std::string text;
        int32_t id = -1;
    };

    /// GPT-2's byte-to-unicode table: printable bytes map to themselves, the rest to 256+n.
    void BuildByteTables();
    /// Pre-tokenize `text` into the pieces BPE runs on, with llama.cpp's `unicode_regex_split`.
    std::vector<std::string> PreTokenize(const std::string& text) const;
    /// Byte-encode `piece` into the vocab's unicode representation.
    std::string ByteEncode(const std::string& piece) const;
    /// Byte-decode a vocab token back into raw bytes.
    std::string ByteDecode(const std::string& token) const;
    /// Apply ranked BPE merges to one byte-encoded piece and return its token ids.
    std::vector<int32_t> MergePiece(const std::string& encoded) const;

    std::vector<std::string> m_idToToken;
    std::unordered_map<std::string, int32_t> m_tokenToId;
    /// The pre-tokenizer regex(es) for this vocabulary, from `tokenizer.ggml.pre`.
    std::vector<std::string> m_regexExprs;
    /// Tokens with type CONTROL or USER_DEFINED, matched before BPE when `parseSpecial` is set.
    std::vector<SpecialToken> m_specialTokens;
    /// Per-id marker for the same set, for the output path.
    std::vector<uint8_t> m_isSpecial;
    /// Ranked merges, keyed by `left + '\0' + right`. The NUL separator is safe because a symbol is
    /// a UTF-8 code point and a raw NUL never appears.
    std::unordered_map<std::string, int32_t> m_mergeRank;
    std::string m_byteToUnicode[256];
    std::unordered_map<uint32_t, uint8_t> m_codePointToByte;
};
}  // namespace muyo_llm
