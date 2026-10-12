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
/// **Scope:** the pre-tokenizer implements the Qwen2 pattern over ASCII. `\p{L}` and `\p{N}` are
/// ASCII letters and digits, and other scripts are not handled - they fall through the
/// "not space, not letter, not digit" branch and tokenize differently from a real Qwen2 tokenizer.
/// That is enough to drive an ASCII prompt end to end and to round-trip; it is not a general
/// tokenizer. Recorded here rather than discovered by a wrong token id later.
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

    /// Tokenize `text`.
    /// @param text Input text (ASCII is fully supported; other scripts are best-effort).
    /// @return The token ids.
    std::vector<int32_t> Encode(const std::string& text) const;

    /// Reconstruct text from token ids.
    /// @param ids Token ids.
    /// @return The decoded text.
    std::string Decode(const std::vector<int32_t>& ids) const;

private:
    /// GPT-2's byte-to-unicode table: printable bytes map to themselves, the rest to 256+n.
    void BuildByteTables();
    /// Pre-tokenize `text` into the pieces BPE runs on (ASCII subset of the Qwen2 pattern).
    std::vector<std::string> PreTokenize(const std::string& text) const;
    /// Byte-encode `piece` into the vocab's unicode representation.
    std::string ByteEncode(const std::string& piece) const;
    /// Byte-decode a vocab token back into raw bytes.
    std::string ByteDecode(const std::string& token) const;
    /// Apply ranked BPE merges to one byte-encoded piece and return its token ids.
    std::vector<int32_t> MergePiece(const std::string& encoded) const;

    std::vector<std::string> m_idToToken;
    std::unordered_map<std::string, int32_t> m_tokenToId;
    /// Ranked merges, keyed by `left + '\0' + right`. The NUL separator is safe because a symbol is
    /// a UTF-8 code point and a raw NUL never appears.
    std::unordered_map<std::string, int32_t> m_mergeRank;
    std::string m_byteToUnicode[256];
    std::unordered_map<uint32_t, uint8_t> m_codePointToByte;
};
}  // namespace muyo_llm
