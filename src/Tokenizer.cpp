#include "Tokenizer.h"

#include <cctype>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "Gguf.h"

namespace muyo_llm
{
namespace
{
bool IsAsciiLetter(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool IsAsciiDigit(unsigned char c) { return c >= '0' && c <= '9'; }
bool IsAsciiSpace(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
bool IsNewline(unsigned char c) { return c == '\n' || c == '\r'; }

/// Append the UTF-8 encoding of `codePoint`.
void AppendUtf8(std::string& out, uint32_t codePoint)
{
    if (codePoint < 0x80)
    {
        out.push_back(static_cast<char>(codePoint));
    }
    else if (codePoint < 0x800)
    {
        out.push_back(static_cast<char>(0xC0 | (codePoint >> 6)));
        out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    }
    else if (codePoint < 0x10000)
    {
        out.push_back(static_cast<char>(0xE0 | (codePoint >> 12)));
        out.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    }
    else
    {
        out.push_back(static_cast<char>(0xF0 | (codePoint >> 18)));
        out.push_back(static_cast<char>(0x80 | ((codePoint >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    }
}

/// Decode one UTF-8 code point at `i`, advancing `i`.
uint32_t DecodeUtf8(const std::string& text, size_t& i)
{
    const auto b0 = static_cast<unsigned char>(text[i]);
    uint32_t codePoint = b0;
    size_t length = 1;
    if (b0 >= 0xF0)
    {
        codePoint = b0 & 0x07;
        length = 4;
    }
    else if (b0 >= 0xE0)
    {
        codePoint = b0 & 0x0F;
        length = 3;
    }
    else if (b0 >= 0xC0)
    {
        codePoint = b0 & 0x1F;
        length = 2;
    }
    for (size_t k = 1; k < length && i + k < text.size(); ++k)
    {
        codePoint = (codePoint << 6) | (static_cast<unsigned char>(text[i + k]) & 0x3F);
    }
    i += length;
    return codePoint;
}

/// Split a UTF-8 string into one-string-per-code-point symbols.
std::vector<std::string> SplitSymbols(const std::string& text)
{
    std::vector<std::string> symbols;
    size_t i = 0;
    while (i < text.size())
    {
        const size_t start = i;
        DecodeUtf8(text, i);
        symbols.push_back(text.substr(start, i - start));
    }
    return symbols;
}
}  // namespace

Tokenizer::Tokenizer(const GgufReader& reader)
{
    const size_t tokenCount = reader.StringArraySize("tokenizer.ggml.tokens");
    if (tokenCount == 0)
    {
        throw std::runtime_error("Tokenizer: GGUF has no tokenizer.ggml.tokens");
    }
    m_idToToken.resize(tokenCount);
    for (size_t i = 0; i < tokenCount; ++i)
    {
        m_idToToken[i] = reader.StringArrayAt("tokenizer.ggml.tokens", i);
        m_tokenToId.emplace(m_idToToken[i], static_cast<int32_t>(i));
    }

    const size_t mergeCount = reader.StringArraySize("tokenizer.ggml.merges");
    for (size_t i = 0; i < mergeCount; ++i)
    {
        const std::string& merge = reader.StringArrayAt("tokenizer.ggml.merges", i);
        const size_t space = merge.find(' ');
        if (space == std::string::npos)
        {
            continue;
        }
        // A symbol is one code point and no byte maps to a raw space (0x20 -> U+0120), so the
        // first space is the separator.
        m_mergeRank.emplace(merge.substr(0, space) + '\0' + merge.substr(space + 1), static_cast<int32_t>(i));
    }

    BuildByteTables();
}

void Tokenizer::BuildByteTables()
{
    // GPT-2 bytes_to_unicode: printable ASCII, then Latin-1 letters, map to themselves; every other
    // byte maps to 256+n. Notably the space byte (0x20) is not in the printable set, so it becomes
    // U+0120 'G with a dot' - which is why a raw space is never a symbol and can be the merge
    // separator.
    bool isPrintable[256] = {};
    for (int b = '!'; b <= '~'; ++b)
    {
        isPrintable[b] = true;
    }
    for (int b = 0xA1; b <= 0xAC; ++b)
    {
        isPrintable[b] = true;
    }
    for (int b = 0xAE; b <= 0xFF; ++b)
    {
        isPrintable[b] = true;
    }

    uint32_t nextCodePoint = 256;
    for (int b = 0; b < 256; ++b)
    {
        const uint32_t codePoint = isPrintable[b] ? static_cast<uint32_t>(b) : nextCodePoint++;
        std::string encoded;
        AppendUtf8(encoded, codePoint);
        m_byteToUnicode[b] = encoded;
        m_codePointToByte.emplace(codePoint, static_cast<uint8_t>(b));
    }
}

std::string Tokenizer::ByteEncode(const std::string& piece) const
{
    std::string encoded;
    for (unsigned char c : piece)
    {
        encoded += m_byteToUnicode[c];
    }
    return encoded;
}

std::string Tokenizer::ByteDecode(const std::string& token) const
{
    std::string decoded;
    size_t i = 0;
    while (i < token.size())
    {
        const uint32_t codePoint = DecodeUtf8(token, i);
        const auto it = m_codePointToByte.find(codePoint);
        if (it != m_codePointToByte.end())
        {
            decoded.push_back(static_cast<char>(it->second));
        }
    }
    return decoded;
}

std::vector<std::string> Tokenizer::PreTokenize(const std::string& text) const
{
    // The ASCII reading of Qwen2's pattern, alternatives tried in order at each position:
    //   contractions | optional-non-letter + letters | digit | optional-space + symbols + newlines
    //   | whitespace up to a newline | trailing whitespace | whitespace
    std::vector<std::string> pieces;
    size_t i = 0;
    const size_t n = text.size();
    while (i < n)
    {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        size_t match = 0;

        // 1. contractions: 's 't 're 've 'm 'll 'd
        if (c == '\'')
        {
            const std::string tail = text.substr(i, 3);
            if (tail.size() >= 2)
            {
                const char c1 = static_cast<char>(std::tolower(static_cast<unsigned char>(tail[1])));
                if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd')
                {
                    match = 2;
                }
                else if (tail.size() >= 3)
                {
                    const char c2 = static_cast<char>(std::tolower(static_cast<unsigned char>(tail[2])));
                    if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l'))
                    {
                        match = 3;
                    }
                }
            }
        }
        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        if (match == 0)
        {
            size_t j = i;
            if (!IsAsciiLetter(c) && !IsAsciiDigit(c) && !IsNewline(c))
            {
                ++j;
            }
            if (j < n && IsAsciiLetter(static_cast<unsigned char>(text[j])))
            {
                while (j < n && IsAsciiLetter(static_cast<unsigned char>(text[j])))
                {
                    ++j;
                }
                match = j - i;
            }
        }
        // 3. \p{N}
        if (match == 0 && IsAsciiDigit(c))
        {
            match = 1;
        }
        // 4. ?[^\s\p{L}\p{N}]+[\r\n]*
        if (match == 0)
        {
            size_t j = i;
            if (c == ' ')
            {
                ++j;
            }
            size_t symbols = j;
            while (j < n && !IsAsciiSpace(static_cast<unsigned char>(text[j])) &&
                   !IsAsciiLetter(static_cast<unsigned char>(text[j])) &&
                   !IsAsciiDigit(static_cast<unsigned char>(text[j])))
            {
                ++j;
            }
            if (j > symbols)
            {
                while (j < n && IsNewline(static_cast<unsigned char>(text[j])))
                {
                    ++j;
                }
                match = j - i;
            }
        }
        // 5. \s*[\r\n]+ : the longest whitespace run up to and including its last newline.
        if (match == 0 && IsAsciiSpace(c))
        {
            size_t j = i;
            size_t lastNewline = 0;
            bool hasNewline = false;
            while (j < n && IsAsciiSpace(static_cast<unsigned char>(text[j])))
            {
                if (IsNewline(static_cast<unsigned char>(text[j])))
                {
                    lastNewline = j + 1;
                    hasNewline = true;
                }
                ++j;
            }
            if (hasNewline)
            {
                match = lastNewline - i;
            }
        }
        // 6. \s+(?!\S) : trailing whitespace.
        if (match == 0 && IsAsciiSpace(c))
        {
            size_t j = i;
            while (j < n && IsAsciiSpace(static_cast<unsigned char>(text[j])))
            {
                ++j;
            }
            if (j >= n)
            {
                match = j - i;
            }
        }
        // 7. \s+
        if (match == 0 && IsAsciiSpace(c))
        {
            match = 1;
        }
        if (match == 0)
        {
            match = 1;  // Unmatched byte: emit it alone rather than loop forever.
        }
        pieces.push_back(text.substr(i, match));
        i += match;
    }
    return pieces;
}

std::vector<int32_t> Tokenizer::MergePiece(const std::string& encoded) const
{
    std::vector<std::string> symbols = SplitSymbols(encoded);
    while (symbols.size() > 1)
    {
        int32_t bestRank = INT32_MAX;
        size_t bestIndex = 0;
        for (size_t i = 0; i + 1 < symbols.size(); ++i)
        {
            const auto it = m_mergeRank.find(symbols[i] + '\0' + symbols[i + 1]);
            if (it != m_mergeRank.end() && it->second < bestRank)
            {
                bestRank = it->second;
                bestIndex = i;
            }
        }
        if (bestRank == INT32_MAX)
        {
            break;
        }
        symbols[bestIndex] += symbols[bestIndex + 1];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(bestIndex) + 1);
    }

    std::vector<int32_t> ids;
    for (const std::string& symbol : symbols)
    {
        const auto it = m_tokenToId.find(symbol);
        if (it != m_tokenToId.end())
        {
            ids.push_back(it->second);
            continue;
        }
        // Fall back to per-code-point ids; every single byte should be in the vocab.
        for (const std::string& codePoint : SplitSymbols(symbol))
        {
            const auto single = m_tokenToId.find(codePoint);
            if (single != m_tokenToId.end())
            {
                ids.push_back(single->second);
            }
        }
    }
    return ids;
}

std::vector<int32_t> Tokenizer::Encode(const std::string& text) const
{
    std::vector<int32_t> ids;
    for (const std::string& piece : PreTokenize(text))
    {
        const std::vector<int32_t> pieceIds = MergePiece(ByteEncode(piece));
        ids.insert(ids.end(), pieceIds.begin(), pieceIds.end());
    }
    return ids;
}

std::string Tokenizer::Decode(const std::vector<int32_t>& ids) const
{
    std::string text;
    for (int32_t id : ids)
    {
        if (id >= 0 && static_cast<size_t>(id) < m_idToToken.size())
        {
            text += ByteDecode(m_idToToken[static_cast<size_t>(id)]);
        }
    }
    return text;
}

const std::string& Tokenizer::TokenFor(size_t id) const
{
    static const std::string empty;
    return id < m_idToToken.size() ? m_idToToken[id] : empty;
}
}  // namespace muyo_llm
