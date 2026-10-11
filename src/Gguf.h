#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace muyo_llm
{
/// The GGUF value types, from `ggml/src/gguf.h`'s `gguf_type`.
enum class GgufType : uint32_t
{
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

/// One metadata value. Arrays keep their element type so the tokenizer arrays (`tokens`, `merges`)
/// and numeric arrays can be read without guessing.
using GgufScalar =
    std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float, bool, uint64_t, int64_t, double>;
using GgufArray =
    std::variant<std::vector<uint8_t>, std::vector<int8_t>, std::vector<uint16_t>, std::vector<int16_t>,
                 std::vector<uint32_t>, std::vector<int32_t>, std::vector<float>, std::vector<bool>,
                 std::vector<std::string>, std::vector<uint64_t>, std::vector<int64_t>, std::vector<double>>;
using GgufValue = std::variant<GgufScalar, std::string, GgufArray>;

/// One tensor's entry in the file's tensor table. The data lives at `fileOffset` after the
/// alignment padding that follows the header.
struct GgufTensorInfo
{
    std::string name;
    std::vector<uint64_t> dimensions;  ///< Fastest-varying dimension first, as GGUF stores it.
    uint32_t type = 0;                 ///< `ggml_type` enumerant.
    uint64_t offset = 0;               ///< Byte offset from the start of the data section.
};

/// A minimal GGUF reader: header, metadata and the tensor table.
///
/// Deliberately does not load tensor data. The plan only needs the metadata (tokenizer) and the
/// tensor manifest to slice weights later, and a full 6 GB model should not be read into memory to
/// answer "what is in it".
class GgufReader
{
public:
    /// Parse the file at `path`.
    /// @param path File to read.
    /// @throws std::runtime_error on a missing file, a bad magic, or a truncated/short read.
    explicit GgufReader(const std::string& path);

    /// @return The GGUF version the file declares (2 or 3 for the files in use).
    uint32_t Version() const { return m_version; }
    /// @return True when a metadata key is present.
    bool Has(const std::string& key) const;
    /// @return The value for `key`.
    /// @throws std::out_of_range when the key is absent.
    const GgufValue& Metadata(const std::string& key) const;
    /// @return The tensor table.
    const std::vector<GgufTensorInfo>& Tensors() const { return m_tensors; }
    /// @return The data-section offset, i.e. where tensor 0's bytes start (after alignment).
    uint64_t DataOffset() const { return m_dataOffset; }

    /// @return A metadata value as a string, or `fallback` when absent or not a string.
    std::string StringOr(const std::string& key, const std::string& fallback) const;
    /// @return The length of a string-array metadata value, or 0 when absent or not one.
    size_t StringArraySize(const std::string& key) const;
    /// @return Element `index` of a string-array metadata value.
    /// @throws std::out_of_range when the key is absent, not a string array, or out of range.
    const std::string& StringArrayAt(const std::string& key, size_t index) const;

private:
    uint32_t m_version = 0;
    std::vector<std::pair<std::string, GgufValue>> m_metadata;
    std::vector<GgufTensorInfo> m_tensors;
    uint64_t m_dataOffset = 0;
};
}  // namespace muyo_llm
