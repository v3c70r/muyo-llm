#include "Gguf.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace muyo_llm
{
namespace
{
constexpr uint32_t K_GGUF_MAGIC = 0x46554747;  // "GGUF" little-endian.

template <typename T>
T ReadPod(std::ifstream& stream)
{
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!stream)
    {
        throw std::runtime_error("GGUF: unexpected end of file");
    }
    return value;
}

std::string ReadString(std::ifstream& stream)
{
    const uint64_t length = ReadPod<uint64_t>(stream);
    std::string value(length, '\0');
    if (length > 0)
    {
        stream.read(value.data(), static_cast<std::streamsize>(length));
        if (!stream)
        {
            throw std::runtime_error("GGUF: truncated string");
        }
    }
    return value;
}

GgufScalar ReadScalar(std::ifstream& stream, GgufType type)
{
    switch (type)
    {
        case GgufType::UINT8:
            return ReadPod<uint8_t>(stream);
        case GgufType::INT8:
            return ReadPod<int8_t>(stream);
        case GgufType::UINT16:
            return ReadPod<uint16_t>(stream);
        case GgufType::INT16:
            return ReadPod<int16_t>(stream);
        case GgufType::UINT32:
            return ReadPod<uint32_t>(stream);
        case GgufType::INT32:
            return ReadPod<int32_t>(stream);
        case GgufType::FLOAT32:
            return ReadPod<float>(stream);
        case GgufType::BOOL:
            return ReadPod<uint8_t>(stream) != 0;
        case GgufType::UINT64:
            return ReadPod<uint64_t>(stream);
        case GgufType::INT64:
            return ReadPod<int64_t>(stream);
        case GgufType::FLOAT64:
            return ReadPod<double>(stream);
        default:
            throw std::runtime_error("GGUF: scalar read of a non-scalar type");
    }
}

template <typename T, GgufType Type>
void ReadNumericArray(std::ifstream& stream, uint64_t count, GgufArray& out)
{
    std::vector<T> values(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i)
    {
        if constexpr (Type == GgufType::BOOL)
        {
            values[i] = ReadPod<uint8_t>(stream) != 0;
        }
        else
        {
            values[i] = ReadPod<T>(stream);
        }
    }
    out = std::move(values);
}

GgufArray ReadArray(std::ifstream& stream)
{
    const auto elementType = static_cast<GgufType>(ReadPod<uint32_t>(stream));
    const uint64_t count = ReadPod<uint64_t>(stream);
    GgufArray array;
    if (elementType == GgufType::STRING)
    {
        std::vector<std::string> strings(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i)
        {
            strings[i] = ReadString(stream);
        }
        array = std::move(strings);
        return array;
    }
    switch (elementType)
    {
        case GgufType::UINT8:
            ReadNumericArray<uint8_t, GgufType::UINT8>(stream, count, array);
            break;
        case GgufType::INT8:
            ReadNumericArray<int8_t, GgufType::INT8>(stream, count, array);
            break;
        case GgufType::UINT16:
            ReadNumericArray<uint16_t, GgufType::UINT16>(stream, count, array);
            break;
        case GgufType::INT16:
            ReadNumericArray<int16_t, GgufType::INT16>(stream, count, array);
            break;
        case GgufType::UINT32:
            ReadNumericArray<uint32_t, GgufType::UINT32>(stream, count, array);
            break;
        case GgufType::INT32:
            ReadNumericArray<int32_t, GgufType::INT32>(stream, count, array);
            break;
        case GgufType::FLOAT32:
            ReadNumericArray<float, GgufType::FLOAT32>(stream, count, array);
            break;
        case GgufType::BOOL:
            ReadNumericArray<bool, GgufType::BOOL>(stream, count, array);
            break;
        case GgufType::UINT64:
            ReadNumericArray<uint64_t, GgufType::UINT64>(stream, count, array);
            break;
        case GgufType::INT64:
            ReadNumericArray<int64_t, GgufType::INT64>(stream, count, array);
            break;
        case GgufType::FLOAT64:
            ReadNumericArray<double, GgufType::FLOAT64>(stream, count, array);
            break;
        default:
            throw std::runtime_error("GGUF: unsupported array element type");
    }
    return array;
}

GgufValue ReadValue(std::ifstream& stream, GgufType type)
{
    if (type == GgufType::STRING)
    {
        return ReadString(stream);
    }
    if (type == GgufType::ARRAY)
    {
        return ReadArray(stream);
    }
    return ReadScalar(stream, type);
}

uint64_t AlignUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }
}  // namespace

GgufReader::GgufReader(const std::string& path)
{
    m_path = path;
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("GGUF: cannot open " + path);
    }

    const uint32_t magic = ReadPod<uint32_t>(stream);
    if (magic != K_GGUF_MAGIC)
    {
        throw std::runtime_error("GGUF: bad magic in " + path);
    }
    m_version = ReadPod<uint32_t>(stream);
    const uint64_t tensorCount = ReadPod<uint64_t>(stream);
    const uint64_t metadataCount = ReadPod<uint64_t>(stream);

    m_metadata.reserve(static_cast<size_t>(metadataCount));
    for (uint64_t i = 0; i < metadataCount; ++i)
    {
        std::string key = ReadString(stream);
        const auto type = static_cast<GgufType>(ReadPod<uint32_t>(stream));
        m_metadata.emplace_back(std::move(key), ReadValue(stream, type));
    }

    m_tensors.reserve(static_cast<size_t>(tensorCount));
    for (uint64_t i = 0; i < tensorCount; ++i)
    {
        GgufTensorInfo info;
        info.name = ReadString(stream);
        const uint32_t dimensionCount = ReadPod<uint32_t>(stream);
        info.dimensions.resize(dimensionCount);
        for (uint32_t d = 0; d < dimensionCount; ++d)
        {
            info.dimensions[d] = ReadPod<uint64_t>(stream);
        }
        info.type = ReadPod<uint32_t>(stream);
        info.offset = ReadPod<uint64_t>(stream);
        m_tensors.push_back(std::move(info));
    }

    const uint64_t alignment = [&]
    {
        const auto it = std::find_if(m_metadata.begin(), m_metadata.end(),
                                     [](const auto& entry) { return entry.first == "general.alignment"; });
        if (it == m_metadata.end())
        {
            return static_cast<uint64_t>(32);
        }
        if (const auto* scalar = std::get_if<GgufScalar>(&it->second))
        {
            return static_cast<uint64_t>(std::visit([](auto v) { return static_cast<uint64_t>(v); }, *scalar));
        }
        return static_cast<uint64_t>(32);
    }();

    m_dataOffset = AlignUp(static_cast<uint64_t>(stream.tellg()), alignment);
}

bool GgufReader::Has(const std::string& key) const
{
    return std::any_of(m_metadata.begin(), m_metadata.end(), [&key](const auto& entry) { return entry.first == key; });
}

const GgufValue& GgufReader::Metadata(const std::string& key) const
{
    const auto it =
        std::find_if(m_metadata.begin(), m_metadata.end(), [&key](const auto& entry) { return entry.first == key; });
    if (it == m_metadata.end())
    {
        throw std::out_of_range("GGUF: no metadata key '" + key + "'");
    }
    return it->second;
}

std::string GgufReader::StringOr(const std::string& key, const std::string& fallback) const
{
    const auto it =
        std::find_if(m_metadata.begin(), m_metadata.end(), [&key](const auto& entry) { return entry.first == key; });
    if (it == m_metadata.end())
    {
        return fallback;
    }
    if (const auto* value = std::get_if<std::string>(&it->second))
    {
        return *value;
    }
    return fallback;
}

size_t GgufReader::StringArraySize(const std::string& key) const
{
    const auto it =
        std::find_if(m_metadata.begin(), m_metadata.end(), [&key](const auto& entry) { return entry.first == key; });
    if (it == m_metadata.end())
    {
        return 0;
    }
    if (const auto* array = std::get_if<GgufArray>(&it->second))
    {
        if (const auto* strings = std::get_if<std::vector<std::string>>(array))
        {
            return strings->size();
        }
    }
    return 0;
}

const std::string& GgufReader::StringArrayAt(const std::string& key, size_t index) const
{
    const auto it =
        std::find_if(m_metadata.begin(), m_metadata.end(), [&key](const auto& entry) { return entry.first == key; });
    if (it == m_metadata.end())
    {
        throw std::out_of_range("GGUF: no metadata key '" + key + "'");
    }
    const auto* array = std::get_if<GgufArray>(&it->second);
    if (array == nullptr)
    {
        throw std::out_of_range("GGUF: key '" + key + "' is not an array");
    }
    const auto* strings = std::get_if<std::vector<std::string>>(array);
    if (strings == nullptr)
    {
        throw std::out_of_range("GGUF: key '" + key + "' is not a string array");
    }
    return strings->at(index);
}

const GgufTensorInfo* GgufReader::FindTensor(const std::string& name) const
{
    const auto it = std::find_if(m_tensors.begin(), m_tensors.end(),
                                 [&name](const GgufTensorInfo& tensor) { return tensor.name == name; });
    return it == m_tensors.end() ? nullptr : &*it;
}

std::vector<uint8_t> GgufReader::ReadTensorData(const std::string& name) const
{
    const GgufTensorInfo* info = FindTensor(name);
    if (info == nullptr)
    {
        throw std::runtime_error("GGUF: no tensor '" + name + "'");
    }

    // Only the element-integer types the engine can upload today. A quantized tensor's layout is a
    // separate concern, so it throws here rather than handing back bytes to misinterpret.
    uint64_t elementSize = 0;
    switch (info->type)
    {
        case 0:  // GGML_TYPE_F32
            elementSize = 4;
            break;
        case 1:  // GGML_TYPE_F16
            elementSize = 2;
            break;
        default:
            throw std::runtime_error("GGUF: tensor '" + name + "' has type " + std::to_string(info->type) +
                                     ", which is not F32 or F16");
    }

    uint64_t elementCount = 1;
    for (uint64_t dimension : info->dimensions)
    {
        elementCount *= dimension;
    }
    const uint64_t byteCount = elementCount * elementSize;

    std::ifstream stream(m_path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("GGUF: cannot reopen " + m_path);
    }
    stream.seekg(static_cast<std::streamoff>(m_dataOffset + info->offset));
    std::vector<uint8_t> bytes(static_cast<size_t>(byteCount));
    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(byteCount));
    if (!stream)
    {
        throw std::runtime_error("GGUF: truncated data for tensor '" + name + "'");
    }
    return bytes;
}
}  // namespace muyo_llm
