#pragma once

#include <cstdint>

namespace muyo_llm
{
/// `add.comp`'s `push_constant` block, from `generic_binary_head.glsl`.
///
/// This is half of the shader's contract; the other half is the compile-time `-D` set in
/// `CMakeLists.txt` and the descriptor bindings the node declares. The layout is fixed by the
/// shader, so it is reproduced field for field and checked by `static_assert`. Getting a field
/// wrong does not fail — it reads the wrong stride and produces a wrong number.
///
/// `ne`/`nb` are ggml's dimension counts and strides for the three operands (0 = A, 1 = B,
/// 2 = D), most-minor dimension first. **`nb` is in elements, not bytes**: ggml passes
/// `tensor->nb[i] / type_size` (`ggml_vk_add` in ggml-vulkan.cpp), because the shader indexes a
/// typed array. Passing byte strides here writes every element to a quarter of the intended
/// positions - a wrong number, not an error, which is exactly the trap this header exists to catch.
/// `misalign_offsets` packs byte offsets for the three operands into three bytes
/// (`A << 16 | B << 8 | D`); zero when each buffer starts at offset 0, which is what explicit
/// bindings give.
struct LlamaAddPushConstants
{
    uint32_t ne;

    uint32_t ne00, ne01, ne02, ne03;
    uint32_t nb00, nb01, nb02, nb03;
    uint32_t ne10, ne11, ne12, ne13;
    uint32_t nb10, nb11, nb12, nb13;
    uint32_t ne20, ne21, ne22, ne23;
    uint32_t nb20, nb21, nb22, nb23;

    uint32_t misalignOffsets;

    float param1;
    float param2;
    int32_t param3;
};

/// 29 four-byte fields. A change here is a change to a shader we do not own.
static_assert(sizeof(LlamaAddPushConstants) == 29 * sizeof(uint32_t),
              "LlamaAddPushConstants no longer matches generic_binary_head.glsl's `parameter` block");

/// Descriptor bindings of `add.comp` / `generic_binary_head.glsl`, in one place so the node
/// declarations and the push-constant/define sets cannot drift apart.
namespace llama_add_bindings
{
constexpr uint32_t A = 0;
constexpr uint32_t B = 1;
constexpr uint32_t D = 2;
constexpr uint32_t PARTIAL_SUMS = 3;  ///< Declared unconditionally; only read when ADD_RMS is set.
}  // namespace llama_add_bindings

/// `constant_id` values `add.comp` declares.
namespace llama_add_spec_constants
{
constexpr uint32_t NOREPEAT = 0;  ///< True when A and B have the same shape, so B needs no modulo.
}  // namespace llama_add_spec_constants
}  // namespace muyo_llm
