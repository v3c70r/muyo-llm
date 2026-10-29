#pragma once

#include <cstdint>

namespace muyo_llm
{
/// `add.comp`'s `push_constant` block, from `generic_binary_head.glsl`.
///
/// This is half of the shader's contract; the other half is the compile-time `-D` set in
/// `CMakeLists.txt` and the `heap[]` slot values the node fills. The layout is fixed by the
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

    /// Descriptor-heap indices in `llama_add_bindings` order (A, B, D, PARTIAL_SUMS). The graph is
    /// heap-only, so the node resolves them with `RenderGraphNodeContext::GetHeapIndex` and records
    /// them as push data; the shader indexes its `layout(descriptor_heap)` arrays with them.
    uint32_t heap[4];
};

/// 33 four-byte fields (29 shader fields + 4 heap indices). A change here is a change to a shader we
/// do not own.
static_assert(sizeof(LlamaAddPushConstants) == 33 * sizeof(uint32_t),
              "LlamaAddPushConstants no longer matches generic_binary_head.glsl's `parameter` block");

/// Descriptor-heap slots of `add.comp` / `generic_binary_head.glsl`, in one place so the node's
/// push-data table and the shader's `heap[]` indices cannot drift apart.
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

/// `mul_mat_vec.comp`'s `push_constant` block, from `mul_mat_vec_base.glsl`.
///
/// A GEMV: `d[row] = sum_k A[row, k] * B[k]`, one workgroup per output row group. `strideA`,
/// `strideB` and `strideD` are in elements - like `add.comp`'s `nb`, ggml passes tensor strides,
/// and the shader indexes typed arrays. `strideD` doubles as the row count the shader bounds
/// against. The batch fields are zero for the single-vector decode case.
struct LlamaMatVecPushConstants
{
    uint32_t ncols;  ///< K, the reduction dimension.
    uint32_t strideA;
    uint32_t strideB;
    uint32_t strideD;  ///< Also the output row count.

    uint32_t batchStrideA;
    uint32_t batchStrideB;
    uint32_t batchStrideD;

    uint32_t fusionFlags;  ///< Bias/scale fusion bits; 0 disables all of them.

    uint32_t baseWorkGroupY;
    uint32_t ne02;
    uint32_t ne12;
    uint32_t broadcast2;
    uint32_t broadcast3;

    /// Descriptor-heap indices in `llama_mat_vec_bindings` order (A, B, D, FUSE0, FUSE1).
    uint32_t heap[5];
};

/// 18 four-byte fields (13 shader fields + 5 heap indices), matching
/// `vk_mat_vec_push_constants` in ggml-vulkan-push-constants.h plus the heap table.
static_assert(sizeof(LlamaMatVecPushConstants) == 18 * sizeof(uint32_t),
              "LlamaMatVecPushConstants no longer matches mul_mat_vec_base.glsl's `parameter` block");

/// Descriptor-heap slots of `mul_mat_vec.comp` / `mul_mat_vec_iface.glsl`. `FUSE0`/`FUSE1` are
/// declared unconditionally and only read when the matching `fusionFlags` bit is set.
namespace llama_mat_vec_bindings
{
constexpr uint32_t A = 0;
constexpr uint32_t B = 1;
constexpr uint32_t D = 2;
constexpr uint32_t FUSE0 = 3;
constexpr uint32_t FUSE1 = 4;
}  // namespace llama_mat_vec_bindings

/// `constant_id` values `mul_mat_vec.comp` declares.
namespace llama_mat_vec_spec_constants
{
constexpr uint32_t BLOCK_SIZE = 0;  ///< Also `local_size_x_id`: the workgroup size.
constexpr uint32_t NUM_ROWS = 1;    ///< Rows one workgroup produces; the dispatch divides by it.
constexpr uint32_t NUM_COLS = 2;    ///< Column-groups per workgroup; 1 for the plain path.
}  // namespace llama_mat_vec_spec_constants

/// `fusionFlags` bits of `mul_mat_vec.comp`, from `ggml-vulkan-types.h`. A set bit makes the GEMV
/// add `FUSE0`/`FUSE1` at the output row, which is how the projections fold their bias in (so a
/// bias is a descriptor binding, not a second pass over the row).
namespace llama_mat_vec_fusion
{
constexpr uint32_t BIAS0 = 0x1;   ///< Add `FUSE0[d_offset + row]` to the result.
constexpr uint32_t BIAS1 = 0x2;   ///< Add `FUSE1[d_offset + row]` to the result.
constexpr uint32_t SCALE0 = 0x4;  ///< Multiply the result by `FUSE0` (batched index).
constexpr uint32_t SCALE1 = 0x8;  ///< Multiply the result by `FUSE1` (batched index).
}  // namespace llama_mat_vec_fusion

/// `rope_neox.comp`'s `push_constant` block, from `rope_params.glsl`. 30 four-byte words.
///
/// RoPE rotates each head's channels in place. The fields a caller must get right are `nDims`
/// (the channels rotated, not the whole head), `thetaScale` (`freqBase^(-2/nDims)`), and the
/// strides, which are in elements like every other ggml stride.
struct LlamaRopePushConstants
{
    uint32_t ropeMode;
    uint32_t nrows;
    uint32_t nDims;
    uint32_t nOffs;
    float freqScale;
    float freqBase;
    float extFactor;
    float attnFactor;
    float corrDims[2];
    float thetaScale;
    uint32_t hasFf;
    int32_t sections[4];
    uint32_t isImrope;
    uint32_t isBack;
    uint32_t setRowsStride;
    uint32_t ne00;
    uint32_t ne01;
    uint32_t ne02;
    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t aOffset;
    uint32_t dOffset;

    /// Descriptor-heap indices in `llama_rope_bindings` order (X, Y, Z, D, I).
    uint32_t heap[5];
};

/// 35 four-byte words (30 shader fields + 5 heap indices), matching
/// `vk_op_rope_push_constants` in ggml-vulkan-push-constants.h plus the heap table.
static_assert(sizeof(LlamaRopePushConstants) == 35 * sizeof(uint32_t),
              "LlamaRopePushConstants no longer matches rope_params.glsl's `rope_params` struct");

/// Descriptor-heap slots of `rope_head.glsl`.
namespace llama_rope_bindings
{
constexpr uint32_t X = 0;  ///< Input rows.
constexpr uint32_t Y = 1;  ///< Positions, `int` per row.
constexpr uint32_t Z = 2;  ///< Frequency factors, read only when `hasFf` is set.
constexpr uint32_t D = 3;  ///< Output rows.
constexpr uint32_t I = 4;  ///< `uvec2` set-rows indices, read only for the fused path.
}  // namespace llama_rope_bindings

/// `rope_params.rope_mode` values, from `ggml.h`'s `GGML_ROPE_TYPE_*`.
namespace llama_rope_modes
{
constexpr uint32_t NORMAL = 0;
constexpr uint32_t NEOX = 2;  ///< Half-split rotation; what Qwen and most modern models use.
}  // namespace llama_rope_modes

/// `glu_head.glsl`'s `push_constant` block (shared by `swiglu.comp`, `geglu.comp`, ...).
/// 27 four-byte words.
///
/// The `ne2_*mp` / `ne2_*L` pairs are magic numbers for the shader's `fastdiv`, which the host
/// computes with ggml's `init_fastdiv_values`. Reproducing them is not optional: `fastdiv` with
/// zeroed magic returns the input unchanged, so every element would land on row 0's address.
struct LlamaGluPushConstants
{
    uint32_t N;
    uint32_t ne00;
    uint32_t ne20;
    uint32_t mode;
    float alpha;
    float limit;
    uint32_t nb00, nb01, nb02, nb03;
    uint32_t nb10, nb11, nb12, nb13;
    uint32_t nb20, nb21, nb22, nb23;
    uint32_t ne21;
    uint32_t ne22;
    uint32_t misalignOffsets;
    uint32_t ne2_012mp, ne2_012L;
    uint32_t ne2_01mp, ne2_01L;
    uint32_t ne2_0mp, ne2_0L;

    /// Descriptor-heap indices in `llama_glu_bindings` order (A, B, D).
    uint32_t heap[3];
};

/// 30 four-byte words (27 shader fields + 3 heap indices), matching
/// `vk_op_glu_push_constants` in ggml-vulkan-push-constants.h plus the heap table.
static_assert(sizeof(LlamaGluPushConstants) == 30 * sizeof(uint32_t),
              "LlamaGluPushConstants no longer matches glu_head.glsl's `parameter` struct");

/// Descriptor-heap slots of `glu_head.glsl`.
namespace llama_glu_bindings
{
constexpr uint32_t A = 0;
constexpr uint32_t B = 1;  ///< The second operand; only read in split mode.
constexpr uint32_t D = 2;
}  // namespace llama_glu_bindings

/// Descriptor-heap slots of `copy.comp` / `generic_unary_head.glsl` (A, D).
namespace llama_copy_bindings
{
constexpr uint32_t A = 0;
constexpr uint32_t D = 1;
}  // namespace llama_copy_bindings

/// `glu_main.glsl`'s `p.mode` values: 0 default (one input split in half), 1 swapped, 2 split
/// (two separate inputs). A transformer's gate/up pair is split.
namespace llama_glu_modes
{
constexpr uint32_t DEFAULT_HALVES = 0;
constexpr uint32_t SWAPPED_HALVES = 1;
constexpr uint32_t SPLIT = 2;
}  // namespace llama_glu_modes

/// `generic_head.glsl`'s `push_constant` block, shared by the simple elementwise/reduction shaders
/// (`argmax.comp` among them). 6 four-byte words.
struct LlamaGenericHeadPushConstants
{
    uint32_t KX;  ///< Row length.
    uint32_t KY;  ///< Row count.
    float param1;
    float param2;
    float param3;
    float param4;

    /// Descriptor-heap indices in `llama_argmax_bindings` order (A, D).
    uint32_t heap[2];
};

/// 8 four-byte words (6 shader fields + 2 heap indices), matching
/// `vk_op_push_constants` in ggml-vulkan-push-constants.h plus the heap table.
static_assert(sizeof(LlamaGenericHeadPushConstants) == 8 * sizeof(uint32_t),
              "LlamaGenericHeadPushConstants no longer matches generic_head.glsl's `parameter` block");

/// Descriptor-heap slots of `argmax.comp`.
namespace llama_argmax_bindings
{
constexpr uint32_t A = 0;  ///< Input row (`float`).
constexpr uint32_t D = 1;  ///< Output index (`int`).
}  // namespace llama_argmax_bindings

/// `constant_id` values `argmax.comp` declares.
namespace llama_argmax_spec_constants
{
constexpr uint32_t BLOCK_SIZE = 0;  ///< Also `local_size_x_id`: the reduction width.
}  // namespace llama_argmax_spec_constants

/// `soft_max.comp`'s `push_constant` block. 17 four-byte words.
///
/// `KY = 0` disables the additive mask, which is the case for plain softmax over a row. `scale` is
/// applied inside (`v = a * scale`), so a scorer divides by `sqrt(headDim)` here rather than before.
struct LlamaSoftMaxPushConstants
{
    uint32_t KX;  ///< Row length (number of keys).
    uint32_t KY;  ///< 0 when there is no mask operand.
    uint32_t ne00;
    uint32_t ne01;
    uint32_t ne02;
    uint32_t ne12;
    uint32_t ne13;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    float scale;
    float maxBias;
    float m0;
    float m1;
    uint32_t nHeadLog2;
    uint32_t nrowsX;
    uint32_t hasSinks;

    /// Descriptor-heap indices in `llama_soft_max_bindings` order (X, Y, Z, D).
    uint32_t heap[4];
};

/// 21 four-byte words (17 shader fields + 4 heap indices), matching
/// `vk_op_soft_max_push_constants` in ggml-vulkan-push-constants.h plus the heap table.
static_assert(sizeof(LlamaSoftMaxPushConstants) == 21 * sizeof(uint32_t),
              "LlamaSoftMaxPushConstants no longer matches soft_max.comp's `parameter` block");

/// Descriptor-heap slots of `soft_max.comp`. Y and Z are declared unconditionally; Y is the additive
/// mask (read when `KY > 0`) and Z the attention sinks (read when `hasSinks != 0`).
namespace llama_soft_max_bindings
{
constexpr uint32_t X = 0;  ///< Input scores.
constexpr uint32_t Y = 1;  ///< Additive mask.
constexpr uint32_t Z = 2;  ///< Attention sinks.
constexpr uint32_t D = 3;  ///< Normalized output (read-write: the tail columns are scaled in place).
}  // namespace llama_soft_max_bindings

/// `constant_id` values `soft_max.comp` declares.
namespace llama_soft_max_spec_constants
{
constexpr uint32_t BLOCK_SIZE = 0;  ///< Also `local_size_x_id`: the reduction width.
}  // namespace llama_soft_max_spec_constants
}  // namespace muyo_llm
