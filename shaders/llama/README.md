# Vendored llama.cpp Vulkan shaders

These files are copied verbatim from [llama.cpp](https://github.com/llama.cpp)'s Vulkan backend,
`ggml/src/ggml-vulkan/vulkan-shaders/`, at commit `edd6e2bbd` (MIT License, "Copyright (c) 2023-2026
The ggml authors"). See llama.cpp's `LICENSE` for the full text.

They are compiled here by our own CMake rules rather than by llama.cpp's generator, and driven
through Muyo's RenderGraph. This is deliberate: the point of the project is to reuse the shader
library as-is, contracts and all, and find where that is hard, so a local edit to a vendored file is
a finding rather than a fix.

Do not edit these files to make them fit. Each one has a **contract** — a set of `-D` defines, a
`push_constant` block layout, descriptor bindings, and specialization constants — that
`ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` and `ggml-vulkan/ggml-vulkan.cpp` define. Our
side of that contract lives in `src/LlamaContracts.h` and the per-shader compile rules in
`CMakeLists.txt`; a mismatch shows up as a wrong number, not an error.

## Vendored so far

| File | Why |
| --- | --- |
| `add.comp` | Elementwise binary op; the first real op through the graph (M1). |
| `mul_mm.comp` | Tiled GEMM; the matmul kernel. Compiled in M1 as evidence, dispatched in M2. |
| `types.glsl` | Included by both; ggml tensor type and block definitions. |
| `generic_binary_head.glsl` | `add.comp`'s push-constant block and index helpers. |
| `utils.glsl` | Index decomposition and fast division used by the head. |
| `dot_product_funcs.glsl` | `mul_mm.comp`'s inner-product helpers. |
| `mul_mm_funcs.glsl` | `mul_mm.comp`'s tile load/store helpers. |
| `mul_mm_id_funcs.glsl` | Included unconditionally by `mul_mm.comp` (guarded uses are inside). |
| `mul_mat_vec.comp` | GEMV; the decode workhorse (M1). |
| `mul_mat_vec_base.glsl` | Its push-constant block, reductions and offsets. |
| `mul_mat_vec_iface.glsl` | Its bindings and operand types. |
| `dequant_funcs.glsl` | Included by GEMV; the non-quant F32/F16 paths live here too. |
