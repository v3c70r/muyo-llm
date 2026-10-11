cmake_minimum_required(VERSION 3.19)
project(muyo_llm CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Muyo, the library under test ─────────────────────────────────────────────
# The submodule is the shipping path. MUYO_SOURCE_DIR overrides it for local work
# against a development checkout, so iterating on Muyo itself does not require
# re-initialising submodules. Relies on Muyo being embeddable (#68): it uses its own
# root for every path and guards its development-only targets behind MUYO_TOP_LEVEL.
set(MUYO_SOURCE_DIR "" CACHE PATH "Muyo checkout to build against (default: thirdparty/Muyo)")
if (MUYO_SOURCE_DIR STREQUAL "")
    set(MUYO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/Muyo")
endif()
if (NOT EXISTS "${MUYO_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "Muyo not found at ${MUYO_SOURCE_DIR}. Initialise the submodule "
        "(git submodule update --init --recursive) or pass -DMUYO_SOURCE_DIR=<path>.")
endif()
add_subdirectory("${MUYO_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/muyo")

find_package(Vulkan REQUIRED)

# ── muyo_llm ─────────────────────────────────────────────────────────────────
file(GLOB_RECURSE MUYO_LLM_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")

# Sanitizer configuration, mirroring Muyo's. Muyo's flags are directory-scoped to its own
# targets and do not reach an embedder, so the consumer adds its own for `scripts/sanity.sh
# sanitize`. Same generator expression, so only CMAKE_BUILD_TYPE=Sanitize turns it on. Both
# calls are directory-scoped and apply only to targets created after them, so they come before
# add_executable; link options in particular must, or the sanitizer runtime is never linked and
# Muyo's instrumented objects fail with undefined `__asan_*` / `__ubsan_*` references.
set(SANITIZE_CONDITION "$<CONFIG:Sanitize>")
add_compile_options("$<${SANITIZE_CONDITION}:-fsanitize=address,undefined>"
                    "$<${SANITIZE_CONDITION}:-fno-omit-frame-pointer>")
add_link_options("$<${SANITIZE_CONDITION}:-fsanitize=address,undefined>")

add_executable(muyo_llm ${MUYO_LLM_SOURCES})
target_include_directories(muyo_llm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
# muyo_rg links muyo PUBLIC (#68). SPIRV-Reflect is no longer named: Muyo declares it as a PUBLIC
# dependency of muyo now (its commit 6f8d65b, the fix for muyo-llm finding F1), so an embedder
# inherits it and linking it here would be a workaround for a bug that no longer exists.
target_link_libraries(muyo_llm PRIVATE muyo_rg muyo ${Vulkan_LIBRARIES})
target_compile_options(muyo_llm PRIVATE -Wall)

# ── Shaders ──────────────────────────────────────────────────────────────────
# The graph loads `<name>.spv` from `shaders/` relative to the working directory
# (ShaderAsset.h), so the SPIR-V lands next to the binary and the program is run from
# the build directory. llama.cpp's shaders arrive in M1; M0 compiles only our own.
find_program(GLSLC_EXECUTABLE NAMES glslc REQUIRED)
set(MUYO_LLM_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")

# Own shaders: plain, one stage, no defines. Not recursive, so the vendored llama.cpp
# shaders under shaders/llama/ (which each need their own -D set) are handled separately below.
file(GLOB MUYO_LLM_SHADER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp")

set(MUYO_LLM_SPIRV_FILES "")
foreach (SHADER_SOURCE ${MUYO_LLM_SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
    set(SPIRV "${MUYO_LLM_SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${MUYO_LLM_SHADER_OUTPUT_DIR}"
        COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.3 -fshader-stage=compute ${SHADER_SOURCE} -o ${SPIRV}
        DEPENDS ${SHADER_SOURCE}
        COMMENT "Compiling ${SHADER_NAME}"
        VERBATIM)
    list(APPEND MUYO_LLM_SPIRV_FILES ${SPIRV})
endforeach()

# ── Vendored llama.cpp shaders ───────────────────────────────────────────────
# Each needs its own define set, taken verbatim from llama.cpp's generator
# (ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp). The defines are the shader's
# contract: the graph reflects bindings, push constants and specialization constants,
# but the operand types are chosen here at compile time. Target vulkan1.2 to match
# llama.cpp's non-coopmat path; cooperative-matrix variants use vulkan1.3.
set(LLAMA_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders/llama")

# Compile one vendored shader. NAME is the output stem, so `add_f32` produces
# `add_f32.comp.spv`, which the graph loads as `add_f32.comp`. SOURCE is the .comp under
# shaders/llama; DEFINES is a `;`-separated -D list; remaining arguments are extra deps.
function(muyo_llm_llama_shader name source defines)
    set(flags "")
    foreach (def ${defines})
        list(APPEND flags "-D${def}")
    endforeach()
    set(spv "${MUYO_LLM_SHADER_OUTPUT_DIR}/${name}.comp.spv")
    add_custom_command(
        OUTPUT ${spv}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${MUYO_LLM_SHADER_OUTPUT_DIR}"
        COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.2 -fshader-stage=compute ${flags}
                "${LLAMA_SHADER_DIR}/${source}" -o ${spv}
        DEPENDS "${LLAMA_SHADER_DIR}/${source}" ${ARGN}
        COMMENT "Compiling ${name}.comp (llama.cpp)"
        VERBATIM)
    set_property(GLOBAL APPEND PROPERTY MUYO_LLM_LLAMA_SPIRV ${spv})
endfunction()

# add_f32: gen.cpp's loop over {"add", ...} for src0_f16=src1_f16=dst_f16=false.
muyo_llm_llama_shader(add_f32 add.comp
    "A_TYPE=float;B_TYPE=float;D_TYPE=float;FLOAT_TYPE=float;ADD_RMS=0"
    "${LLAMA_SHADER_DIR}/types.glsl" "${LLAMA_SHADER_DIR}/generic_binary_head.glsl"
    "${LLAMA_SHADER_DIR}/utils.glsl")

# rms_norm_f32: gen.cpp's rms_norm_f32. The `mul` variant is the same SPIR-V with the
# `do_multiply` specialization constant set to 1, which the node supplies.
muyo_llm_llama_shader(rms_norm_f32 rms_norm.comp
    "A_TYPE=float;B_TYPE=float;D_TYPE=float;FLOAT_TYPE=float;FLOAT_TYPEV2=vec2"
    "${LLAMA_SHADER_DIR}/types.glsl" "${LLAMA_SHADER_DIR}/generic_binary_head.glsl"
    "${LLAMA_SHADER_DIR}/utils.glsl")

# rope_neox_f32: gen.cpp's rope_neox_f32. NEOX (half-split) rotation, which is what Qwen uses.
muyo_llm_llama_shader(rope_neox_f32 rope_neox.comp
    "A_TYPE=float;ROPE_D_TYPE=float"
    "${LLAMA_SHADER_DIR}/types.glsl" "${LLAMA_SHADER_DIR}/rope_head.glsl"
    "${LLAMA_SHADER_DIR}/rope_params.glsl" "${LLAMA_SHADER_DIR}/rope_funcs.glsl")

# swiglu_f32: the gated MLP activation, `silu(a) * b`. The split mode (two operands) is what a
# transformer's gate/up pair uses.
muyo_llm_llama_shader(swiglu_f32 swiglu.comp
    "A_TYPE=float;D_TYPE=float"
    "${LLAMA_SHADER_DIR}/types.glsl" "${LLAMA_SHADER_DIR}/glu_head.glsl"
    "${LLAMA_SHADER_DIR}/glu_main.glsl" "${LLAMA_SHADER_DIR}/utils.glsl")

# get_rows_f32: the embedding lookup (and any row gather). Weight rows F32, indices int.
muyo_llm_llama_shader(get_rows_f32 get_rows.comp
    "DATA_A_F32;B_TYPE=int;D_TYPE=float;TEMP_TYPE=FLOAT_TYPE;FLOAT_TYPE=float;FLOAT_TYPEV2=vec2"
    "${LLAMA_SHADER_DIR}/types.glsl" "${LLAMA_SHADER_DIR}/generic_binary_head.glsl"
    "${LLAMA_SHADER_DIR}/utils.glsl")

# argmax_f32: greedy sampling - the index of the largest logit.
muyo_llm_llama_shader(argmax_f32 argmax.comp
    "A_TYPE=float;D_TYPE=int;FLOAT_TYPE=float;FLOAT_TYPEV2=vec2"
    "${LLAMA_SHADER_DIR}/types.glsl" "${LLAMA_SHADER_DIR}/generic_head.glsl")

# soft_max_f32: the attention-weights normalization, without an ALiBi mask (KY = 0).
muyo_llm_llama_shader(soft_max_f32 soft_max.comp
    "A_TYPE=float;B_TYPE=float;D_TYPE=float;FLOAT_TYPE=float;FLOAT_TYPEV2=vec2"
    "${LLAMA_SHADER_DIR}/types.glsl")

# mul_mat_vec_f16_f32_f32: the decode GEMV - F16 weights, F32 activation, F32 output. The
# non-subgroup path (USE_SUBGROUP_ADD undefined), so the reduction is the shared-memory
# tree and BLOCK_SIZE need not match the subgroup size.
muyo_llm_llama_shader(mul_mat_vec_f16_f32_f32 mul_mat_vec.comp
    "DATA_A_F16;B_TYPE=float;B_TYPEV2=vec2;B_TYPEV4=vec4;D_TYPE=float;FLOAT_TYPE=float;FLOAT_TYPEV2=vec2"
    "${LLAMA_SHADER_DIR}/mul_mat_vec_base.glsl" "${LLAMA_SHADER_DIR}/mul_mat_vec_iface.glsl"
    "${LLAMA_SHADER_DIR}/dequant_funcs.glsl" "${LLAMA_SHADER_DIR}/types.glsl")

# mul_mat_vec_f32_f32_f32: the same GEMV with F32 weights.
muyo_llm_llama_shader(mul_mat_vec_f32_f32_f32 mul_mat_vec.comp
    "DATA_A_F32;B_TYPE=float;B_TYPEV2=vec2;B_TYPEV4=vec4;D_TYPE=float;FLOAT_TYPE=float;FLOAT_TYPEV2=vec2"
    "${LLAMA_SHADER_DIR}/mul_mat_vec_base.glsl" "${LLAMA_SHADER_DIR}/mul_mat_vec_iface.glsl"
    "${LLAMA_SHADER_DIR}/dequant_funcs.glsl" "${LLAMA_SHADER_DIR}/types.glsl")

# mul_mm_f32_f16: the F32-A x F16-B GEMM variant, compiled as evidence that glslc and the
# graph's reflection handle its 74 specialization constants. Its tile spec-constant values
# and grid contract are the matmul work in M2, so this variant is not dispatched yet.
muyo_llm_llama_shader(mul_mm_f32_f16 mul_mm.comp
    "DATA_A_F32;LOAD_VEC_A=8;LOAD_VEC_B=8;B_TYPE=mat2x4;B_TYPE_SCALAR=float16_t;B_TYPEV4=f16vec4;D_TYPE=float;FLOAT_TYPE=float;FLOAT_TYPEV2=vec2;FLOAT_TYPEV4=vec4;FLOAT_TYPEV8=mat2x4;ACC_TYPE=float;ACC_TYPEV2=vec2;ACC_TYPE_VEC4=vec4"
    "${LLAMA_SHADER_DIR}/types.glsl" "${LLAMA_SHADER_DIR}/dot_product_funcs.glsl"
    "${LLAMA_SHADER_DIR}/mul_mm_funcs.glsl" "${LLAMA_SHADER_DIR}/mul_mm_id_funcs.glsl")

get_property(MUYO_LLM_LLAMA_SPIRV GLOBAL PROPERTY MUYO_LLM_LLAMA_SPIRV)
list(APPEND MUYO_LLM_SPIRV_FILES ${MUYO_LLM_LLAMA_SPIRV})

add_custom_target(muyo_llm_shaders ALL DEPENDS ${MUYO_LLM_SPIRV_FILES})
add_dependencies(muyo_llm muyo_llm_shaders)