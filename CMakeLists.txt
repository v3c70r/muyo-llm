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
# muyo_rg links muyo PUBLIC (#68). Both are named because the executable uses both: the
# graph, and the device/allocation singletons the graph reaches for until #53 injects them.
target_link_libraries(muyo_llm PRIVATE muyo_rg muyo SPIRV-Reflect ${Vulkan_LIBRARIES})
# SPIRV-Reflect is an undeclared dependency of muyo: ShaderReflectionFetcher.cpp lives in
# muyo and calls spvReflect*, but only the in-tree `tests` target links the library, so an
# embedder gets undefined references at link. Named here to build; filed as a finding. The
# target is directory-scoped in Muyo's CMake but its name is global, so this resolves.
target_compile_options(muyo_llm PRIVATE -Wall)

# ── Shaders ──────────────────────────────────────────────────────────────────
# The graph loads `<name>.spv` from `shaders/` relative to the working directory
# (ShaderAsset.h), so the SPIR-V lands next to the binary and the program is run from
# the build directory. llama.cpp's shaders arrive in M1; M0 compiles only our own.
find_program(GLSLC_EXECUTABLE NAMES glslc REQUIRED)
set(MUYO_LLM_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")

file(GLOB_RECURSE MUYO_LLM_SHADER_SOURCES CONFIGURE_DEPENDS
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

add_custom_target(muyo_llm_shaders ALL DEPENDS ${MUYO_LLM_SPIRV_FILES})
add_dependencies(muyo_llm muyo_llm_shaders)
