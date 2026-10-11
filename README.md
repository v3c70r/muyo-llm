# muyo-llm

A lightweight LLM inference engine built on [Muyo](https://github.com/v3c70r/Muyo)'s
RenderGraph library.

It exists to be the second, non-renderer consumer of `muyo_rg`: a compute-only, windowless,
long-running workload that tests the library boundary the Muyo roadmap asks for. It is
architecture validation first and an inference engine second.

- **Scope (v1):** decode-only, F16 weights, Qwen2.5-0.5B, llama.cpp shaders.
- **Plan:** [`docs/Plan.md`](docs/Plan.md).
- **Findings:** [`docs/Findings.md`](docs/Findings.md).
- **Working agreement:** [`AGENTS.md`](AGENTS.md).

## Status

**M0 complete.** A headless host (device, allocator, resource and descriptor managers, no window)
runs one graph-owned storage buffer through a compute node and verifies the result on the host:

```
$ cd build && ./muyo_llm --smoke
[device] AMD Radeon Graphics (Vulkan 1.4.313), dedicated compute family present
[device] cooperative matrix: available
[graph] execution order: SmokeFill
[smoke] OK: 1024 elements written by the compute node and verified on the host
```

**M1 in progress.** llama.cpp's `add.comp` runs through the graph as `C = A + B` and is checked
against a CPU reference; `mul_mm.comp` compiles with its 74 specialization constants. Correct
matmul dispatch is M2 work.

```
$ cd build && ./muyo_llm --add
[graph] execution order: LlamaAdd
[add] OK: llama.cpp add_f32 over 1000 elements matches the CPU reference
```

## Build

Muyo is embedded as a submodule; a development checkout can be substituted with
`-DMUYO_SOURCE_DIR`. The Vulkan SDK must be on the environment, and configure needs network (VMA is
fetched).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DMUYO_SOURCE_DIR=/path/to/Muyo
cmake --build build --target muyo_llm -j"$(nproc)"
cd build && ./muyo_llm --smoke        # the graph loads shaders relative to the working directory
```

`scripts/sanity.sh` runs the format check and the build + smoke; `scripts/sanity.sh sanitize` adds
ASan/UBSan.

## Upstream state

The gaps that blocked M0 and M1 in the plan have closed:

- **Embeddable build** — [Muyo #68](https://github.com/v3c70r/Muyo/issues/68) landed (`0617102`).
  `add_subdirectory(<Muyo>)` plus `target_link_libraries(... muyo_rg)` configures and builds.
- **Specialization constants** — [Muyo #62](https://github.com/v3c70r/Muyo/issues/62), [PR #66](https://github.com/v3c70r/Muyo/pull/66).
- **Cooperative-matrix enablement** — [Muyo #63](https://github.com/v3c70r/Muyo/issues/63), PR #71.
  On this machine it is available through **AMDVLK** (11 property sets, `F16/F16->F32` 16x16x16), not
  through the default RADV driver; AMDVLK is opt-in via `VK_DRIVER_FILES`.

Open, and not blocking M0/M1: reading the cooperative-matrix property sets can crash under one
driver + validation-layer pairing ([Muyo #72](https://github.com/v3c70r/Muyo/issues/72)), and the
library boundary itself ([Muyo #53](https://github.com/v3c70r/Muyo/issues/53)).
