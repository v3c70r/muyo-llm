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

**M1 in progress.** llama.cpp's shaders run through the graph, each checked against a CPU
reference: `add.comp` (`C = A + B`), `mul_mat_vec.comp` (decode GEMV, F16 weights), `rms_norm.comp`,
`rope_neox.comp`, `swiglu.comp`, `get_rows.comp` (embedding lookup) and `argmax.comp` (greedy
sampling). `mul_mm.comp` compiles with its 74 specialization constants; correct GEMM dispatch is M2
work, as is `flash_attn.comp`.

**M2 complete.** A GGUF reader parses the header, metadata and tensor table and reads F32/F16 tensor
data; a GPT-2/Qwen2 byte-level BPE tokenizer reads the vocab and merges; `--embed <gguf> <token>`
loads a model's token embedding and gathers one row through the graph, verified against the file.
`--tokenize <gguf> <text>` round-trips and checks the known Qwen2 ids (`"Hello world!"` ->
`9707 1879 0`). The tokenizer's pre-tokenizer covers ASCII only.

The F16 target model is downloaded for M4 (the attention cache append, M3b, waits on Muyo #81):

```
curl -sL https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf \
    -o testdata/stories260K.gguf
$ ./muyo_llm --gguf testdata/stories260K.gguf
[gguf] general.architecture = llama
[gguf] tensor token_embd.weight  type=0 dims=64x512
$ ./muyo_llm --embed testdata/stories260K.gguf 42
[embed] OK: token 42's 64-value embedding loaded from testdata/stories260K.gguf matches the file
```

```
$ cd build && ./muyo_llm --add
[add] OK: llama.cpp add_f32 over 1000 elements matches the CPU reference
$ ./muyo_llm --matvec
[matvec] OK: llama.cpp mul_mat_vec (F16 weights) over 128x256 matches the CPU reference
$ ./muyo_llm --rmsnorm
[rmsnorm] OK: llama.cpp rms_norm over 8x512 matches the CPU reference
$ ./muyo_llm --rope
[rope] OK: llama.cpp rope_neox over 8 channels matches the CPU reference
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
