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

**M0 complete.** A headless host (device, allocator, resource manager, descriptor heap, no window)
runs one graph-owned storage buffer through a compute node and verifies the result on the host:

```
$ cd build && ./muyo_llm --smoke
[device] llvmpipe (LLVM 21.1.8, 256 bits) (Vulkan 1.4.354), single queue family
[device] cooperative matrix: available
[graph] execution order: SmokeFill
[smoke] OK: 1024 elements written by the compute node and verified on the host
```

**M1 complete.** llama.cpp's shaders run through the graph, each checked against a CPU
reference: `add.comp` (`C = A + B`), `mul_mat_vec.comp` (decode GEMV, F16 weights), `rms_norm.comp`,
`rope_neox.comp`, `swiglu.comp`, `get_rows.comp` (embedding lookup) and `argmax.comp` (greedy
sampling). `mul_mm.comp` compiles with its 74 specialization constants; correct GEMM dispatch is M2
work, as is `flash_attn.comp`.

**M2 complete.** A GGUF reader parses the header, metadata and tensor table and reads F32/F16 tensor
data; a GPT-2/Qwen2 byte-level BPE tokenizer reads the vocab and merges; `--embed <gguf> <token>`
loads a model's token embedding and gathers one row through the graph, verified against the file.
`--tokenize <gguf> <text>` round-trips and checks the known Qwen2 ids (`"Hello world!"` ->
`9707 1879 0`). `--model <gguf>` validates the tensor manifest against the Qwen2 layout - all 291
tensors of qwen2.5-0.5b-instruct-fp16, with their shapes and dtypes - and prints the config the
forward pass needs. The tokenizer's pre-tokenizer covers ASCII only.

The F16 target model is downloaded for M4:

```
curl -sL https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf \
    -o testdata/stories260K.gguf
$ ./muyo_llm --gguf testdata/stories260K.gguf
[gguf] general.architecture = llama
[gguf] tensor token_embd.weight  type=0 dims=64x512
$ ./muyo_llm --embed testdata/stories260K.gguf 42
[embed] OK: token 42's 64-value embedding loaded from testdata/stories260K.gguf matches the file
$ ./muyo_llm --model testdata/qwen2.5-0.5b-instruct-fp16.gguf
[model] architecture=qwen2 blocks=24 hidden=896 ffn=4864 heads=14 kv_heads=2
[model] OK: manifest matches the qwen2 layout for 24 layers
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
$ ./muyo_llm --attention
[attention] OK: QK^T -> softmax -> V matches the CPU reference
$ ./muyo_llm --layer
[layer] OK: the 14-node transformer layer matches the CPU reference
$ ./muyo_llm --kvcache
[kvcache] OK: token 1 at position 1 over 2 cached keys matches the CPU reference
```

**M3 complete.** The fourteen-node transformer layer runs end to end (`--layer`), and the KV-cache
append (`--kvcache`) writes the key and the transposed value of each token into a persistent cache
at the current position, then attends over the keys written so far. Both run through the graph with
no new Muyo primitive — the key rides in `rope_neox`'s `d_offset` and the value column rides in
`copy.comp`'s `misalign_offsets` (see `docs/Findings.md`).

**M4 complete.** The whole 24-layer Qwen2.5-0.5B runs as one graph: embedding, 24 layers of
rms-norm / QKV+bias / RoPE / grouped-query attention / output projection / residual / gated MLP,
then the final norm, the LM head and sampling. `--forward` feeds the prompt one token at a time,
then generates:

```
$ ./muyo_llm --forward testdata/qwen2.5-0.5b-instruct-fp16.gguf -p "Hello world" -n 8
! I'm trying to create a program
$ ./muyo_llm --forward testdata/qwen2.5-0.5b-instruct-fp16.gguf -p "The capital of France is" \
      -n 64 --temp 0.7 --top-k 50 --top-p 0.9 --repeat-penalty 1.15 --seed 7
<sampled text>
$ scripts/verify-forward.sh            # builds a libllama oracle and diffs the greedy ids
verify-forward: OK (8 greedy tokens match llama.cpp)
```

Flags: `-p/--prompt <text>` or `--tokens id,id`, `-n`, `--chat`, `-i`, `--system`, and the sampler
`--temp/--top-k/--top-p/--repeat-penalty/--repeat-last-n/--seed`. The default is greedy, so a run
without sampler flags is reproducible. The generated text streams to stdout (a UTF-8 character
spanning two byte-piece tokens is held back until it completes, and special tokens are not printed);
diagnostics and the token-id summary go to stderr. The context is capped at `K_MAX_SEQ = 2048`.

`--chat` frames the prompt with Qwen2's ChatML template, and `-i` reads further turns from stdin,
appending each to the KV cache rather than re-prefilling:

```
$ ./muyo_llm --forward testdata/qwen2.5-0.5b-instruct-fp16.gguf -p "What is the capital of France?" \
      --chat -n 48 --temp 0.7 --top-p 0.9 --seed 3
The capital of France is Paris.
$ printf 'What is 2+2?\nWho are you?\n' | ./muyo_llm --forward testdata/qwen2.5-0.5b-instruct-fp16.gguf \
      --chat -i -n 48 --temp 0.7 --seed 5
2+2 is equal to 4.
I am Qwen, a large language model created by Alibaba Cloud. ...
```

**Tokenizer.** Byte-level BPE over llama.cpp's pre-tokenizer: the Unicode helpers are vendored
verbatim from llama.cpp under `src/llama/` (as the shaders are), so the split matches for any script
and special tokens (`<|im_start|>`) are recognized as units. `scripts/verify-forward.sh` checks the
token ids against llama.cpp for ASCII, accented Latin, Japanese, emoji, punctuation and a rendered
ChatML turn, not just the greedy tokens.

Not done: batched prefill (the prompt is still fed one token at a time, so a long prompt is O(P)
full forwards — `mul_mm` and `flash_attn` are the M6 work). `tools/reference_qwen2.py` is a numpy
reference for the same forward, and `MUYO_LOGITS=<file>` / `MUYO_DUMP=<dir>` make the engine write
logits and per-layer intermediates so they can be compared element by element.

## Visualising the graph model

`MUYO_RG_DOT=<path>` with `--forward` writes a **static** Graphviz DOT of the authored model. It is
taken from the node and dependency declarations, before `Build()` — no pipeline is compiled, no
resource is allocated, nothing executes — so it needs nothing but the model's config:

```bash
cd build
MUYO_RG_DOT=/tmp/model.dot ./muyo_llm --forward ../testdata/qwen2.5-0.5b-instruct-fp16.gguf -p hi
dot -Tsvg /tmp/model.dot-overview -o model-overview.svg
dot -Tpng /tmp/model.dot-layer0     -o model-layer0.png
```

Six files are written, three Graphviz DOT and three Mermaid:

| File | What it shows |
| --- | --- |
| `<path>` | DOT: all 436 nodes, one cluster per transformer layer, all 507 ordering edges |
| `<path>-overview` | DOT: each layer collapsed to one node: `embed → layer 0 … layer 23 → final_norm → lm_head → argmax` (27 edges) |
| `<path>-layer0` | DOT: the 18-node layer in detail: `pre_norm → {q,k,v} → attention → o_proj → residual1 → post_norm → {gate,up} → swiglu → down_proj → residual2` (22 edges) |
| `<path>-overview.mmd` | Mermaid: the overview |
| `<path>-layer0.mmd` | Mermaid: layer 0 |
| `<path>.md` | a markdown page embedding both Mermaid blocks, which GitHub renders natively |

[`docs/graph/model.md`](docs/graph/model.md) is the committed page; GitHub shows the two diagrams
inline when the file is opened. `docs/graph/` also keeps the `.dot` files and the raw `.mmd`.

The diagram syntax is Muyo's `RenderGraph::ToDot` / `RenderGraph::ToMermaid` (exposed for the
compiled plan in Muyo #181); this repository only projects the recorded nodes and dependency edges
into the `TraceGraph` topology they take. The projection is per **node**, not per execution segment,
because a compute-only graph compiles to a single segment and the question here is "what is the
model".

This is the static model, not the execution trace. Muyo's trace (#165/#166) records segments,
barriers and timeline values *per execution* and is emitted from `Execute()` via `SetTraceSink` or
`MUYO_RG_TRACE`; the lane view (#168) is the windowed renderer's ImGui page. For a windowless,
long-running consumer the declaration-level DOT is the thing to read.

## Build

Muyo is embedded as a submodule; a development checkout can be substituted with
`-DMUYO_SOURCE_DIR`. The Vulkan SDK (1.4.363 or newer) must be on the environment - **source its
`setup-env.sh`**, not just install it - because the graph is descriptor-heap-only: the headers must
define `VK_EXT_descriptor_heap`, `glslc` must accept `GL_EXT_descriptor_heap`, and running needs the
SDK's loader and validation layers (an older loader rejects the heap structures). Configure also
needs network (VMA is fetched).

```bash
source "$VULKAN_SDK/setup-env.sh"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DMUYO_SOURCE_DIR=/path/to/Muyo
cmake --build build --target muyo_llm -j"$(nproc)"
cd build && ./muyo_llm --smoke        # the graph loads shaders relative to the working directory
```

A device must expose `VK_EXT_descriptor_heap` and `VK_KHR_shader_untyped_pointers`; `HeadlessContext`
requests both and fails at creation with the missing extension named. llvmpipe exposes them and is
what the checks below run on.

`scripts/sanity.sh` runs the format check and the build + smoke; `scripts/sanity.sh sanitize` adds
ASan/UBSan.

## Upstream state

**Validated against Muyo `b7ce1fe`** ("docs: reconcile the roadmaps and board after B2"). Rebuilt and
re-ran `scripts/sanity.sh` (all 12 modes, validation on) with no compile error and no named runtime
failure. `scripts/verify-forward.sh` is **not** green here: the 0.5B model's embedding tensor is
272 MB and this machine's llvmpipe reports `maxStorageBufferRange = 128 MB`, so the descriptor write
is rejected. The same run fails identically on the previous pin (`64947cd`), so it is a device limit,
not the migration; it is recorded as F18.

Since the previous pin `64947cd` (44 commits), the consumer-relevant change is the **descriptor-heap
track** (B0-B2: `82a10a0` … `1896bae`, with the graph made heap-only by `2ef6505`): the graph path no
longer has descriptor sets, descriptor-set layouts, descriptor pools or a pipeline layout. A node's
resources are reached through `layout(descriptor_heap)` arrays indexed by words the node supplies,
and the node records its push block with `vkCmdPushDataEXT` (`ctx.PushData`); every GPU node needs
the injected `DescriptorHeapService`, and the device needs `VK_EXT_descriptor_heap` plus
`VK_KHR_shader_untyped_pointers`. The same window also landed per-execution transient arenas (#40,
`1896bae`), shared pipeline/shader caches (#17, `650381b`/`5b8cc63`/`658dfe3`), the
`shader64BitIndexing`-aware descriptor range work, and a large tooling/warnings pass. This consumer
was migrated in the same pass: every dispatched shader indexes a `heap[]` table in its push block,
every node fills it from `ctx.GetHeapIndex`, and `src/GraphServices.h` injects the heap service. The
findings this produced are F15-F19 in `docs/Findings.md`; the workarounds are the capability
strip in `tools/strip_nonuniform_capability.py`, the explicit
`VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` on every buffer, and a model-sized
`DescriptorHeapService::Config` in `HeadlessContext`.

The previous pin `64947cd` was the ThreadSanitizer scoped-sanity change (`0a9bd55` in master).
The pin before it, `f3ecbf1`, landed **#16's transient aliasing** (`36d8ab2` … `9ef01ac`, with
lifetime enforcement in `27eb53e`) — a graph-owned transient is no longer a resource-manager resource
(`ResolveResource` is the lookup) and its content is valid only from its first write within one
execution, so every resource whose content comes from the host or is read back by the host is now
`Persistent`. The finding, and the evidence that the aliasing works on the real workload, is
`docs/Findings.md` F14. The consumer report for the aliasing row was
[PR #114](https://github.com/v3c70r/Muyo/pull/114) (merged `33e9346`).

The previous pin, `fe40dd3`, added the injected graph services (`0294326`, the injection slice of
#53): this repo switched to the services constructor in the same pass (`src/GraphServices.h`), which
is the switch that flips the `injected, not global` row in Muyo's `docs/Consumers.md` from *gap* to
*holds* ([PR #89](https://github.com/v3c70r/Muyo/pull/89)).

The gaps that blocked M0 and M1 in the plan have closed:

- **Embeddable build** — [Muyo #68](https://github.com/v3c70r/Muyo/issues/68) landed (`0617102`).
  `add_subdirectory(<Muyo>)` plus `target_link_libraries(... muyo_rg)` configures and builds.
- **Specialization constants** — [Muyo #62](https://github.com/v3c70r/Muyo/issues/62), [PR #66](https://github.com/v3c70r/Muyo/pull/66).
- **Cooperative-matrix enablement** — [Muyo #63](https://github.com/v3c70r/Muyo/issues/63), PR #71.
  On this machine it is available through **AMDVLK** (11 property sets, `F16/F16->F32` 16x16x16), not
  through the default RADV driver; AMDVLK is opt-in via `VK_DRIVER_FILES`.
- **Static resource byte range** — [Muyo #81](https://github.com/v3c70r/Muyo/issues/81) landed as
  [PR #82](https://github.com/v3c70r/Muyo/pull/82) (`a8c4b55`): a resource use may declare a byte
  `offset`/`size`. This is the weight-slicing primitive; the KV append turned out not to need it.
- **Injected graph services** — [Muyo #53](https://github.com/v3c70r/Muyo/issues/53)'s injection
  slice landed as `0294326`; see above. Its `docs/Consumers.md` `injected, not global` row is flipped
to *holds* for this consumer in [PR #89](https://github.com/v3c70r/Muyo/pull/89)
([issue #88](https://github.com/v3c70r/Muyo/issues/88)).

Open, and not blocking M0/M4: reading the cooperative-matrix property sets can crash under one
driver + validation-layer pairing ([Muyo #72](https://github.com/v3c70r/Muyo/issues/72)), and the
rest of the library boundary ([Muyo #53](https://github.com/v3c70r/Muyo/issues/53): the stated public
surface, a shader-source provider, and declaring errors that name the entity).
