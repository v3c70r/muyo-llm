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

**Validated against Muyo `e71f610`** ("helloVulkanRG: recreate the swapchain when the window size
changes"). Rebuilt from clean and re-ran `scripts/sanity.sh` (all 12 modes, validation on) and
`scripts/verify-forward.sh` (greedy tokens + token ids vs llama.cpp) with no compile error, no
deprecation warning and no named runtime failure.

Since the previously recorded pin `fe40dd3`, upstream landed #16's transient aliasing (`36d8ab2` …
`9ef01ac`, with lifetime enforcement in `27eb53e`), the descriptor work (#107/#108/#109), `#55`'s
`Build()` diagnostics (`347eb11`), the lavapipe CI gate (#22), and the renderer's present path
(#110/#111/#112). The one that reached this consumer is **#16**: a graph-owned transient is no
longer a resource-manager resource (`ResolveResource` is the lookup) and its content is valid only
from its first write within one execution, so every resource whose content comes from the host or is
read back by the host is now `Persistent`. The finding, and the evidence that the aliasing works on
the real workload, is `docs/Findings.md` F14.

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
