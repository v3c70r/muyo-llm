# muyo-llm — implementation plan

A lightweight LLM inference engine built on Muyo's RenderGraph. Its purpose is not
throughput: it is the **second, non-renderer consumer** of `muyo_rg`, and therefore the
boundary test the Muyo roadmap asks for (`docs/Roadmap.md`, Risk #4; `docs/RenderGraph-Library-Design.md`).

Status: draft for review. No code yet.

---

## 1. Goal and non-goals

**Goal.** Run a small decoder-only LLM on the GPU through `muyo_rg`, decode-only, and use
that workload to find where the RenderGraph is not yet ready to be a library. Deliver two
artifacts:

1. a working engine, held to the same code, formatting, sanitizer and documentation standards
   as Muyo;
2. an assessment of RenderGraph maturity — a list of findings, each with a reproduction
   command or other evidence, filed upstream as issues or pull requests.

**Non-goals (v1).** Performance and throughput targets; an HTTP/server surface; multi-model or
multi-session serving; training; prefill/batching (deferred to v2); quantization other than F16
(deferred, see §8).

## 2. Scope

- **Phase:** decode only — one token per graph execution, repeated.
- **Precision:** F16 weights, F32 accumulation.
- **Model:** Qwen2.5-0.5B (GQA: asymmetric KV head count, which exercises the KV cache).
- **Shaders:** llama.cpp's Vulkan compute shaders (`ggml-vulkan/vulkan-shaders`), reused as
  source and compiled by us. `ggml-vulkan` itself is not linked.
- **Platform:** Vulkan 1.3+, cooperative matrix enabled (see §5.2).

## 3. Why this workload

Muyo's roadmap names ML/LLM inference as the intended second consumer and states that "a
small compute-only example is worth more than any amount of interface design" (Risk #4). The
library constraints it lists are exactly what decode stresses:

| Constraint (Muyo roadmap) | How decode exercises it |
| --- | --- |
| No frame boundary in the library | Each execution is one token; weights and KV cache outlive it. |
| Compute-only is first-class | No window, no swapchain, no colour attachment anywhere. |
| Long-lived resources are first-class | Weights (process lifetime) and KV cache (generation lifetime). |
| Device/allocator/descriptor injected, not global | Currently impossible without dragging in the engine's 15 singletons. |
| Diagnostics an agent can act on | Failures surface as wrong numbers, not validation errors. |

Decode is a deep, narrow graph (~7 nodes x N layers per token, tiny M) with a growing KV
cache — the case the current `RenderGraphBuilder` (single in-flight slot, transient
single-instance resources) is least proven on. Prefill is a wide graph much closer to a
render frame and can follow.

## 4. Decisions

| Area | Decision | Rationale |
| --- | --- | --- |
| Repository | `muyo-llm`, separate from Muyo | Keeps consumer and library separate; same owner (`v3c70r`). |
| Consuming Muyo | git submodule + `add_subdirectory`, link `muyo` + `muyo_rg` | `muyo_rg` cannot yet link standalone; the coupling is a finding, not a prerequisite to fix. |
| Shaders | Vendor llama.cpp `vulkan-shaders` sources + our own `glslc` rules | No `ggml-vulkan` dependency; only sources and contracts. |
| Shader contract layer | Per-op C++ wrapper replicating defines, spec constants, push-constant struct and bindings | Muyo has no such layer; this is the bulk of the work. |
| Push constants | Define the struct in muyo-llm; `vkCmdPushConstants` inside the node callback using `ctx.pipelineLayout` | Works today (see §5.1); no upstream change needed. |
| Spec constants | Upstream support required; tracked as Muyo #62 | Blocking (§5.2). |
| Graph mapping | One compute node per op; explicit `AddDependency` chains | Matches the current API (no implicit edges). |
| Weights / KV | Imported / `Persistent` buffers, reused across executions | Directly tests the long-lived-resource constraint. |
| Correctness oracle | Primary: llama.cpp **Vulkan** backend, bit-exact. Secondary: ggml **CPU**, tolerance. | See §7. |
| Documentation language | English | Requested; matches Muyo. |

## 5. Upstream gaps

### 5.1 Push constants — no functional gap

The pipeline layout gets its push-constant ranges from shader reflection
(`src/RenderGraph/RenderGraphBuilder.cpp:179-190`), and each node's callback receives
`context.pipelineLayout` (`RenderGraphBuilder.cpp:915`). A consumer can therefore define the
struct locally and push it:

```cpp
vkCmdPushConstants(ctx.commandBuffer, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                   0, sizeof(MulMatParams), &params);
```

Ergonomic gaps only, candidates for the `#55` (fail at Build) workstream:

- `ResourceUsage::PUSH_CONSTANTS` is a dead enum — `ResourceUseResolver.h:157` throws on it.
- No build-time check that a pushed range matches reflection.

### 5.2 Specialization constants — blocking, filed as Muyo #62

The graph reflects `specializationConstants` (`ShaderReflection.h:72-82`,
`ShaderReflectionFetcher.cpp:145-162`) but never passes `VkSpecializationInfo` when creating a
pipeline (`RenderGraphBuilder.cpp:225-237`). llama.cpp shaders require it:

- `mul_mm.comp` emits **74 `OpSpecConstant`s** in the `DATA_A_F32` x `B_TYPE=mat2x4` variant
  (verified locally with `glslc` 2023.8 + `spirv-dis | grep -c OpSpecConstant`);
- `mul_mat_vec.comp` uses `layout(local_size_x_id = 0, ...)`, so the workgroup size itself is a
  specialization constant.

Upstream issue: <https://github.com/v3c70r/Muyo/issues/62>.

### 5.3 Cooperative matrix — upstream enablement required, on M1's critical path

Decision: target Vulkan 1.3+ and use cooperative matrix where available; do not carry a
separate non-coopmat path. The local device supports it:

```
VK_KHR_cooperative_matrix   : extension revision 2
VK_NV_cooperative_matrix2   : extension revision 1
```

But Muyo never queries or enables it (`grep -rni cooperative_matrix src/ docs/ CMakeLists.txt`
→ no matches) and its device creation only enables `synchronization2` and `timelineSemaphore`
(`VkRenderDevice.cpp:227-315`). A consumer that needs cooperative matrix cannot ask for it.

**Prior art exists in the repository.** Branch `origin/coopMat`, commit `5bf2840` ("Add GEMM pass
with cooperative matrix extension"), experiments with this in the legacy `RenderPassManager`
path:

- `src/helloVulkan.cpp` adds `VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME` to the required device
extensions and puts `VkPhysicalDeviceCooperativeMatrixFeaturesNV` on the device feature chain;
- `src/RenderDevice/RenderPassGEMM.{h,cpp}` and `shaders/shmem.comp` run a cooperative-matrix
GEMM and query `VkCooperativeMatrixPropertiesNV`;
- `ComputePipelineBuilder::AddShaderModule` gains a `const VkSpecializationInfo*` parameter —
the same wiring #62 asks for on the graph path.

The graph path needs the equivalent, targeting `VK_KHR_cooperative_matrix` (the portable
extension llama.cpp's `COOPMAT` shaders use, rather than the NV one the branch targeted). This is
on muyo-llm's M1 critical path; Muyo's roadmap planning will place it.

### 5.4 Shader source injection

`ShaderAssetManager::LoadShader(name)` reads `shaders/<name>.spv` relative to the CWD
(`ShaderAsset.h:84`) and accepts only a name. A library client cannot supply SPIR-V bytes, a
search path, or an in-memory module. Candidate for `#53`.

### 5.5 Known constraints, to be measured rather than assumed

- 15 singleton call sites inside `src/RenderGraph/` (`GetRenderDevice`,
  `GetRenderResourceManager`, `GetDescriptorManager`).
- `RenderGraphNodeContext` exposes `meshManager`, which is meaningless to a non-renderer.
- No implicit data-flow edges: sharing a resource does not order nodes (`docs/RenderGraph.md`).
- `RenderGraphBuilder` owns a single in-flight slot (`inFlightCount == 1`), so a second
  `Execute()` blocks; cross-token overlap needs the executor API directly and touches the
  transient single-instance limitation.

## 6. Milestones

Each milestone lists its deliverable and its acceptance check.

### M0 — Skeleton, host, and smoke graph

- CMake project, Muyo submodule, `.clang-format` / `.clang-tidy` / `san.supp` copied from
  Muyo, a `scripts/sanity.sh` analogue, English `docs/`.
- Headless host modelled on `src/tests/GraphicsTestEnv.h`: device, memory allocator, resource
  manager, descriptor manager. No window.
- One compute node through `RenderGraphBuilder` + `Execute()`.

**Acceptance:** `./build/muyo_llm --smoke` exits 0 and prints the compiled node/resource list.
**Side artifact:** the list of engine pieces that had to be pulled in to run headless (first
batch of findings).

### M1 — llama.cpp shader pipeline and the first real op

- Vendor the shader sources; build `add.comp` (elementwise) to SPIR-V with an explicit define set.
- Wire specialization constants into Muyo (implements #62) on a `rg-*` branch and PR it.
- Push-constant wrapper for one op.

**Acceptance:** `C = A + B` matches a CPU reference elementwise; reverting the spec-constant
wiring makes the test fail (teeth). Confirm `mul_mm.comp` with its 74 constants also compiles
and dispatches.

### M2 — Model loading and embedding

- GGUF parser (F16/F32 only), tensor manifest, weight upload.
- Tokenizer (BPE; reuse llama.cpp's vocab parsing or a minimal port).
- `get_rows.comp` for embedding.

**Acceptance:** target model loads; embedding for a fixed token matches the ggml reference.

### M3 — One transformer layer

- `rms_norm`, QKV `mul_mat_vec`, `rope_*`, KV write, attention, `swiglu`, output projection,
  residual `add`.
- KV cache as a `Persistent` buffer; position via push constant.

**Acceptance:** single-layer hidden state matches the reference within tolerance, with the
comparison command recorded.

### M4 — Full model and sampling

- N layers, final norm, LM head, `argmax` greedy sampling.
- End-to-end generation.

**Acceptance:** for one prompt and model, the first K greedy tokens match `llama-cli`, and
logits match the CPU reference within tolerance.

### M5 — Maturity assessment and feedback

- Findings written in Muyo's format (`C/A/N/P`, confidence, acceptance criterion).
- A cross-token pipeline experiment: overlap token N+1 with token N; record the concrete limits.
- Produce `docs/Assessment.md` and file/PR the upstream fixes.

**Acceptance:** every finding has a copy-pasteable reproduction command or other evidence.

### M6+ (deferred)

Prefill with `mul_mm` (M = P) and `flash_attn`; quantization (Q4_0, Q8_0, K-quants).

## 7. Verification strategy

Three layers, because bit-exactness against a CPU is not a fair target.

Floating-point results differ between any two accumulation orders: GPU GEMMs reduce over tiles
in parallel, CPUs sum sequentially; FMA and fast-math differ; special functions (exp, erf,
rsqrt) are approximated differently. llama.cpp is itself not bit-exact across backends. So:

| Layer | Oracle | Standard | Purpose |
| --- | --- | --- | --- |
| Single op | llama.cpp **Vulkan**, same variant and spec constants | **bit-exact** | Isolates graph-level faults (barriers, ordering, lifetime). Same math, so any difference is ours. |
| Whole-model logits | ggml **CPU** | tolerance (e.g. max abs diff < 1e-2) | Coarse correctness; catches systematic errors. |
| End-to-end | `llama-cli` greedy | first N token ids match; record any divergence point | User-visible behaviour. |

The key property: Muyo's barriers and ordering do not change values. When the oracle is the
same shader with the same parameters, a numeric difference points at our graph wiring, not at
floating-point luck.

## 8. Quantization (deferred) — what it costs

F16 is the easy case: weights are f16 arrays, matmul reads them directly. Adding quantization
means:

1. **Shader variants.** Each quant type has a `dequant_*.comp` and matmul variants selected by
   `-D` (`mul_mat_vecq.comp` for decode, `mul_mmq.comp` for prefill). We compile only the
   variants the target model needs.
2. **Block layouts.** Weights are block-quantized (`block_q4_0`, K-quants with 256-element
   blocks, ...); C++ offsets and strides must match `types.glsl`.
3. **In-shader dequantization**, accumulation still F32, so the precision loss is the
   quantization's, not ours.
4. **Two kernels per phase.** Decode and prefill use different matmul shaders.

Design mitigation: make weights a `TensorDesc` carrying a dtype (mapping to a ggml type id)
from the first commit, so adding a quant type is a new node factory plus variants, not a
refactor.

## 9. Repository and engineering standards

Mirrors Muyo so a change is reviewable in the same terms:

- `.clang-format` (clang-format 18.1.3), `.clang-tidy`, `san.supp` copied from Muyo.
- `scripts/sanity.sh` with `docs` / `format` / `tidy` / `sanitize` subcommands; formatting
  checked on changed lines only (`clang-format-diff`), never the world.
- Sanitizer build (`CMAKE_BUILD_TYPE=Sanitize`, ASan + UBSan) run for anything touching
  memory, lifetimes or resource ownership.
- Doxygen coverage for public entities, with the same "undocumented public entity fails the
  build" rule (`docs/Doxyfile`, a coverage checker).
- A working agreement (`AGENTS.md`-style) because work spans sessions; and, per the consumer
  discussion, a `docs/Consumers.md` in **Muyo** listing consumers and the guarantees they rely
  on, so "library boundary" is a checkable list rather than a slogan.

**Upstream contribution:** consumer-driven Muyo changes follow Muyo's process — file an issue
first if the roadmap does not already cover the gap, then a `rg-*` branch, squash PR, a test
with teeth, and the standard verification table. `AGENTS.md` updates may go straight to trunk.

## 10. Risks

1. **Shader contract reconstruction.** Each op needs its define set, spec constants,
   push-constant layout and bindings read out of `vulkan-shaders-gen.cpp` and
   `ggml-vulkan.cpp`. Not hard, but large and silent: a wrong field yields a wrong number, not
   an error. Mitigation: one op at a time, each verified bit-exact against llama.cpp Vulkan.
2. **Cooperative matrix enablement.** Muyo does not enable the extension (§5.3). If we commit
   to coopmat-only, that upstream change is on the critical path for M1.
3. **glslc version.** Locally 2023.8; llama.cpp's newer shaders may need a newer toolchain for
   some variants. Mitigation: pin the variants we use and record the toolchain.
4. **Scope creep into performance work.** Out of scope; the deliverable is the maturity
   assessment.

## 11. Open items

- File the cooperative-matrix enablement gap (see §5.3); on this project's M1 critical path.
- File (or fold into #53) shader-source injection.
- `docs/Assessment.md` lives in this repository (decided).
- muyo-llm is registered in Muyo as a first-party architecture-validation consumer; Muyo's
  `docs/Consumers.md` states the guarantees this project relies on.

## 12. References

- Muyo roadmap: `~/Projects/Muyo/docs/Roadmap.md` (Risk #4, constraints table).
- Muyo library design: `~/Projects/Muyo/docs/RenderGraph-Library-Design.md`.
- RenderGraph concept: `~/Projects/Muyo/docs/RenderGraph.md`.
- Muyo working agreement: `~/Projects/Muyo/AGENTS.md`.
- Specialization constants gap: <https://github.com/v3c70r/Muyo/issues/62>.
- llama.cpp shaders: `~/Projects/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/`.
