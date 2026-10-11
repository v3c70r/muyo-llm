# Findings

What building a second, non-renderer consumer of the RenderGraph has actually turned up. This is
the working log; `Assessment.md` will be the summary written once M5 is reached. Each entry states
what was observed, the command or file that shows it, how sure we are, and what it would take
upstream to close it.

Format follows Muyo's review vocabulary (`C` correctness, `A` architecture, `N` new issue, `P`
polish; confidence `confirmed` / `needs runtime validation` / `design preference`).

## Positive: what held

These are as important as the gaps, because they are the guarantees `docs/Consumers.md` claims.

- **Embedding works.** After Muyo #68 (`0617102`), `add_subdirectory(<Muyo>)` plus
  `target_link_libraries(muyo_llm PRIVATE muyo_rg)` configures and builds from outside the tree.
  Evidence: this repository's first configure.
- **Compute-only, windowless execution works end to end.** Device, allocator, resource manager,
  descriptor manager, then a graph-owned storage buffer written by a compute node and read back on
  the host. No surface, no swapchain, no frame. Evidence: `muyo_llm --smoke` → `[smoke] OK`.
- **Specialization constants (#62) and cooperative-matrix enablement (#63) hold.** The device
  reports `cooperative matrix: available` on the development machine (AMDVLK). Evidence:
  `muyo_llm --smoke`'s `[device]` lines.

## F1 — an embedder must link `SPIRV-Reflect` by hand

**A, resolved** — fixed by Muyo `6f8d65b`, which declares `SPIRV-Reflect` a `PUBLIC` dependency of
`muyo`. The consumer's hand-written link line is removed and the build links without it. The finding
is kept for the record. Original text:

`muyo` compiles `src/ShaderReflectionFetcher.cpp`, which calls `spvReflect*`, but
neither `muyo` nor `muyo_rg` declares a dependency on the `SPIRV-Reflect` target; only the in-tree
`tests` target links it (`Muyo/CMakeLists.txt:391`). A consumer that links `muyo_rg` — which needs
shader reflection at `Build()` — fails to link:

```
undefined reference to `spvReflectCreateShaderModule'
undefined reference to `spvReflectEnumerateDescriptorSets'
...
```

Evidence: this repository's first `cmake --build build --target muyo_llm`. Worked around by naming
`SPIRV-Reflect` on our own link line. Upstream: make it a `PUBLIC` dependency of `muyo` (it owns the
implementation), so an embedder inherits it. Same class as the `muyo_rg`/`muyo` declaration #68 fixed.

## F2 — a compute-only graph still creates the graphics semantic set layouts

**A, confirmed; subsumed by #15, not filed.** Constructing a `RenderGraphBuilder` constructs
`RenderGraphDescriptorSets`, whose constructor unconditionally creates the three built-in semantic
set **layouts** (`PER_VIEW`, `PER_OBJ`, `MATERIAL`) — `RenderGraphDescriptorSets.h`. A graph whose
every node is compute with explicit `DescriptorBinding`s never references any of them. The cost is
three `vkCreateDescriptorSetLayout` calls, **not** the 1024-descriptor pool cliff: descriptor-pool
accounting is charged when a *set* is allocated per graphics node, which never happens here. Evidence:
our smoke adds one explicit-binding compute node and never uses a semantic set.

This was overstated when first written — it is not the MATERIAL cliff. Muyo #15 and
`docs/DescriptorSet-Lifecycle-Design.md` own that cliff and the redesign ("derive sets instead of
declaring them"), under which a compute-only graph would not declare semantic layouts at all, so
this falls out of #15 rather than needing its own issue. Re-recorded with the correct scope so the
conclusion not to file is on the record.

## F3 — a GPU storage-buffer writer must declare `READ_WRITE`, because `WRITE` means *host* write

**C, confirmed; already tracked by Muyo #12, not filed.** `ResourceUseResolver.h`'s
`K_USAGE_POLICIES` maps `{STORAGE_BUFFER, WRITE}` to
`{VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_WRITE_BIT}`, while the enum documents `WRITE` as
"Write-only access." A compute shader that writes a storage buffer must therefore spell `READ_WRITE`
to get `ALL_COMMANDS | SHADER_READ | SHADER_WRITE`; the natural spelling yields a host profile.

This is not a discovery here — Muyo already knows, and `RecordBarriers` carries the note verbatim:

> the resolver's `{STORAGE_BUFFER, WRITE}` profile is HOST_WRITE, so the first such node emitted a
> barrier this rejected. The *semantic* fix, giving a GPU writer a SHADER_WRITE destination instead
> of a host profile, is #12's subject and deliberately not here.

`RecordBarriers` currently works around it with `ALL_COMMANDS | HOST` on both sides. Evidence: the
policy table, that comment (`RenderGraphBuilder.cpp`, master), and our smoke node, which uses
`READ_WRITE`. Filed as input to [#12](https://github.com/v3c70r/Muyo/issues/12) rather than as a new
issue.

## F4 — the graph does not barrier out of the last node to the host

**P, confirmed.** The graph plans the barriers *into* each node. The compute-write → host-read pair
after the final dispatch is the caller's to record; `RenderGraph.md`'s "what the graph handles for
you" lists barriers without saying that edge is excluded. Evidence: the smoke's node callback
records an explicit `VkMemoryBarrier2` (`COMPUTE_SHADER` → `HOST`). Not a defect, but a documented
boundary would save the next consumer the same discovery.

## F5 — every node context carries a `MeshResourceManager&`

**A, confirmed.** `RenderGraphNodeContext` holds `resourceManager` and `meshManager` references
(`RenderGraphNodeContext.h`), and `Execute()` dereferences the mesh singleton for every node
including compute ones (`RenderGraphBuilder.cpp`: `context.meshManager = *GetMeshResourceManager()`).
A non-renderer consumer inherits a renderer-shaped context and a renderer singleton it has no use
for. Upstream: the injected-services work (#53).

## F6 — the whole engine is linked to run headless
**A, confirmed.** `muyo_rg` links `muyo` (#68's deliberate encoding of the current coupling), so a
windowless compute consumer builds and links ImGui, ImGuizmo, imnodes, GLFW, tinygltf, tinyobj, stb,
meshoptimizer, and the RenderPasses / Scene / Debug layers it never calls. Evidence: this
repository's configure and build output. This is the cost #53 and #58 exist to remove; recorded here
as the size of it, since "the whole engine" is easy to say and hard to size.

## F7 — LeakSanitizer cannot be a clean signal on this driver

**P, confirmed.** Under `LSAN_OPTIONS` with the suppressions in `san.supp`, the sanitized smoke is
clean. Without them, LeakSanitizer reports leaks with no frame in this repository: 40 bytes inside
`amdvlk64.so`, and 72,704 bytes from the Vulkan loader's `dlopen` of the ICD inside a library
constructor (`<unknown module>` ← `call_init` ← `_dl_init`). Both outlive `main()` by design rather
than by defect. Recorded because a leak report that is 100% third-party is easy to mistake for a
bug in the consumer, and because AMDVLK is the only driver here that exposes cooperative matrix, so
`docs/Findings.md`'s F-row for the driver cannot be avoided by switching drivers. `san.supp` carries
the module- and constructor-scoped suppressions, and the rationale, rather than a blanket
`detect_leaks=0`.

## F8 — a consumer cannot enable a Vulkan 1.1 feature, so F16 weights cannot run

**C, resolved** — fixed by Muyo `643ce26` (#78), which sets `storageBuffer16BitAccess` and
`uniformAndStorageBuffer16BitAccess` on `features11` and verifies both. The GEMV now runs the F16
variant and matches the CPU reference; the F32 workaround is gone. Kept for the record. Original
text:

Enabling an F16
storage buffer needs `storageBuffer16BitAccess`. That feature is Vulkan 1.1, so it lives in
`VkPhysicalDeviceVulkan11Features` — which `VkRenderDevice::CreateDevice` builds and chains
itself (`VkRenderDevice.cpp`, the `features11` block: `multiview`, `shaderDrawParameters`). A
consumer therefore cannot supply it either way: appending a `VkPhysicalDevice16BitStorageFeatures`
is rejected because the promoted feature is already present, and appending a second
`VkPhysicalDeviceVulkan11Features` is a duplicate structure. Validation, on the first F16 GEMV:

```
vkCreateDevice(): pNext chain includes a VkPhysicalDeviceVulkan11Features structure, then it
must not include a VkPhysicalDevice16BitStorageFeatures structure ... (VUID-VkDeviceCreateInfo-pNext-02829)
```

Impact: **every llama.cpp matmul kernel that reads F16 weights fails to create its shader module**
(`vkCreateShaderModule(): SPIR-V contains a 16-bit OpVariable with StorageBuffer StorageClass, but
storageBuffer16BitAccess was not enabled`). That is the whole F16 inference path — the precision the
plan chose. The renderer never hits it because its own shaders do not use 16-bit storage.

Workaround in place: the GEMV runs with F32 weights (`mul_mat_vec_f32_f32_f32`), which needs no
optional feature. The F16 variant is vendored and compiles; it is one device feature away.

**This is an enablement gap, not a support gap.** `storageBuffer16BitAccess` was promoted to core in
Vulkan 1.1, but a core *optional feature* is still off by default - it has to be set to `VK_TRUE` in
`VkDeviceCreateInfo`'s chain. Querying (`vkGetPhysicalDeviceFeatures2`) only answers whether it is
supported. The device does support it:

```
VkPhysicalDeviceVulkan11Features:
    storageBuffer16BitAccess           = true
    uniformAndStorageBuffer16BitAccess = true
```

So nothing about the API version blocks it; `CreateDevice`'s own `features11` struct does.
Filed upstream as Muyo #78 (the narrow unblock), and linked from #53/#58 as evidence for the
general shape: feature policy belongs to the host.

Upstream: this is the "feature policy moves to the host" half of the device boundary
(`RenderGraph-Library-Design.md`, "What the device layer must shed"; #53, #58). The minimal fix is
to let `CreateDevice` accept the Vulkan 1.1 feature set (or the whole promoted-feature chain) from
the caller rather than constructing `features11` itself. Recorded here rather than filed because
#53/#58 already own the boundary; it is concrete evidence for both.

## F9 — the KV-cache append has no expression in the graph

**A, confirmed; workaround in place.** A decoder writes the current token's K and V into a view of
the cache at a dynamic position (row `position`). With the shaders vendored so far, neither half of
that is directly expressible:

- `copy.comp` (`cpy`) computes its destination index from the destination tensor's shape and
  strides, plus a **sub-buffer offset** for the row (ggml passes `vk_tensor_offset(t) + t->view_offs`
  as the descriptor offset). The graph binds whole buffers — a `ResourceUse` names a resource, not a
  byte range — so `cache[position]` has nowhere to put the position. The shader's own
  `misalign_offsets` field is 8 bits per operand (up to 255 bytes) and is for alignment, not a row
  index.
- The GEMV can write a **row** offset: setting `base_work_group_y = position` and
  `batch_stride_d = rowStride` makes `d_offset = position * rowStride`, and `stride_d` still bounds
  the rows (found while writing the single layer). That covers a row-major K cache, and a row-major
  V cache.
- But the `V^T . p` product needs V **transposed** to be a plain `mul_mat_vec`, and a transposing
  append is a scatter — no shader writes one element per column at stride `sequence`. `mul_mat_vec_nc`
  does not help: its column stride is 1.

So the single layer (`src/TransformerLayer.cpp`) reads a **host-provided transposed V cache** and
omits the K/V projections; the append is M3b. The two ways out: give the graph a per-resource offset
or view (which is also what an imported slice of a shared buffer needs, Muyo-side), or add a
transposing append op. Recorded because "where does the KV cache row go" is the first thing a
decode engine needs and the library has no answer for it.

Related: Muyo's `docs/Consumers.md` lists "long-lived imported resources survive across executions"
as a hold. This is the next level of the same concern — not surviving, but being *written in place*
at a chosen offset.

## F10 — a graph with no nodes builds and runs silently

**P, confirmed.** `RenderGraphBuilder::Build()` on a builder that never had `AddNode` called succeeds,
and `Execute()` completes and does nothing. A consumer that forgot its node — which this session did
while writing the embedding path — sees an all-zero result and no error; the only signal is an empty
`GetExecutionOrder()`. Related to #55 ("fail at Build, and name the graph entity"): a render graph
with nothing to render is almost certainly a mistake, and "no nodes" is a declaration state the
graph can check for free. Either reject it at `Build()` or warn once.

The cost here was a debugging round: zeros read as a sync or binding problem (the first fix was a
missing host barrier, which was also wrong to guess at) before the empty execution order was printed.

### F9 follow-up: which mechanism, decided

Checked against llama.cpp rather than guessed, because muyo-llm's model is llama.cpp's:

- llama.cpp binds every resource as a **static** `vk_subbuffer{buffer, offset, size}` →
  `VkDescriptorBufferInfo` (`ggml-vulkan-types.h:1081`), writes descriptors **at record time** per
  dispatch, and uses **no** dynamic offsets. So the primitive is a static byte offset, and the part
  that matters is that it is resolved at **Execute**, not at Build.
- Dynamic offsets are therefore not the fix, and `muyo-llm` should not ask for them. Only the
  transposed-V column append needs the offset at all; K and row-major V append already go through
  the GEMV's `base_work_group_y`.
- GPU-driven would need **no** new API: bind the whole buffer and read `offset`/`range` from a
  device buffer. Recorded so the graph does not grow a dynamic-offset mechanism for a case that
  would not use it.

Written into Muyo #81 as a comment rather than a second issue.
