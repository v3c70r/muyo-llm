# Findings

What building a second, non-renderer consumer of the RenderGraph has actually turned up. This is
the working log; `Assessment.md` will be the summary written once M5 is reached. Each entry states
what was observed, the command or file that shows it, how sure we are, and what it would take
upstream to close it.

Format follows Muyo's review vocabulary (`C` correctness, `A` architecture, `N` new issue, `P`
polish; confidence `confirmed` / `needs runtime validation` / `design preference`).

## Positive: what held

These are as important as the gaps, because they are the guarantees `docs/Consumers.md` claims.

- **The graph is given its services, not globals (#53).** Muyo `0294326` replaced the 15 singleton
  lookups inside `src/RenderGraph/` with a `RenderGraphServices` constructor (today
  `{device, resourceManager, descriptorHeap}`, plus optional caches). Every `RenderGraphBuilder` in this repo is now built with it
  (`src/GraphServices.h`) rather than the transitional `(VkRenderDevice*)` shim, so the graph no
  longer reaches for the engine's globals on the consumer's path. This is the switch Muyo's #53
  commit says flips the `injected, not global` row in `docs/Consumers.md` from *gap* to *holds*;
  that row is flipped in Muyo [PR #89](https://github.com/v3c70r/Muyo/pull/89)
  ([issue #88](https://github.com/v3c70r/Muyo/issues/88)).
- **Embedding works.** After Muyo #68 (`0617102`), `add_subdirectory(<Muyo>)` plus
  `target_link_libraries(muyo_llm PRIVATE muyo_rg)` configures and builds from outside the tree.
  Evidence: this repository's first configure.
- **Compute-only, windowless execution works end to end.** Device, allocator, resource manager,
  descriptor heap, then a graph-owned storage buffer written by a compute node and read back on
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

**A, confirmed; subsumed by #15, not filed. Historical:** B1b deleted the descriptor-set path
(`2ef6505`), so the layouts this entry is about no longer exist. Kept as the record of what the
graph cost a compute-only consumer before the heap.

Constructing a `RenderGraphBuilder` constructs
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

**A, resolved** — fixed by Muyo `0294326` (the injection slice of #53), which deletes the
`MeshResourceManager&` from `RenderGraphNodeContext` and the two includes with it. Nothing read it
for a compute node, so the fix is a removal rather than an injection, and it is compiler-enforced.
The finding is kept for the record. Original text:

`RenderGraphNodeContext` holds `resourceManager` and `meshManager` references
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

## F11 — `mul_mat_vec` hardcodes the A row stride to `ncols`

**C, llama.cpp-side, worked around.** The vendored `mul_mat_vec.comp` computes the A row base as
`ibi = first_row * p.ncols` — it never reads `p.stride_a` for the *row* stride (it is used elsewhere).
A is therefore assumed row-contiguous with row stride equal to the reduction length. The
`--matvec` case never exposed it because `strideA` and `ncols` were the same value.

It surfaced in M3b: the value cache is stored transposed (`V_T[channel][position]`, row stride
`max_seq`) so `V_T · probs` is an ordinary GEMV, but the reduction must span the keys written so
far (`ncols = position + 1`) — which is not the row stride. The fix is to run the reduction over
the **whole** cache row (`ncols = max_seq`) and zero the unwritten columns once at allocation.
Columns past the current position are zero in `V_T`, so they contribute nothing no matter what the
probabilities are. This is a llama.cpp contract, not a Muyo one; it is recorded because a consumer
that reshapes a matrix into the shader will hit it, and because it constrains the cache layout.

## F12 — reproducing `init_pushconst_fastdiv` for `copy.comp`

**C, consumer-side, fixed.** `generic_unary_head.glsl` decomposes the linear index for
`src0_idx`/`dst_idx` with `fastdiv(idx, mp, L)`. The multipliers are built by
`init_pushconst_fastdiv(vk_op_unary_push_constants&)`, which divides by the products **including the
innermost dimension**: `ne02*ne01*ne00`, `ne01*ne00`, `ne00` (and `ne12*ne11*ne10`, …). The first
M3b attempt used the products *excluding* the innermost dimension (`ne03*ne02*ne01`, …); for a 1-D
source whose innermost dimension is the only non-1 one, that makes the divisor `1`, and
`fastdiv(idx, 1, 0) = (mulhi(idx,1) + idx) >> 0 = idx`, so the outer indices take the value `idx` and
every index past the first underflows. With the divisors copied correctly from llama.cpp there is no
problem for any non-empty tensor.

This is recorded as a correction rather than a llama.cpp defect: the contract is "the magic values
are the cumulative products from the innermost dimension outward", and getting it wrong produces a
first-element-only write, which looks like a binding or barrier bug rather than an arithmetic one.
The transposing value append works with the correct multipliers and no descriptor offset — the
destination base rides in `misalign_offsets`' low 16 bits (`get_doffset()`), enough for a cache
position up to 65535. A second trap sits next to the first: `pack_fastdiv_L(l012, l01, l0)` packs
the **outermost** divisor's shift in the low byte, so building the word as `l0 | l1<<8 | l2<<16`
reverses the shifts while the multipliers stay correct, and the destination index is silently wrong
past the first few elements. The M4 forward hit exactly that (the transposed value cache came out
half-written), which is why both halves of the contract are written down here.

## F13 — the graph takes the whole model, but bindings are fixed at Build

**A, positive with a constraint.** The full Qwen2.5-0.5B forward is **one** graph of 436 nodes over
roughly 700 buffer resources (291 weights, 48 KV-cache buffers, the rest per-layer transients).
`Build()` is called once and `Execute()` is called per token; the only thing that changes between
tokens is the KV state and three push constants (the position, the RoPE destination offset and the
score dispatch). It builds and runs without a graph-side limit being hit, which is the answer to the
scalability half of the boundary test.

The constraint is that a resource use is bound at Build, so a 24-layer model needs 24 sets of weight
resources — one node per layer. A decode engine cannot build **one** layer graph and re-point it at
layer `l`'s weights per execution; the alternatives are 24 nodes (what this does), rebuilding the
graph per layer, or putting every layer's weights in one arena and binding a static slice per node
— which is exactly what Muyo #81/PR #82 now enables. So the static range is not only for imported
slices; it is also how a single-arena weight layout is bound. The deferred "base known only per
execution" half would be the thing that lets one graph node rotate through layers, and nothing here
needs it.

Under B1b the "binding" is a heap index in push data rather than a descriptor-set slot, but it is
still fixed at `Build()` for a graph-owned or imported resource: the index is a static descriptor,
and only a ring-allocated imported resource can be re-pointed per execution (#83). The 436-node
build still holds; F18 is the device-range limit that stops the *forward pass* from completing on
llvmpipe, not a graph limit.

Related: Muyo inserts no barrier between the last node and a host `Map()`, so the `argmax` node ends
with its own compute-to-host `VkMemoryBarrier2` (Finding F4) — the only place this model has to
reach past the graph API.

## F14 — #16's aliasing makes host-touched resources require `Persistent`

**A, breaking, adapted.** Muyo #16's aliasing half (`36d8ab2`, `d87c2b2`, `b60220f`, `9ef01ac`) plus
its lifetime enforcement (`27eb53e`) changed what a graph-owned `Transient` buffer is:

- It is no longer a resource-manager resource. `GetResource<BufferResource>(name)` returns null; the
  graph's own lookup is `RenderGraphBuilder::ResolveResource` (graph-owned → imported → manager).
- Its content is valid only from its first write **within one execution**. A transient with no writer
  is rejected by name at `Execute()`; a host mapping must not outlive the node whose use declared the
  resource.
- Two transients whose live ranges do not overlap now share one block, so memory is reused between
  them.

Each of the three hit this consumer, and the failure that named the node and the resource was the
useful one:

- The smoke graph's output stopped being allocated at all (`[fail] the graph did not allocate
  SmokeValues`) because it is read back on the host after `Execute()`.
- The 436-node model's weights, uploaded once by the host and read by a node much later, would have
  been overwritten by an aliased neighbour **without any error** — an upload is not part of a live
  range, so the planner cannot see it. This is the silent one, and it is why the rule below is not
  optional.
- `Node 'b0.k_rope' reads transient resource 'rope_scratch' with nothing writing it first` — the rope
  shader's declared-but-unused bindings (frequency factors, set-rows) are read-only and never written.

The rule this leaves: **a resource whose content comes from the host, or is read back by the host,
must be `Persistent` (or imported); only GPU-produced, GPU-consumed intermediates may be `Transient`.**
This is what Muyo's `b60220f` means by "`Persistent` lifetimes still go through the manager". Applied
here: weights, KV caches, the pre-masked `scores`, the unused dummy bindings, the logits and the
per-step I/O are `Persistent`; the per-layer intermediates stay `Transient`.

**Evidence that the aliasing works on a real workload.** With the fix, the model's planner reports
`transients: 157184 bytes reserved for 4959232 individual (4 slots, sharing)` — the per-layer
intermediates pack into 4 slots, about 31x less memory — and the greedy tokens still match llama.cpp
exactly, so the reused bytes are not leaking into the result.

Upstream note: `docs/Consumers.md` still lists "Graph-managed transients have a defined lifetime and
aliasing policy" as a **gap (#16)** even though the aliasing landed. This run is the consumer
exercising it, and the row flip is reported as Muyo
[PR #114](https://github.com/v3c70r/Muyo/pull/114) (with #16 left to close alongside #40 if the
review agrees that criterion 3's remaining in-flight\ >1 scope is #40's).

## M3b — the KV-cache append, resolved without a new primitive

The blocker recorded in F9 is lifted, and not by the part of it we expected. Muyo #81 landed as
commit `a8c4b55` ("A resource use can declare a byte range of a buffer…"): a resource use may carry
a **static** byte `offset`/`size` that is written into `VkDescriptorBufferInfo::offset`/`::range`,
resolved at record time. That is the weight-slicing primitive (an imported slice, a shared buffer),
and the PR deliberately leaves the "offset only known per execution" base case open.

The append itself did not need it. Reading the shaders first, both halves of the KV write have an
expression that already exists in llama.cpp's set:

- **K append.** `rope_neox` has a full 32-bit `d_offset` push constant, so the key of the current
  token is roped straight into `KeyCache` at `d_offset = position * kv_dim`.
- **V append.** `copy.comp` scatters through `dst_idx`, whose base is `get_doffset()` — 16 bits of
  element offset from `misalign_offsets`. Setting the destination strides so the channel stride is
  `seq` and the position stride is 1 turns a row write into a transposed column write, with the
  position as the base.

Both work at `--kvcache` over a growing cache of two positions and match a CPU reference. The
lesson for the assessment: the missing resource-range expression was real for *slices*, but the
first thing a decode engine needs — write this row into that arena at this offset — was already
expressible through per-dispatch push constants, because llama.cpp's shaders were built to be
driven that way. The graph did not have to grow an API for it.

## F15 — GLSL descriptor-heap shaders need a non-uniform-indexing feature the device does not enable

**A, confirmed, worked around.** Against Muyo `b7ce1fe`, `--smoke` aborts in `vkCreateShaderModule`:

```
SPIR-V Capability StorageBufferArrayNonUniformIndexing was declared, but one of the following
requirements must be met (VkPhysicalDeviceVulkan12Features::shaderStorageBufferArrayNonUniformIndexing).
```

glslang emits `ShaderNonUniform` plus `StorageBufferArrayNonUniformIndexing` for a dynamically
indexed `layout(descriptor_heap)` array. `VkRenderDevice::CreateDevice` enables
`shaderSampledImageArrayNonUniformIndexing` but not the storage-buffer counterpart, and a consumer
cannot add it: the `VkPhysicalDeviceVulkan12Features` struct is already in the chain
(`VkRenderDevice.cpp:374`), the same ownership problem as F8. Muyo's own heap shaders are Slang and
reach a resource through a function-local `.Handle()`, which emits neither capability, so its tests
do not exercise the GLSL path.

The index in this consumer is one word of push data, identical for every invocation, so the
non-uniform semantics the capability promises are not used. `tools/strip_nonuniform_capability.py`
removes the two capability lines after `glslc`; `spirv-val --target-env vulkan1.4` accepts the
result and all 12 modes pass (`scripts/sanity.sh`). Evidence: `muyo_llm --smoke`.

Upstream: enable `shaderStorageBufferArrayNonUniformIndexing` (and the other array-indexing features
the heap permits) at device creation, or state that a GLSL heap client may only use constant
indices. The Slang path is not a substitute for GLSL clients.

## F16 — a heap-bound buffer must carry `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT`, and the graph does not add it

**C, confirmed, worked around.** A graph-owned buffer declared with `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`
only, then bound by a node, aborts at `Build()` when the heap service calls `vkGetBufferDeviceAddress`:

```
pInfo->buffer (VkBuffer ...) was created with VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT but requires
VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT.
```

The design note says buffers placed in the resource heap need the bit
(`docs/DescriptorHeap-Bindless-Design.md:379`); `AddResource`/`ImportResource` do not OR it in, and
the failure names the buffer only as a handle. Every buffer declaration in this repo now carries the
bit (15 sites in `src/`). Upstream: add it to the allocation usage when the heap will bind the
resource, or reject the declaration at `Build()` naming the resource and the missing usage. Note the
same requirement applies to imported buffers, where the graph cannot change the usage at all.

## F17 — `muyo_rg` does not propagate the Vulkan include directory to an embedder

**A, confirmed, worked around.** With the 1.4.363 SDK on `CMAKE_PREFIX_PATH`, Muyo's own targets
compile against the SDK headers, but the consumer's translation units still resolve
`<vulkan/vulkan.h>` from `/usr/include` (1.4.313) and fail on every descriptor-heap type
(`PFN_vkCmdPushDataEXT`, `VkPhysicalDeviceDescriptorHeapPropertiesEXT`, ...). `find_package(Vulkan)`
resolves the SDK, but `target_link_libraries(muyo_llm PRIVATE muyo_rg)` carries neither
`${Vulkan_INCLUDE_DIRS}` nor a version floor, although Muyo's public headers include
`<vulkan/vulkan.h>`.

Workaround: the consumer adds `${Vulkan_INCLUDE_DIRS}` and a configure-time compile check that fails
with a named message. Upstream: make the Vulkan headers a PUBLIC usage requirement of `muyo`/
`muyo_rg`, or state the version floor in the embed instructions. The shader build has the same shape:
a distribution `glslc` accepts the command line but not `GL_EXT_descriptor_heap`, so the consumer
prefers the SDK's `glslc` (CMakeLists.txt).

## F18 — a descriptor range larger than `maxStorageBufferRange` is rejected, and `--forward` hits it on llvmpipe

**C, pre-existing, device-limited.** `--forward testdata/qwen2.5-0.5b-instruct-fp16.gguf` binds the
272,269,312-byte embedding tensor as one storage-buffer descriptor. llvmpipe reports
`maxStorageBufferRange = 134,217,728`, so validation aborts:

- heap path (`b7ce1fe`): `vkWriteResourceDescriptorsEXT(): pResources[0].data.pAddressRange->size
  (272269312) is greater than maxStorageBufferRange (134217728)`;
- descriptor-set path (old pin `64947cd`, before this repo's `build/` was rebuilt): `vkUpdateDescriptorSets():
  pBufferInfo[0].range (272269312) is greater than maxStorageBufferRange`.

So the migration did not introduce it; the full-model path was already blocked on this device. The
12 small modes are unaffected, and `--embed`/`--model` succeed. `shader64BitIndexing` would relax the
limit, but `VK_EXT_shader_64bit_indexing` is not exposed by llvmpipe and the consumer cannot enable
it anyway (the feature-chain ownership of F8). Upstream: clamp or split a descriptor range at
`Build()`, or enable `shader64BitIndexing` when the device exposes it. Consumer-side: a weight layout
split at the limit is the model-side fix.

## F19 — the descriptor-heap defaults are frame-sized, and exhaustion names neither the count nor the limit

**C, confirmed, worked around.** The full-model forward (`--forward … -i -n 64`) aborts at `Build()`
on RADV:

```
[fail] DescriptorHeapService: static descriptor region exhausted
```

`DescriptorHeapService::Config` defaults to 256 static buffer slots and 32 ring slots per execution,
which is a frame-sized graph. qwen2.5-0.5B needs **371** distinct static storage-buffer descriptors
before the first token (291 weight tensors, 24 layers x `{kcache, vcache, scores}` = 72, and 8 global
I/O/dummy buffers), and about **820** per-execution descriptor acquisitions per token for the
transient intermediates; the ring is consumed per use, not deduplicated, so 32 is far short. The
exception names neither the region's capacity nor the number needed, and the graph cannot discover
the requirement until it is already allocating.

Workaround: `HeadlessContext` initialises the heap with
`{.imageSlots = 64, .bufferSlots = 2048, .accelerationSlots = 8, .samplerSlots = 16,
.ringSlots = 2048, .inFlightCount = 1}`. That is a host policy decision, which is the right place for
it, but the default cannot serve the workload the library exists to validate. Upstream: either size
the static region from the compiled graph at `Build()` (the count is known before the first
`Acquire*`), or fail with the needed and available counts; the ring is the same problem at `Execute`.

LLVMPIPE note: this host cannot reach the failure because the 272 MB embedding tensor trips F18 first;
the RADV report is the evidence for F19.
