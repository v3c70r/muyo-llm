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

**A, confirmed.** `muyo` compiles `src/ShaderReflectionFetcher.cpp`, which calls `spvReflect*`, but
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
