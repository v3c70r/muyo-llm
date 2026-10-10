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

## F2 — a compute-only graph still allocates the graphics semantic descriptor sets

**A, confirmed.** Constructing a `RenderGraphBuilder` constructs `RenderGraphDescriptorSets`,
whose constructor unconditionally allocates the three built-in semantic set layouts
(`PER_VIEW`, `PER_OBJ`, `MATERIAL`), and `MATERIAL` carries a `MAX_BINDLESS_TEXTURE_COUNT = 1024`
combined-image-sampler array (`RenderGraphDescriptorSets.h`). A graph whose every node is compute
with explicit `DescriptorBinding`s never touches any of them, but pays for them at construction, and
forces a headless consumer to initialise a sampler manager it does not use. Evidence: our smoke graph
adds one explicit-binding compute node and never uses a semantic set. Upstream candidate: allocate a
semantic layout lazily, on the first graphics node that needs it — which is also where #15's
content-addressed design lands.

## F3 — `STORAGE_BUFFER` + `WRITE` means a *host* write

**C, confirmed.** `ResourceUseResolver.h`'s `K_USAGE_POLICIES` maps `{STORAGE_BUFFER, WRITE}` to
`{VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_WRITE_BIT}`. A compute shader that writes a storage
buffer must therefore declare `READ_WRITE` to get
`ALL_COMMANDS | SHADER_READ | SHADER_WRITE`; the natural spelling (`WRITE`) produces a host-stage
policy and no compute barrier. Nothing fails loudly — the shader still runs, but the graph has
recorded the wrong synchronisation for it. Evidence: the policy table, and our smoke node, which
uses `READ_WRITE`. Upstream candidate: either a GPU-write spelling distinct from the host-write one,
or reject `{STORAGE_BUFFER, WRITE}` at `Build()` for a compute node.

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
