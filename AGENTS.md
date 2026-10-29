# AGENTS.md — working agreement

Shared context for anyone working in this repository, human or agent. It exists because work spans
sessions that do not share history, and because this repository is a **consumer** of another one:
the standard is Muyo's, not a new one.

Read [`../Muyo/AGENTS.md`](../Muyo/AGENTS.md) — or the copy at `thirdparty/Muyo/AGENTS.md` — first.
It is the parent standard for branch and merge flow, evidence over assertion, review vocabulary
(`C`/`A`/`N`/`P`, confidence, acceptance criterion) and the definition of done. This file only
records what is different here.

## What this repository is

A lightweight LLM inference engine built directly on Muyo's RenderGraph, and — more to the point —
the **first-party architecture-validation consumer** of it. It is compute-only, windowless and
long-running, which is the workload the RenderGraph has never been exercised by. It is not a
product: throughput is out of scope, and the deliverable is as much the findings as the engine.

- Plan: [`docs/Plan.md`](docs/Plan.md).
- Findings found so far: [`docs/Findings.md`](docs/Findings.md).
- The guarantees this project relies on, and their state: Muyo's `docs/Consumers.md`.

## Building

Muyo is embedded via `add_subdirectory`. The submodule is the shipping path; a development checkout
of Muyo can be substituted without re-initialising submodules.

The Vulkan SDK must be **on the environment and sourced** (`source <sdk>/setup-env.sh`), 1.4.363 or
newer. Since Muyo's B1b the graph is descriptor-heap-only: configuring needs headers that define
`VK_EXT_descriptor_heap`, the shader build needs a `glslc` that accepts `GL_EXT_descriptor_heap`, and
running needs the SDK's loader and validation layers (an older loader rejects the heap structures).
The device must expose `VK_EXT_descriptor_heap` and `VK_KHR_shader_untyped_pointers`; llvmpipe does.
CMake fails with a named message if the headers are too old, rather than letting Muyo's headers
produce the diagnosis.

```bash
# against the submodule (after `git submodule update --init --recursive`)
source "$VULKAN_SDK/setup-env.sh"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build --target muyo_llm -j"$(nproc)"

# or against a Muyo worktree
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DMUYO_SOURCE_DIR=/path/to/Muyo
```

The graph loads `<name>.spv` from `shaders/` **relative to the working directory**
(`Muyo/src/ShaderAsset.h`), so the binary is run from the build directory:

```bash
cd build && ./muyo_llm --smoke
```

## Standards

Same as Muyo's, replicated here so both repositories are judged alike:

- `.clang-format` (clang-format 18) and `.clang-tidy` are copied from Muyo. Format the lines you
  change, never the world.
- `scripts/sanity.sh` is the local entry point: `format` (changed lines), `build` (+ smoke),
  `sanitize` (ASan + UBSan, `san.supp`), `tidy` (report only). `scripts/sanity.sh` with no argument
  runs `format` then `build`. Read its exit status, not a grep of its output.
- Run `sanitize` before any commit that touches memory, lifetimes or resource ownership — which,
  with Vulkan, is most of them.
- Documentation is coverage-enforced once there is a public surface of our own; until then the
  Markdown here is the artifact and is expected to be current.
- Never initialise Vulkan structs positionally (see Muyo's `docs/CodingConventions.md`). The sync2
  barrier layouts make a positional initialiser a silent field shuffle.

## Findings and upstream work

Every gap this project hits is a **finding**, and findings are the point of the repository.
`docs/Findings.md` is the log; `docs/Assessment.md` (M5) is the summary.

- A finding states what was observed, the command or file that shows it, its confidence, and what
  upstream would take to close it. "The docs say so" is not evidence about these headers.
- A gap that Muyo's roadmap already covers gets referenced, not re-filed. One that is not covered
  gets an issue on Muyo **before** the code that needs it — the issue names this project as the
  consumer and the use case, not "a library user".
- Consumer-driven changes to Muyo follow Muyo's process: an up-to-date `master`, an `rg-*` branch, a
  squash PR with a test that has teeth, and a row updated in `docs/Consumers.md`. One concern per PR.
- A workaround here (linking `SPIRV-Reflect` by hand, say) must name the finding it works around, so
  nobody mistakes it for the intended design.

## Definition of done

- [ ] `scripts/sanity.sh` passes (format + build + smoke).
- [ ] `scripts/sanity.sh sanitize` passes for anything touching Vulkan lifetimes.
- [ ] New behaviour has a test with teeth, or the change says why one is impractical. The smoke test
      verifies GPU output against a host-side expectation, so a wrong barrier or binding fails it.
- [ ] Any new gap is recorded in `docs/Findings.md`, and filed upstream if the roadmap does not
      already cover it.
- [ ] This file is updated if the process itself changed.
