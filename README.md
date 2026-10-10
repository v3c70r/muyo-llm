# muyo-llm

A lightweight LLM inference engine built on [Muyo](https://github.com/v3c70r/Muyo)'s
RenderGraph library.

It exists to be the second, non-renderer consumer of `muyo_rg`: a compute-only, windowless,
long-running workload that tests the library boundary the Muyo roadmap asks for. It is
architecture validation first and an inference engine second.

- **Scope (v1):** decode-only, F16 weights, Qwen2.5-0.5B, llama.cpp shaders.
- **Plan:** [`docs/Plan.md`](docs/Plan.md).
- **Specialization constants:** supported upstream as of [Muyo PR #66](https://github.com/v3c70r/Muyo/pull/66)
  (issue [#62](https://github.com/v3c70r/Muyo/issues/62), closed). Nodes declare `{id, value}` and the graph
  takes the width from reflection, so this no longer blocks anything.
- **Upstream blocker:** cooperative-matrix enablement, [Muyo #63](https://github.com/v3c70r/Muyo/issues/63) —
  the plan carries no non-coopmat fallback, so this blocks the implementation rather than one milestone.
- **And the development machine's GPU does not have it.** `VK_KHR_cooperative_matrix` is exposed by the
  **software** device (llvmpipe, rev 2) and **not** by the real one (RADV, RDNA2, `deviceID 0x1681`). Where
  this project runs its coopmat path — a second machine, a fallback path, or llvmpipe for correctness only —
  is a decision, and §5.3 records it as one.
- **Specialization constants:** landed in [Muyo PR #66](https://github.com/v3c70r/Muyo/pull/66); no longer a gap.
- **Also blocking starting:** Muyo cannot be configured as a subdirectory today — [Muyo #68](https://github.com/v3c70r/Muyo/issues/68).

Status: planning. No code yet.
