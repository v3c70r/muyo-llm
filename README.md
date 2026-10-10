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
- **No upstream blocker for M1.** Specialization constants landed in [PR #66](https://github.com/v3c70r/Muyo/pull/66);
  cooperative matrix ([#63](https://github.com/v3c70r/Muyo/issues/63)) is needed for the coopmat GEMM path
  later, not for M1's `add.comp`.
- **What actually blocks starting:** Muyo cannot be configured as a subdirectory today — [Muyo #68](https://github.com/v3c70r/Muyo/issues/68).

Status: planning. No code yet.
