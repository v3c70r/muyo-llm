#pragma once

namespace muyo_llm
{
/// Run one transformer decoder layer (single head, no GQA) through the graph and verify it against
/// a CPU reference.
///
/// Fourteen nodes in one dependency chain:
/// `norm -> q_proj -> rope -> scores -> softmax -> attn_out -> o_proj -> residual`
/// `-> norm -> gate_proj, up_proj -> swiglu -> down_proj -> residual`.
///
/// This is the deep, narrow graph the RenderGraph has never been exercised by: many small compute
/// nodes, real intermediate graph-owned buffers, and a chain of read-after-write dependencies the
/// graph must order and barrier. The K/V projections and the KV-cache append are deliberately not
/// here yet - the cache is provided by the host, transposed for the `V^T . p` product, because the
/// append needs a mechanism the graph does not have (see docs/Findings.md).
///
/// Assumes the Muyo subsystems are already initialised (see `HeadlessContext`).
/// @return 0 when the GPU result matches the CPU reference within tolerance, 1 otherwise.
int RunTransformerLayer();
}  // namespace muyo_llm
