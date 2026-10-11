#pragma once

namespace muyo_llm
{
/// Run scaled dot-product attention for one query through the graph, using three nodes:
/// a GEMV for `scores = K . q`, a softmax, and a GEMV for `out = V^T . probs`.
///
/// This is the first composition - more than one node, real intermediate graph-owned resources,
/// and an explicit dependency chain rather than a single dispatch. It is the shape a transformer
/// layer is built from, and the first place the graph's ordering and barrier generation are
/// exercised across nodes rather than within one.
///
/// Assumes the Muyo subsystems are already initialised (see `HeadlessContext`).
/// @return 0 when the GPU result matches the CPU reference within tolerance, 1 otherwise.
int RunAttention();
}  // namespace muyo_llm
