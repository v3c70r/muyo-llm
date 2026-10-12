#pragma once

namespace muyo_llm
{
/// Demonstrate and verify the KV-cache append: for each token, rope the key into the cache and
/// scatter the value into the transposed cache, then attend over the cache grown so far.
///
/// Two tokens are run through one graph, at positions 0 and 1. The per-execution position is read
/// by the nodes' record-time callbacks from a value the caller updates between executions - which
/// is what makes the append possible without rewriting any descriptor: the rope's destination base
/// is a full 32-bit push-constant field, and `copy.comp`'s is 16 bits of `misalign_offsets`, both
/// far wider than an 8-bit alignment field.
///
/// Assumes the Muyo subsystems are already initialised (see `HeadlessContext`).
/// @return 0 when both tokens' attention outputs match the CPU reference, 1 otherwise.
int RunKvCache();
}  // namespace muyo_llm
