#pragma once

#include <string>
#include <vector>

namespace muyo_llm
{
/// One ordering edge of the authored graph model, by node name.
struct GraphDotEdge
{
    std::string from;
    std::string to;
};

/// Write diagrams of the declared graph model, using Muyo's `ToDot`/`ToMermaid` renderers.
///
/// This module no longer formats any diagram syntax itself: it projects the recorded nodes and
/// dependency edges into a `Muyo::RenderGraph::TraceGraph` (a labelled topology) and calls
/// `RenderGraph::ToDot` / `RenderGraph::ToMermaid` on it. The projection is per node, not per
/// execution segment, because that is the question this consumer asks — "what is the model" — and a
/// compute-only graph compiles to a single segment.
///
/// It is written before `Build()` from the declaration, so no pipeline is compiled, no resource is
/// allocated and nothing executes. Three projections are written:
///
/// - `path`, `path-overview`, `path-layer0`: Graphviz DOT.
/// - `path-overview.mmd`, `path-layer0.mmd`: the same overview and layer as Mermaid.
/// - `path.md`: a markdown page embedding both Mermaid blocks, which GitHub renders natively.
///
/// Node groups are the `b<digits>` prefix of a name; anything else (`embed`, `final_norm`, `lm_head`,
/// `argmax`) is "global". `dot -Tsvg path` renders a DOT file.
///
/// @param path  Output path for the full graph; the companions derive from it.
/// @param nodes Node names, in declaration order.
/// @param edges Ordering edges, in declaration order.
void WriteGraphDiagrams(const std::string& path, const std::vector<std::string>& nodes,
                        const std::vector<GraphDotEdge>& edges);
}  // namespace muyo_llm
