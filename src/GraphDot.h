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

/// Write a static Graphviz DOT of the declared graph model.
///
/// This is the *authored* declaration, not an execution trace: it is written before `Build()`, from
/// the node and dependency declarations themselves, so it needs no pipeline compilation, no resource
/// allocation and no execution. Three files are written:
///
/// - `path`: every node, with each `b<layer>.` group as a cluster.
/// - `path-overview`: each layer collapsed to one node, so the 24-layer chain and the globals are
///   readable at a glance.
/// - `path-layer0`: layer 0 in detail, with the global/main-link nodes it touches as context.
///
/// Node groups are the `b<digits>` prefix of a name; anything else (`embed`, `final_norm`, `lm_head`,
/// `argmax`) is "global". `dot -Tsvg path` renders it.
///
/// @param path  Output path for the full graph; the two companions derive from it.
/// @param nodes Node names, in declaration order.
/// @param edges Ordering edges, in declaration order.
void WriteGraphDot(const std::string& path, const std::vector<std::string>& nodes,
                   const std::vector<GraphDotEdge>& edges);
}  // namespace muyo_llm
