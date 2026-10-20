#include "GraphDot.h"

#include <cctype>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "RenderGraph/RenderGraphTrace.h"

namespace muyo_llm
{
namespace
{
using Muyo::RenderGraph::ToDot;
using Muyo::RenderGraph::ToMermaid;
using Muyo::RenderGraph::TraceGraph;
using Muyo::RenderGraph::TraceGraphEdge;
using Muyo::RenderGraph::TraceGraphNode;

/// A node group: `b<digits>` for a layer, empty for a global node.
std::string GroupOf(const std::string& name)
{
    if (name.size() >= 2 && name[0] == 'b' && std::isdigit(static_cast<unsigned char>(name[1])))
    {
        const size_t dot = name.find('.');
        if (dot != std::string::npos)
        {
            return name.substr(0, dot);
        }
    }
    return {};
}

std::string ShortLabel(const std::string& node, const std::string& group)
{
    if (!group.empty() && node.compare(0, group.size(), group) == 0 && node.size() > group.size() &&
        node[group.size()] == '.')
    {
        return node.substr(group.size() + 1);
    }
    return node;
}

std::string LayerLabel(const std::string& group) { return group.empty() ? "global" : ("layer " + group.substr(1)); }

/// Every declared node and every declared edge, one graph node per declared node.
TraceGraph FullGraph(const std::vector<std::string>& nodes, const std::vector<GraphDotEdge>& edges)
{
    TraceGraph graph;
    std::map<std::string, std::size_t> index;
    for (const std::string& node : nodes)
    {
        const std::size_t at = graph.nodes.size();
        index.emplace(node, at);
        graph.nodes.push_back(TraceGraphNode{.segment = at, .label = node});
    }
    for (const GraphDotEdge& edge : edges)
    {
        const auto from = index.find(edge.from);
        const auto to = index.find(edge.to);
        if (from != index.end() && to != index.end())
        {
            graph.edges.push_back(TraceGraphEdge{.from = from->second, .to = to->second, .label = {}});
        }
    }
    return graph;
}

/// Each layer collapsed to one node; an edge label is how many authored edges it stands for.
TraceGraph OverviewGraph(const std::vector<std::string>& nodes, const std::vector<GraphDotEdge>& edges)
{
    TraceGraph graph;
    // A layer is one node keyed by its group; a global node is its own key (the empty group is not a
    // shared bucket, or every global would collapse into the first one).
    std::map<std::string, std::size_t> index;
    for (const std::string& node : nodes)
    {
        const std::string group = GroupOf(node);
        const std::string key = group.empty() ? node : group;
        if (index.count(key) != 0)
        {
            continue;
        }
        const std::size_t at = graph.nodes.size();
        index.emplace(key, at);
        graph.nodes.push_back(TraceGraphNode{.segment = at, .label = group.empty() ? node : LayerLabel(group)});
    }

    std::vector<std::pair<std::size_t, std::size_t>> order;
    std::set<std::pair<std::size_t, std::size_t>> seen;
    std::map<std::pair<std::size_t, std::size_t>, size_t> counts;
    for (const GraphDotEdge& edge : edges)
    {
        const std::string fromGroup = GroupOf(edge.from);
        const std::string toGroup = GroupOf(edge.to);
        const auto from = index.find(fromGroup.empty() ? edge.from : fromGroup);
        const auto to = index.find(toGroup.empty() ? edge.to : toGroup);
        if (from == index.end() || to == index.end() || from->second == to->second)
        {
            continue;
        }
        const auto key = std::make_pair(from->second, to->second);
        if (seen.insert(key).second)
        {
            order.push_back(key);  // declaration order, not lexicographic
        }
        ++counts[key];
    }
    for (const auto& [from, to] : order)
    {
        graph.edges.push_back(
            TraceGraphEdge{.from = from, .to = to, .label = "x" + std::to_string(counts[{from, to}])});
    }
    return graph;
}

/// Layer 0 plus its immediate neighbours only (the input it reads and the layer it feeds), not the
/// transitive closure, which would be the whole graph.
TraceGraph Layer0Graph(const std::vector<std::string>& nodes, const std::vector<GraphDotEdge>& edges)
{
    std::set<std::string> keep;
    for (const std::string& node : nodes)
    {
        if (GroupOf(node) == "b0")
        {
            keep.insert(node);
        }
    }
    for (const GraphDotEdge& edge : edges)
    {
        if (GroupOf(edge.from) == "b0")
        {
            keep.insert(edge.to);
        }
        if (GroupOf(edge.to) == "b0")
        {
            keep.insert(edge.from);
        }
    }

    TraceGraph graph;
    std::map<std::string, std::size_t> index;
    for (const std::string& node : nodes)
    {
        if (keep.count(node) == 0)
        {
            continue;
        }
        const std::size_t at = graph.nodes.size();
        index.emplace(node, at);
        graph.nodes.push_back(TraceGraphNode{.segment = at, .label = ShortLabel(node, GroupOf(node))});
    }
    for (const GraphDotEdge& edge : edges)
    {
        const bool touchesLayer0 = GroupOf(edge.from) == "b0" || GroupOf(edge.to) == "b0";
        const auto from = index.find(edge.from);
        const auto to = index.find(edge.to);
        if (touchesLayer0 && from != index.end() && to != index.end())
        {
            graph.edges.push_back(TraceGraphEdge{.from = from->second, .to = to->second, .label = {}});
        }
    }
    return graph;
}

void WriteTextFile(const std::string& path, const std::string& text)
{
    std::ofstream out(path);
    out << text;
}
}  // namespace

void WriteGraphDiagrams(const std::string& path, const std::vector<std::string>& nodes,
                        const std::vector<GraphDotEdge>& edges)
{
    const TraceGraph full = FullGraph(nodes, edges);
    const TraceGraph overview = OverviewGraph(nodes, edges);
    const TraceGraph layer0 = Layer0Graph(nodes, edges);

    WriteTextFile(path, ToDot(full));
    WriteTextFile(path + "-overview", ToDot(overview));
    WriteTextFile(path + "-layer0", ToDot(layer0));

    const std::string overviewMermaid = ToMermaid(overview);
    const std::string layer0Mermaid = ToMermaid(layer0);
    WriteTextFile(path + "-overview.mmd", overviewMermaid);
    WriteTextFile(path + "-layer0.mmd", layer0Mermaid);

    std::ostringstream page;
    page << "# muyo-llm graph model\n\n";
    page << "Generated from the node and dependency declarations before `Build()` by\n";
    page << "`MUYO_RG_DOT=<path> ./muyo_llm --forward <gguf> -p hi` (see `docs/graph/README.md`).\n";
    page << "The diagram syntax is Muyo's `RenderGraph::ToDot`/`ToMermaid`; this repo only projects the\n";
    page << "declared nodes and edges into the `TraceGraph` topology they take. GitHub renders the\n";
    page << "Mermaid below natively.\n\n";
    page << "## Overview\n\nEach layer is one node; the label on an edge is how many authored edges it\n";
    page << "collapses. `embed → layer 0 … layer 23 → final_norm → lm_head → argmax`.\n\n";
    page << "```mermaid\n" << overviewMermaid << "```\n\n";
    page << "## Layer 0\n\nThe 18-node transformer unit, with the embedding it reads and the next layer it\n";
    page << "feeds as context.\n\n";
    page << "```mermaid\n" << layer0Mermaid << "```\n";
    WriteTextFile(path + ".md", page.str());
}
}  // namespace muyo_llm
