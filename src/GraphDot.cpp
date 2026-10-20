#include "GraphDot.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace muyo_llm
{
namespace
{
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

/// Quote an identifier for DOT. Node names are `b0.q_proj`-shaped, which is not a bare DOT ID.
std::string Quote(const std::string& text)
{
    std::string out = "\"";
    for (char c : text)
    {
        if (c == '"' || c == '\\')
        {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

std::string LayerLabel(const std::string& group)
{
    if (group.empty())
    {
        return "global";
    }
    return "layer " + group.substr(1);
}

/// A Mermaid node id. Dots are not allowed in a bare id, so they become underscores.
std::string MermaidId(const std::string& name)
{
    std::string id;
    id.reserve(name.size() + 1);
    for (char c : name)
    {
        const auto u = static_cast<unsigned char>(c);
        id.push_back(std::isalnum(u) != 0 ? c : '_');
    }
    if (!id.empty() && std::isdigit(static_cast<unsigned char>(id[0])))
    {
        id = "n_" + id;
    }
    return id;
}

void EmitAttributes(std::ostream& out, int fontSize)
{
    out << "  node [shape=box, fontsize=" << fontSize << ", fontname=\"monospace\"];\n";
    out << "  edge [fontsize=" << (fontSize - 1) << "];\n";
}

/// The full graph: every node, one cluster per group, every authored edge.
void EmitFull(std::ostream& out, const std::vector<std::string>& nodes, const std::vector<GraphDotEdge>& edges)
{
    EmitAttributes(out, 9);

    std::vector<std::string> groups;
    std::set<std::string> seen;
    for (const std::string& node : nodes)
    {
        const std::string group = GroupOf(node);
        if (seen.insert(group).second)
        {
            groups.push_back(group);
        }
    }
    for (const std::string& group : groups)
    {
        out << "  subgraph cluster_" << (group.empty() ? "global" : group) << " {\n";
        out << "    label=" << Quote(LayerLabel(group)) << "; style=dashed; color=gray60;\n";
        for (const std::string& node : nodes)
        {
            if (GroupOf(node) != group)
            {
                continue;
            }
            out << "    " << Quote(node);
            if (group.empty())
            {
                out << " [style=filled, fillcolor=gray95]";
            }
            out << ";\n";
        }
        out << "  }\n";
    }
    for (const GraphDotEdge& edge : edges)
    {
        out << "  " << Quote(edge.from) << " -> " << Quote(edge.to) << ";\n";
    }
}

/// Layer 0 plus its immediate neighbours only (the input it reads and the layer it feeds), not the
/// transitive closure, which would be the whole graph.
std::set<std::string> Layer0Keep(const std::vector<std::string>& nodes, const std::vector<GraphDotEdge>& edges)
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
    return keep;
}

void EmitLayer0(std::ostream& out, const std::vector<std::string>& nodes, const std::vector<GraphDotEdge>& edges)
{
    EmitAttributes(out, 10);
    const std::set<std::string> keep = Layer0Keep(nodes, edges);

    for (const std::string& node : nodes)
    {
        if (keep.count(node) == 0)
        {
            continue;
        }
        const bool inLayer = GroupOf(node) == "b0";
        out << "  " << Quote(node) << " [style=filled, fillcolor=" << (inLayer ? "\"#dbeafe\"" : "gray95") << "];\n";
    }
    for (const GraphDotEdge& edge : edges)
    {
        const bool touchesLayer0 = GroupOf(edge.from) == "b0" || GroupOf(edge.to) == "b0";
        if (touchesLayer0 && keep.count(edge.from) != 0 && keep.count(edge.to) != 0)
        {
            out << "  " << Quote(edge.from) << " -> " << Quote(edge.to) << ";\n";
        }
    }
}

/// Each layer collapsed to one node, so the 24-layer chain and the globals are readable at a glance.
void EmitOverview(std::ostream& out, const std::vector<std::string>& nodes, const std::vector<GraphDotEdge>& edges)
{
    EmitAttributes(out, 11);

    std::vector<std::string> groups;
    std::set<std::string> seen;
    for (const std::string& node : nodes)
    {
        const std::string group = GroupOf(node);
        if (seen.insert(group).second)
        {
            groups.push_back(group);
        }
    }
    for (const std::string& group : groups)
    {
        if (group.empty())
        {
            continue;  // globals are emitted as themselves
        }
        out << "  " << Quote(group) << " [label=" << Quote(LayerLabel(group))
            << ", style=filled, fillcolor=\"#dbeafe\"];\n";
    }
    for (const std::string& node : nodes)
    {
        if (GroupOf(node).empty())
        {
            out << "  " << Quote(node) << " [style=filled, fillcolor=gray95];\n";
        }
    }

    std::set<std::pair<std::string, std::string>> collapsed;
    std::map<std::pair<std::string, std::string>, size_t> counts;
    for (const GraphDotEdge& edge : edges)
    {
        const std::string from = GroupOf(edge.from).empty() ? edge.from : GroupOf(edge.from);
        const std::string to = GroupOf(edge.to).empty() ? edge.to : GroupOf(edge.to);
        if (from == to)
        {
            continue;
        }
        collapsed.insert({from, to});
        ++counts[{from, to}];
    }
    for (const auto& [from, to] : collapsed)
    {
        out << "  " << Quote(from) << " -> " << Quote(to)
            << " [color=\"#2563eb\", penwidth=1.6, label=" << Quote("x" + std::to_string(counts[{from, to}])) << "];\n";
    }
}

void WriteDotFile(const std::string& path, const std::string& body)
{
    std::ofstream out(path);
    out << "digraph muyo_llm {\n";
    out << "  rankdir=LR;\n";
    out << "  graph [fontname=\"monospace\", compound=true];\n";
    out << body;
    out << "}\n";
}

/// Mermaid flowchart for the overview: layers as single nodes.
std::string MermaidOverview(const std::vector<std::string>& nodes, const std::vector<GraphDotEdge>& edges)
{
    std::ostringstream out;
    out << "flowchart LR\n";

    std::vector<std::string> groups;
    std::set<std::string> seen;
    for (const std::string& node : nodes)
    {
        const std::string group = GroupOf(node);
        if (seen.insert(group).second)
        {
            groups.push_back(group);
        }
    }
    for (const std::string& group : groups)
    {
        if (!group.empty())
        {
            out << "  " << MermaidId(group) << "[\"" << LayerLabel(group) << "\"]\n";
        }
    }
    for (const std::string& node : nodes)
    {
        if (GroupOf(node).empty())
        {
            out << "  " << MermaidId(node) << "[\"" << node << "\"]\n";
        }
    }

    std::set<std::pair<std::string, std::string>> collapsed;
    std::map<std::pair<std::string, std::string>, size_t> counts;
    for (const GraphDotEdge& edge : edges)
    {
        const std::string from = GroupOf(edge.from).empty() ? edge.from : GroupOf(edge.from);
        const std::string to = GroupOf(edge.to).empty() ? edge.to : GroupOf(edge.to);
        if (from == to)
        {
            continue;
        }
        collapsed.insert({from, to});
        ++counts[{from, to}];
    }
    for (const auto& [from, to] : collapsed)
    {
        out << "  " << MermaidId(from) << " -->|x" << counts[{from, to}] << "| " << MermaidId(to) << "\n";
    }
    return out.str();
}

/// Mermaid flowchart for layer 0, with its input and the next layer as context.
std::string MermaidLayer0(const std::vector<std::string>& nodes, const std::vector<GraphDotEdge>& edges)
{
    std::ostringstream out;
    out << "flowchart LR\n";

    const std::set<std::string> keep = Layer0Keep(nodes, edges);

    out << "  subgraph layer0[\"layer 0\"]\n    direction LR\n";
    for (const std::string& node : nodes)
    {
        if (keep.count(node) != 0 && GroupOf(node) == "b0")
        {
            out << "    " << MermaidId(node) << "[\"" << ShortLabel(node, "b0") << "\"]\n";
        }
    }
    out << "  end\n";
    for (const std::string& node : nodes)
    {
        if (keep.count(node) != 0 && GroupOf(node) != "b0")
        {
            out << "  " << MermaidId(node) << "[\"" << node << "\"]\n";
        }
    }
    for (const GraphDotEdge& edge : edges)
    {
        const bool touchesLayer0 = GroupOf(edge.from) == "b0" || GroupOf(edge.to) == "b0";
        if (touchesLayer0 && keep.count(edge.from) != 0 && keep.count(edge.to) != 0)
        {
            out << "  " << MermaidId(edge.from) << " --> " << MermaidId(edge.to) << "\n";
        }
    }
    return out.str();
}

void WriteTextFile(const std::string& path, const std::string& text)
{
    std::ofstream out(path);
    out << text;
}
}  // namespace

void WriteGraphDot(const std::string& path, const std::vector<std::string>& nodes,
                   const std::vector<GraphDotEdge>& edges)
{
    {
        std::ostringstream body;
        EmitFull(body, nodes, edges);
        WriteDotFile(path, body.str());
    }
    {
        std::ostringstream body;
        EmitOverview(body, nodes, edges);
        WriteDotFile(path + "-overview", body.str());
    }
    {
        std::ostringstream body;
        EmitLayer0(body, nodes, edges);
        WriteDotFile(path + "-layer0", body.str());
    }

    // Mermaid: GitHub renders it natively, so the raw files and a markdown page are written too.
    const std::string overview = MermaidOverview(nodes, edges);
    const std::string layer0 = MermaidLayer0(nodes, edges);
    WriteTextFile(path + "-overview.mmd", overview);
    WriteTextFile(path + "-layer0.mmd", layer0);

    std::ostringstream page;
    page << "# muyo-llm graph model\n\n";
    page << "Generated from the node and dependency declarations before `Build()` by\n";
    page << "`MUYO_RG_DOT=<path> ./muyo_llm --forward <gguf> -p hi` (see `docs/graph/README.md`).\n";
    page << "GitHub renders the Mermaid below natively.\n\n";
    page << "## Overview\n\nEach layer is one node; the label on an edge is how many authored edges it\n";
    page << "collapses. `embed → layer 0 … layer 23 → final_norm → lm_head → argmax`.\n\n";
    page << "```mermaid\n" << overview << "```\n\n";
    page << "## Layer 0\n\nThe 18-node transformer unit, with the embedding it reads and the next layer it\n";
    page << "feeds as context.\n\n";
    page << "```mermaid\n" << layer0 << "```\n";
    WriteTextFile(path + ".md", page.str());
}
}  // namespace muyo_llm
