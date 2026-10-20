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
void EmitLayer0(std::ostream& out, const std::vector<std::string>& nodes, const std::vector<GraphDotEdge>& edges)
{
    EmitAttributes(out, 10);

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
}
}  // namespace muyo_llm
