# Graph diagrams

Static diagrams of `muyo-llm`'s declared graph model. The diagram **syntax** is Muyo's
`RenderGraph::ToDot` / `RenderGraph::ToMermaid` (exposed for the compiled plan in Muyo #181); this
repository only projects the recorded nodes and dependency edges into the `TraceGraph` topology they
take. The projection is per **node**, not per execution segment, because a compute-only graph
compiles to one segment — the question asked here is "what is the model".

It is written before `Build()`, so no pipeline is compiled, no resource is allocated and nothing
executes. GitHub renders the Mermaid in [`model.md`](model.md) natively.

Regenerate with:

```bash
cd build
MUYO_RG_DOT=/tmp/model.dot ./muyo_llm --forward ../testdata/qwen2.5-0.5b-instruct-fp16.gguf -p hi
cp /tmp/model.dot /tmp/model.dot-overview /tmp/model.dot-layer0 \
   /tmp/model.dot-overview.mmd /tmp/model.dot-layer0.mmd /tmp/model.dot.md ../docs/graph/
```

| File | What it shows |
| --- | --- |
| [`model.md`](model.md) | **the two Mermaid diagrams below, GitHub renders them inline** |
| `model-overview.mmd` | Mermaid: each layer collapsed to one node |
| `model-layer0.mmd` | Mermaid: layer 0 in detail |
| `model.dot` | Graphviz: all 436 nodes, 507 ordering edges |
| `model-overview.dot` | Graphviz: `embed → layer 0 … layer 23 → final_norm → lm_head → argmax` (27 edges) |
| `model-layer0.dot` | Graphviz: the 18-node layer (22 edges) |

Render the DOT files with `dot -Tsvg model-overview.dot -o model-overview.svg`.

These are artifacts, not sources: they change when the graph does. This is the **static model**,
not Muyo's execution trace (#165/#166), which records segments, barriers and timeline values per
execution and is emitted from `Execute()` via `SetTraceSink` or `MUYO_RG_TRACE`.
