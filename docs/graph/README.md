# Generated graph diagrams

Static Graphviz DOT of `muyo-llm`'s authored graph model, produced from the node and dependency
declarations **before** `Build()` (no pipeline, no allocation, no execution). Regenerate with:

```bash
cd build
MUYO_RG_DOT=/tmp/model.dot ./muyo_llm --forward ../testdata/qwen2.5-0.5b-instruct-fp16.gguf -p hi
cp /tmp/model.dot /tmp/model.dot-overview /tmp/model.dot-layer0 ../docs/graph/
dot -Tsvg ../docs/graph/model-overview.dot -o model-overview.svg
dot -Tpng ../docs/graph/model-layer0.dot   -o model-layer0.png
```

- `model.dot` — all 436 nodes, one cluster per transformer layer, 507 ordering edges.
- `model-overview.dot` — each layer collapsed to one node:
  `embed → layer 0 … layer 23 → final_norm → lm_head → argmax` (27 edges).
- `model-layer0.dot` — the 18-node layer in detail (22 edges).

These are artifacts, not sources: they change when the graph does. This is the static model, not
Muyo's execution trace (#165/#166), which records barriers, segments and timeline values per
execution and is emitted from `Execute()`.
