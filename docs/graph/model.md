# muyo-llm graph model

Generated from the node and dependency declarations before `Build()` by
`MUYO_RG_DOT=<path> ./muyo_llm --forward <gguf> -p hi` (see `docs/graph/README.md`).
The diagram syntax is Muyo's `RenderGraph::ToDot`/`ToMermaid`; this repo only projects the
declared nodes and edges into the `TraceGraph` topology they take. GitHub renders the
Mermaid below natively.

## Overview

Each layer is one node; the label on an edge is how many authored edges it
collapses. `embed → layer 0 … layer 23 → final_norm → lm_head → argmax`.

```mermaid
flowchart LR
  s0["layer 0"]
  s1["layer 1"]
  s2["layer 2"]
  s3["layer 3"]
  s4["layer 4"]
  s5["layer 5"]
  s6["layer 6"]
  s7["layer 7"]
  s8["layer 8"]
  s9["layer 9"]
  s10["layer 10"]
  s11["layer 11"]
  s12["layer 12"]
  s13["layer 13"]
  s14["layer 14"]
  s15["layer 15"]
  s16["layer 16"]
  s17["layer 17"]
  s18["layer 18"]
  s19["layer 19"]
  s20["layer 20"]
  s21["layer 21"]
  s22["layer 22"]
  s23["layer 23"]
  s24["embed"]
  s25["final_norm"]
  s26["lm_head"]
  s27["argmax"]
  s0 -->|"x1"| s1
  s1 -->|"x1"| s2
  s2 -->|"x1"| s3
  s3 -->|"x1"| s4
  s4 -->|"x1"| s5
  s5 -->|"x1"| s6
  s6 -->|"x1"| s7
  s7 -->|"x1"| s8
  s8 -->|"x1"| s9
  s9 -->|"x1"| s10
  s10 -->|"x1"| s11
  s11 -->|"x1"| s12
  s12 -->|"x1"| s13
  s13 -->|"x1"| s14
  s14 -->|"x1"| s15
  s15 -->|"x1"| s16
  s16 -->|"x1"| s17
  s17 -->|"x1"| s18
  s18 -->|"x1"| s19
  s19 -->|"x1"| s20
  s20 -->|"x1"| s21
  s21 -->|"x1"| s22
  s22 -->|"x1"| s23
  s24 -->|"x1"| s0
  s23 -->|"x1"| s25
  s25 -->|"x1"| s26
  s26 -->|"x1"| s27
```

## Layer 0

The 18-node transformer unit, with the embedding it reads and the next layer it
feeds as context.

```mermaid
flowchart LR
  s0["pre_norm"]
  s1["q_proj"]
  s2["k_proj"]
  s3["v_proj"]
  s4["q_rope"]
  s5["k_rope"]
  s6["v_copy"]
  s7["attn_scores"]
  s8["attn_weights"]
  s9["attn_output"]
  s10["o_proj"]
  s11["residual1"]
  s12["post_norm"]
  s13["gate_proj"]
  s14["up_proj"]
  s15["swiglu"]
  s16["down_proj"]
  s17["residual2"]
  s18["pre_norm"]
  s19["embed"]
  s0 --> s1
  s0 --> s2
  s0 --> s3
  s1 --> s4
  s2 --> s5
  s3 --> s6
  s4 --> s7
  s5 --> s7
  s6 --> s7
  s7 --> s8
  s8 --> s9
  s9 --> s10
  s10 --> s11
  s11 --> s12
  s12 --> s13
  s12 --> s14
  s13 --> s15
  s14 --> s15
  s15 --> s16
  s16 --> s17
  s17 --> s18
  s19 --> s0
```
