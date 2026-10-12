# Vendored llama.cpp Unicode helpers

These four files are copied verbatim from [llama.cpp](https://github.com/llama.cpp):

- `unicode.h`, `unicode.cpp` — UTF-8 decode/encode and `unicode_regex_split`, the hand-written
  regex engine that understands `\p{L}` / `\p{N}` / `\s` and returns the pre-tokenizer's words.
- `unicode-data.h`, `unicode-data.cpp` — the Unicode category / whitespace / case tables it reads.

They are vendored rather than reimplemented for the same reason the Vulkan shaders are: the
tokenizer's output is only correct if it matches llama.cpp's exactly, and the tables are generated
data. **Do not edit these files** — a change here is a divergence from the reference, and the
byte-level BPE tokenizer in `src/Tokenizer.cpp` is verified against llama.cpp (see
`scripts/verify-forward.sh` and the `--tokenize` mode).

`unicode.cpp` is compiled as part of `muyo_llm` (the source glob under `src/` picks it up).
