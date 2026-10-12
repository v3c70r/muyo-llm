#!/usr/bin/env bash
# Build the llama.cpp (ggml CPU) oracle and check that the engine's greedy tokens match it.
#
# This is the M4 acceptance check: the whole 24-layer Qwen2 forward, step by step, against the
# reference implementation. It needs the F16 model (git-ignored) and a llama.cpp build, so it is
# not part of scripts/sanity.sh.
#
#   LLAMA_CPP_DIR=~/Projects/llama.cpp \
#   MODEL=testdata/qwen2.5-0.5b-instruct-fp16.gguf \
#   TOKENS=9707,1879 N=8 scripts/verify-forward.sh
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel)" && cd "$ROOT" || exit 1
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/Projects/llama.cpp}"
MODEL="${MODEL:-$ROOT/testdata/qwen2.5-0.5b-instruct-fp16.gguf}"
TOKENS="${TOKENS:-9707,1879}"
N="${N:-8}"

if [ ! -f "$MODEL" ]; then
    echo "verify-forward: skip ($MODEL not present)"
    exit 0
fi
if [ ! -f "$LLAMA_CPP_DIR/include/llama.h" ]; then
    echo "verify-forward: skip (llama.cpp not found at $LLAMA_CPP_DIR)"
    exit 0
fi

g++ -O2 -std=c++17 tools/llama_oracle.cpp \
    -I"$LLAMA_CPP_DIR/include" -I"$LLAMA_CPP_DIR/ggml/include" \
    -L"$LLAMA_CPP_DIR/build/bin" -lllama -lggml \
    -Wl,-rpath,"$LLAMA_CPP_DIR/build/bin" -o build/llama_oracle || exit 1

expected="$(cd build && ./llama_oracle "$MODEL" "$TOKENS" /tmp/muyo_oracle_logits.bin "$N" 2>/dev/null |
    sed -n 's/^\[oracle\] greedy [0-9]*://p')"
# The engine streams generated text to stdout and diagnostics to stderr; the token ids are the last
# stderr line, so read both streams.
actual="$(cd build && ./muyo_llm --forward "$MODEL" --tokens "$TOKENS" -n "$N" 2>&1 >/dev/null |
    sed -n 's/^\[forward\] generated [0-9]* tokens://p')"

echo "oracle:$expected"
echo "engine:$actual"
if [ -n "$expected" ] && [ "$expected" = "$actual" ]; then
    echo "verify-forward: OK ($N greedy tokens match llama.cpp)"
else
    echo "verify-forward: FAILED"
    exit 1
fi

# Tokenizer: compare this repo's Encode against llama.cpp's on several scripts, and on the rendered
# ChatML prompt (which checks the chat template's framing and the special-token ids at once).
SYS="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
CHAT="$(printf '<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n' "$SYS")"
CASES=("Hello world!" "café résumé" "日本語のテスト" "Emoji: 🚀🎉" "it's don't I'll" "123 456.789" "a=b+c<d>" "$CHAT")
tokenizer_bad=0
for text in "${CASES[@]}"; do
    want="$(cd build && ./llama_oracle "$MODEL" --tokenize "$text" 2>/dev/null | sed 's/^\[tokenize\] //')"
    got="$(cd build && ./muyo_llm --tokenize "$MODEL" "$text" 2>/dev/null | grep -oP '(?<=-> ).*')"
    if [ "$want" != "$got" ]; then
        echo "verify-forward: tokenizer DIFF for [$text]"
        tokenizer_bad=1
    fi
done
if [ "$tokenizer_bad" = 0 ]; then
    echo "verify-forward: tokenizer OK (${#CASES[@]} inputs match llama.cpp)"
else
    exit 1
fi
