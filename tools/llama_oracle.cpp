// A tiny llama.cpp (ggml CPU) oracle for muyo-llm.
//
// Loads a GGUF with the CPU backend, evaluates a fixed sequence of token ids and prints (and
// optionally writes) the logits of the last position. It is the reference the GPU forward is
// compared against: the token ids are passed on the command line so the tokenizer is not part of
// the comparison.
//
// Build (see scripts/build-oracle.sh):
//   g++ -O2 -std=c++17 tools/llama_oracle.cpp -I<llama.cpp>/include -I<llama.cpp>/ggml/include \
//       -L<llama.cpp>/build/bin -lllama -lggml -Wl,-rpath,<llama.cpp>/build/bin -o build/llama_oracle

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: %s <model.gguf> <id,id,...> [logits.bin]\n", argv[0]);
        return 2;
    }

    std::vector<llama_token> tokens;
    {
        std::string field;
        for (const char* cursor = argv[2];; ++cursor)
        {
            if (*cursor == ',' || *cursor == '\0')
            {
                if (!field.empty())
                {
                    tokens.push_back(std::atoi(field.c_str()));
                    field.clear();
                }
                if (*cursor == '\0')
                {
                    break;
                }
            }
            else
            {
                field.push_back(*cursor);
            }
        }
    }
    if (tokens.empty())
    {
        std::fprintf(stderr, "no tokens\n");
        return 2;
    }

    llama_backend_init();
    llama_model_params modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = 0;  // the ggml CPU reference
    llama_model* model = llama_model_load_from_file(argv[1], modelParams);
    if (model == nullptr)
    {
        std::fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);

    llama_context_params contextParams = llama_context_default_params();
    contextParams.n_ctx = tokens.size() + 1;
    contextParams.n_batch = tokens.size();
    llama_context* ctx = llama_init_from_model(model, contextParams);
    if (ctx == nullptr)
    {
        std::fprintf(stderr, "failed to create the context\n");
        return 1;
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    if (llama_decode(ctx, batch) != 0)
    {
        std::fprintf(stderr, "llama_decode failed\n");
        return 1;
    }

    const float* logits = llama_get_logits_ith(ctx, static_cast<int32_t>(tokens.size()) - 1);
    const int32_t nVocab = llama_vocab_n_tokens(vocab);
    if (logits == nullptr)
    {
        std::fprintf(stderr, "no logits\n");
        return 1;
    }

    int32_t argmax = 0;
    for (int32_t i = 1; i < nVocab; ++i)
    {
        if (logits[i] > logits[argmax])
        {
            argmax = i;
        }
    }
    std::printf("[oracle] %d tokens, vocab %d, argmax %d (logit %g)\n", static_cast<int>(tokens.size()), nVocab,
                argmax, logits[argmax]);
    {
        std::vector<int32_t> order(nVocab);
        for (int32_t i = 0; i < nVocab; ++i) order[i] = i;
        std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                          [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
        std::printf("[oracle] top5:");
        for (int i = 0; i < 5; ++i) std::printf(" %d(%.4f)", order[i], logits[order[i]]);
        std::printf("\n");
    }

    const int nPredict = argc > 4 ? std::atoi(argv[4]) : 0;
    if (nPredict > 0)
    {
        std::vector<int32_t> greedy;
        int32_t next = argmax;
        for (int step = 0; step < nPredict; ++step)
        {
            greedy.push_back(next);
            llama_batch one = llama_batch_get_one(&next, 1);
            if (llama_decode(ctx, one) != 0)
            {
                std::fprintf(stderr, "decode failed at step %d\n", step);
                break;
            }
            const float* l = llama_get_logits_ith(ctx, 0);
            next = 0;
            for (int32_t i = 1; i < nVocab; ++i)
            {
                if (l[i] > l[next])
                {
                    next = i;
                }
            }
        }
        std::printf("[oracle] greedy %d:", nPredict);
        for (int32_t t : greedy) std::printf(" %d", t);
        std::printf("\n");
    }

    if (argc > 3)
    {
        std::ofstream out(argv[3], std::ios::binary);
        out.write(reinterpret_cast<const char*>(&nVocab), sizeof(nVocab));
        out.write(reinterpret_cast<const char*>(logits), sizeof(float) * nVocab);
        std::printf("[oracle] wrote %s\n", argv[3]);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
