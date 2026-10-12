#!/usr/bin/env python3
"""A small numpy Qwen2 reference forward.

Loads a Qwen2 GGUF and runs the same decode the engine runs, so a wrong intermediate can be
localised by comparing hidden states. The tokenizer is not used: token ids are given on the
command line. Written to validate `--forward` against llama.cpp's CPU logits.
"""
import os, struct, sys, numpy as np

def parse_gguf(path):
    f = open(path, 'rb')
    def u32(): return struct.unpack('<I', f.read(4))[0]
    def u64(): return struct.unpack('<Q', f.read(8))[0]
    def s():
        n = u64(); return f.read(n).decode('utf-8', 'replace')
    def skip(t):
        if t in (0, 1, 7): f.read(1)
        elif t in (2, 3): f.read(2)
        elif t in (4, 5, 6): f.read(4)
        elif t == 8: s()
        elif t == 9:
            et = u32(); n = u64()
            for _ in range(n): skip(et)
        elif t in (10, 11, 12): f.read(8)
    assert f.read(4) == b'GGUF'
    ver = u32(); nten = u64(); nkv = u64()
    kv = {}
    for _ in range(nkv):
        k = s(); t = u32()
        if t == 4: kv[k] = struct.unpack('<I', f.read(4))[0]
        elif t == 5: kv[k] = struct.unpack('<i', f.read(4))[0]
        elif t == 6: kv[k] = struct.unpack('<f', f.read(4))[0]
        elif t == 8: kv[k] = s()
        elif t == 10: kv[k] = u64()
        elif t == 11: kv[k] = struct.unpack('<q', f.read(8))[0]
        elif t == 12: kv[k] = struct.unpack('<d', f.read(8))[0]
        elif t == 9:
            et = u32(); n = u64()
            for _ in range(n): skip(et)
            kv[k] = [None] * n
        else: skip(t)
    tensors = {}
    for _ in range(nten):
        name = s(); nd = u32(); dims = [u64() for _ in range(nd)]; ty = u32(); off = u64()
        tensors[name] = (ty, dims, off)
    align = kv.get('general.alignment', 32)
    pos = f.tell()
    data_off = (pos + align - 1) & ~(align - 1)
    return kv, tensors, data_off

class Model:
    def __init__(self, path):
        self.kv, self.tensors, self.data_off = parse_gguf(path)
        self.f = open(path, 'rb')
        self.hidden = self.kv['qwen2.embedding_length']
        self.layers = self.kv['qwen2.block_count']
        self.heads = self.kv['qwen2.attention.head_count']
        self.kv_heads = self.kv['qwen2.attention.head_count_kv']
        self.head_dim = self.hidden // self.heads
        self.query_dim = self.heads * self.head_dim
        self.kv_dim = self.kv_heads * self.head_dim
        self.gqa = self.heads // self.kv_heads
        self.eps = self.kv['qwen2.attention.layer_norm_rms_epsilon']
        self.rope_base = self.kv['qwen2.rope.freq_base']
        self.vocab = len(self.kv['tokenizer.ggml.tokens'])

    def t(self, name):
        ty, dims, off = self.tensors[name]
        n = 1
        for d in dims: n *= d
        dt = np.float16 if ty == 1 else np.float32
        self.f.seek(self.data_off + off)
        # ggml lays tensors out with ne[0] fastest, i.e. flat[i1*ne0 + i0]; reversing the
        # dims makes a C-order numpy array agree with it.
        return np.frombuffer(self.f.read(n * np.dtype(dt).itemsize), dtype=dt).astype(np.float32).reshape(dims[::-1])

    def w(self, name):
        # ggml dims are [innermost, ...]; a matmul weight [K, N] is used as [N, K].
        return self.t(name)  # already [ne1 rows, ne0 cols] = [out, in]

def rms_norm(x, w, eps):
    return x / np.sqrt((x * x).mean() + eps) * w

def rope_neox(x, pos, base, head_dim):
    # x: [heads, head_dim]; rotate pairs (j, j+head_dim/2)
    half = head_dim // 2
    j = np.arange(half)
    theta = pos * (base ** (-2.0 * j / head_dim))
    c, s = np.cos(theta), np.sin(theta)
    out = np.empty_like(x)
    out[:, :half] = x[:, :half] * c - x[:, half:] * s
    out[:, half:] = x[:, :half] * s + x[:, half:] * c
    return out

def softmax(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)

def main():
    path, ids, out = sys.argv[1], [int(v) for v in sys.argv[2].split(',')], (sys.argv[3] if len(sys.argv) > 3 else None)
    m = Model(path)
    print(f"[ref] layers={m.layers} hidden={m.hidden} heads={m.heads} kv={m.kv_heads} vocab={m.vocab}", file=sys.stderr)
    logits = None
    kc = [None] * m.layers
    vc = [None] * m.layers
    for step, tok in enumerate(ids):
        h = m.t('token_embd.weight')[tok].copy()
        for l in range(m.layers):
            p = f'blk.{l}.'
            xn = rms_norm(h, m.t(p + 'attn_norm.weight'), m.eps)
            q = m.w(p + 'attn_q.weight') @ xn + m.t(p + 'attn_q.bias')
            k = m.w(p + 'attn_k.weight') @ xn + m.t(p + 'attn_k.bias')
            v = m.w(p + 'attn_v.weight') @ xn + m.t(p + 'attn_v.bias')
            v = v.reshape(m.kv_heads, m.head_dim)
            if step == 0 and l == 0 and os.environ.get('REF_DUMP'):
                d = os.environ['REF_DUMP']
                h.tofile(f'{d}/embedding.bin'); xn.tofile(f'{d}/b0.xn.bin')
                q.tofile(f'{d}/b0.q.bin'); k.tofile(f'{d}/b0.kvec.bin')
            q = rope_neox(q.reshape(m.heads, m.head_dim), step, m.rope_base, m.head_dim)
            if step == 0 and l == 0 and os.environ.get('REF_DUMP'):
                q.reshape(-1).tofile(f"{os.environ['REF_DUMP']}/b0.qr.bin")
                v.reshape(-1).tofile(f"{os.environ['REF_DUMP']}/b0.vvec.bin")
            k = rope_neox(k.reshape(m.kv_heads, m.head_dim), step, m.rope_base, m.head_dim)
            # cache (per layer)
            if kc[l] is None:
                kc[l] = np.zeros((m.kv_heads, 0, m.head_dim), np.float32)
                vc[l] = np.zeros((m.kv_heads, 0, m.head_dim), np.float32)
            kc[l] = np.concatenate([kc[l], k[:, None, :]], axis=1)
            vc[l] = np.concatenate([vc[l], v[:, None, :]], axis=1)
            # GQA attention
            qh = q.reshape(m.heads, 1, m.head_dim)
            kh = np.repeat(kc[l], m.gqa, axis=0)  # [heads, seq, head_dim]
            vh = np.repeat(vc[l], m.gqa, axis=0)
            scores = (qh @ kh.transpose(0, 2, 1)).squeeze(1) / np.sqrt(m.head_dim)
            probs = softmax(scores)
            attn = (probs[:, None, :] @ vh).squeeze(1).reshape(m.query_dim)
            o = m.w(p + 'attn_output.weight') @ attn
            if step == 0 and l == 0 and os.environ.get('REF_DUMP'):
                d = os.environ['REF_DUMP']
                scores.reshape(-1).tofile(f'{d}/b0.scores.bin')
                probs.reshape(-1).tofile(f'{d}/b0.probs.bin')
                attn.tofile(f'{d}/b0.attn_out.bin')
                o.tofile(f'{d}/b0.o.bin')
            h = h + o
            if step == 0 and l == 0 and os.environ.get('REF_DUMP'):
                h.tofile(f"{os.environ['REF_DUMP']}/b0.x1.bin")
            xn2 = rms_norm(h, m.t(p + 'ffn_norm.weight'), m.eps)
            gate = m.w(p + 'ffn_gate.weight') @ xn2
            up = m.w(p + 'ffn_up.weight') @ xn2
            act = (gate / (1 + np.exp(-gate))) * up
            down = m.w(p + 'ffn_down.weight') @ act
            if step == 0 and l == 0 and os.environ.get('REF_DUMP'):
                d = os.environ['REF_DUMP']
                gate.tofile(f'{d}/b0.gate.bin'); up.tofile(f'{d}/b0.up.bin'); act.tofile(f'{d}/b0.act.bin')
                down.tofile(f'{d}/b0.mlp_out.bin')
            h = h + down
            if step == 0 and l == 0 and os.environ.get('REF_DUMP'):
                h.tofile(f"{os.environ['REF_DUMP']}/b0.out.bin")
        hn = rms_norm(h, m.t('output_norm.weight'), m.eps)
        if step == 0 and os.environ.get('REF_DUMP'):
            hn.tofile(f"{os.environ['REF_DUMP']}/final_norm.bin")
        logits = m.w('output.weight') @ hn
        if step == 0 and os.environ.get('REF_DUMP'):
            logits.tofile(f"{os.environ['REF_DUMP']}/logits.bin")
        arg = int(np.argmax(logits))
        top = np.argsort(-logits)[:5]
        print(f"[ref] step {step} argmax {arg} top5 {[(int(i), round(float(logits[i]),4)) for i in top]}", file=sys.stderr)
    if out:
        with open(out, 'wb') as f:
            f.write(struct.pack('<i', m.vocab)); f.write(logits.astype(np.float32).tobytes())
        print(f"[ref] wrote {out}", file=sys.stderr)

main()
