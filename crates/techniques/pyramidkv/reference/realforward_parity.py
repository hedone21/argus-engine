#!/usr/bin/env python3
"""Real-forward keep-set parity: run the REAL kvpress PyramidKVPress on qwen2.5-1.5b-instruct +
the same 502-token prompt the argus engine prefilled, capture the per-layer per-kv-head kept
positions, and compare against the engine's ACTUAL committed keep-set (ARGUS_DUMP_KEEPSET).

Since the decision logic is already proven byte-identical offline, any keep-set difference here is
the attention-computation gap between the two stacks (Rust engine vs PyTorch) on identical weights.
Reported as per-(layer,head) Jaccard + exact-match rate. Runs kvpress in both f16 (matches the
engine's F16 weights) and f32 to bracket the dtype effect."""
import sys, types, shlex, json, argparse
m = types.ModuleType('pipes'); m.quote = shlex.quote; sys.modules['pipes'] = m
import torch
from kvpress import PyramidKVPress
from kvpress.presses import pyramidkv_press as _pkv

MODEL = "/home/go/Workspace/argus/models/qwen2.5-1.5b-instruct"

ap = argparse.ArgumentParser()
ap.add_argument("--engine-keepset", required=True)
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
ap.add_argument("--cr", type=float, default=0.5)
ap.add_argument("--out", default=None, help="write parity metrics JSON here")
args = ap.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=getattr(torch, args.dtype), attn_implementation="eager").eval()

prompt = open(args.prompt_file).read().strip()
ids = tok(prompt, return_tensors="pt").input_ids
seq_len = int(ids.shape[1])

# ── capture per-layer per-kv-head kept positions from the REAL PyramidKVPress.compress ──
captured = {}  # layer_idx -> [ sorted kept positions per kv-head ]
_orig_compress = _pkv.PyramidKVPress.compress
def capturing_compress(self, module, hidden_states, keys, values, attentions, kwargs):
    # mirror the real compress body to grab `indices` (verbatim from pyramidkv_press.py)
    if self.compression_ratio == 0:
        return keys, values
    scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
    k_len = keys.shape[2]
    n_kept = self.get_layer_budget(module, k_len)
    indices = scores.topk(n_kept, dim=-1).indices  # [bsz, n_kv, n_kept]
    captured[int(module.layer_idx)] = [sorted(indices[0, h].tolist()) for h in range(indices.shape[1])]
    return _orig_compress(self, module, hidden_states, keys, values, attentions, kwargs)
_pkv.PyramidKVPress.compress = capturing_compress

press = PyramidKVPress(compression_ratio=args.cr, window_size=64, kernel_size=5, beta=20)
with torch.no_grad(), press(model):
    model(ids, use_cache=True)

# ── compare against the engine dump ──
eng = json.load(open(args.engine_keepset))
assert eng["seq_len"] == seq_len, f"seq_len mismatch: engine={eng['seq_len']} kvpress={seq_len}"
n_layers = eng["n_layers"]; n_kv = eng["n_kv_heads"]

def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0

per_head_j = []
exact = 0; total = 0
budget_mismatch = 0
worst = []
for L in range(n_layers):
    ek = eng["keep"][str(L)]
    kk = captured.get(L)
    if kk is None:
        print(f"  layer {L}: kvpress captured nothing (no compress?)"); continue
    for h in range(n_kv):
        e = ek[str(h)]; k = kk[h]
        total += 1
        if len(e) != len(k):
            budget_mismatch += 1
        j = jaccard(e, k)
        per_head_j.append(j)
        if e == k:
            exact += 1
        else:
            worst.append((j, L, h, len(e), len(k)))

per_head_j.sort()
worst.sort()
mean_j = sum(per_head_j) / len(per_head_j)
print(f"=== real-forward keep-set parity (dtype={args.dtype}, cr={args.cr}, seq_len={seq_len}) ===")
print(f"  per-(layer,head) pairs: {total} | exact-match: {exact} ({100*exact/total:.0f}%) | budget-count mismatch: {budget_mismatch}")
print(f"  Jaccard: mean={mean_j:.4f} min={per_head_j[0]:.4f} p10={per_head_j[len(per_head_j)//10]:.4f} median={per_head_j[len(per_head_j)//2]:.4f}")
print(f"  worst 5 (layer,head): " + ", ".join(f"L{L}h{h}:J={j:.3f}({le}vs{lk})" for j,L,h,le,lk in worst[:5]))

if args.out:
    json.dump({
        "dtype": args.dtype, "cr": args.cr, "seq_len": seq_len,
        "n_pairs": total, "exact_match": exact, "exact_match_frac": exact / total,
        "budget_count_mismatch": budget_mismatch,
        "jaccard_mean": mean_j, "jaccard_min": per_head_j[0],
        "jaccard_median": per_head_j[len(per_head_j)//2],
    }, open(args.out, "w"), indent=2)
