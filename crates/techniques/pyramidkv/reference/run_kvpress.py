#!/usr/bin/env python3
"""Run NVIDIA kvpress `PyramidKVPress` on a HF model and dump the generated token IDs.

This is the REFERENCE side of the end-to-end token-parity check (see ../E2E.md). It greedily
generates `--max-new-tokens` from `--prompt` with PyramidKV compression applied to the prompt
cache, and writes a JSON of {prompt_token_ids, generated_token_ids, generated_text} that
`compare_tokens.py` diffs against the argus-engine run.

Requires a torch-capable Python (<= 3.12 for current torch wheels) + a CUDA GPU:
    python3 -m venv .venv && . .venv/bin/activate
    pip install "torch" "transformers>=4.44" "kvpress" accelerate
    # kvpress: https://github.com/NVIDIA/kvpress  (pin the version you cross-checked the
    # get_layer_budget arithmetic against — see ../reference/README.md).

The argus `pyramidkv` knobs map 1:1 to PyramidKVPress:
    --set compression_ratio=R  ->  compression_ratio=R
    --set window_size=W        ->  window_size=W
    --set kernel_size=K        ->  kernel_size=K
    --set beta=B               ->  beta=B
Use the SAME model weights on both sides (the argus side loads the GGUF/safetensors export of
this exact HF checkpoint) and greedy decoding (do_sample=False) so the token streams are
comparable.
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress import PyramidKVPress


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--compression-ratio", type=float, required=True)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--kernel-size", type=int, default=5)
    ap.add_argument("--beta", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--attn", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=getattr(torch, args.dtype),
            attn_implementation=args.attn,
        )
        .to(args.device)
        .eval()
    )

    press = PyramidKVPress(
        compression_ratio=args.compression_ratio,
        window_size=args.window_size,
        kernel_size=args.kernel_size,
        beta=args.beta,
    )

    input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)
    assert input_ids.shape[1] > args.window_size, (
        f"prompt length {input_ids.shape[1]} must exceed window_size {args.window_size}"
    )

    with torch.no_grad(), press(model):
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    gen = out[0, input_ids.shape[1]:].tolist()

    result = {
        "source": "kvpress.PyramidKVPress",
        "model": args.model,
        "prompt": args.prompt,
        "compression_ratio": args.compression_ratio,
        "window_size": args.window_size,
        "kernel_size": args.kernel_size,
        "beta": args.beta,
        "dtype": args.dtype,
        "attn": args.attn,
        "prompt_token_ids": input_ids[0].tolist(),
        "generated_token_ids": gen,
        "generated_text": tok.decode(gen),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[kvpress] wrote {len(gen)} generated tokens -> {args.out}")
    print(f"[kvpress] text: {result['generated_text']!r}")


if __name__ == "__main__":
    main()
