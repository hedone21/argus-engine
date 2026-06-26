#!/usr/bin/env python3
"""Diff two generated token streams (argus pyramidkv vs kvpress PyramidKVPress).

Each input is a JSON with a `generated_token_ids` list (the format `run_kvpress.py` writes;
the argus side must emit the same — see ../E2E.md for how to dump argus token IDs). Falls back
to comparing `generated_text` when token IDs are absent. Exit 0 iff the streams are identical
(the acceptance criterion: "출력 토큰이 동일하면 돼"), else 1 with the first divergence.
"""

import argparse
import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="JSON A (e.g. argus)")
    ap.add_argument("b", help="JSON B (e.g. kvpress)")
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    ta, tb = a.get("generated_token_ids"), b.get("generated_token_ids")

    if ta is not None and tb is not None:
        n = min(len(ta), len(tb))
        div = next((i for i in range(n) if ta[i] != tb[i]), None)
        if div is None and len(ta) == len(tb):
            print(f"IDENTICAL: {len(ta)} token IDs match exactly.")
            return 0
        if div is None:
            print(f"PREFIX MATCH then length differs: {n} match; "
                  f"len(a)={len(ta)} len(b)={len(tb)}.")
            return 1
        print(f"DIVERGE at token {div}/{n}: a={ta[div]} b={tb[div]}")
        print(f"  matched prefix: {ta[:div]}")
        return 1

    # text fallback
    sa, sb = a.get("generated_text", ""), b.get("generated_text", "")
    if sa == sb:
        print(f"IDENTICAL TEXT ({len(sa)} chars).")
        return 0
    cp = next((i for i in range(min(len(sa), len(sb))) if sa[i] != sb[i]), min(len(sa), len(sb)))
    print(f"TEXT DIVERGES at char {cp}:\n  a: {sa[:cp]}|{sa[cp:cp+40]}\n  b: {sb[:cp]}|{sb[cp:cp+40]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
