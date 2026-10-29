"""Paired comparison of results/<tag>/*.json against a reference config.

Delta is the relative perplexity change exp(mean dNLL/token) - 1 with a segment bootstrap 95% CI.
"""
import argparse
import json
import math
import random
from pathlib import Path


def load(tag):
    return {f.stem: json.loads(f.read_text()) for f in sorted(Path("results", tag).glob("*.json"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="day1")
    ap.add_argument("--ref", default="nvfp4_exact")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    res = load(args.tag)
    ref = res[args.ref]
    rng = random.Random(0)
    n = len(ref["segment_nll"])
    print(f"{'config':22s} {'ppl':>9s} {'vs ' + args.ref:>14s}   95% CI")
    for name, r in sorted(res.items(), key=lambda kv: kv[1]["ppl"]):
        d = [a - b for a, b in zip(r["segment_nll"], ref["segment_nll"])]
        toks = ref["segment_tokens"]
        rel = lambda idx: math.exp(sum(d[i] for i in idx) / sum(toks[i] for i in idx)) - 1
        boots = sorted(rel([rng.randrange(n) for _ in range(n)]) for _ in range(args.boot))
        lo, hi = boots[int(0.025 * args.boot)], boots[int(0.975 * args.boot)]
        print(f"{name:22s} {r['ppl']:9.4f} {100 * rel(range(n)):+13.2f}%   [{100 * lo:+.2f}, {100 * hi:+.2f}]")


if __name__ == "__main__":
    main()
