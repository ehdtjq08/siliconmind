"""Per-segment WikiText-2 NLL of Llama-3.2-1B under a list of hardware configs.

Results go to results/<tag>/<config_name>.json so configs can be compared pairwise per segment.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from siliconmind.configs import CONFIGS, SPLITS
from siliconmind.llm import UnitConfig, patch_model, ppl, segment_nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--configs", default=",".join(c for c in CONFIGS if "_q" not in c and "promote" not in c))
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--segments", type=int, default=16)
    ap.add_argument("--tag", default="day1")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    unit = UnitConfig(CONFIGS["bf16"])
    patch_model(model, unit)

    ids = tok(Path("data/wikitext2_test.txt").read_text(), return_tensors="pt").input_ids[0]
    segs = [ids[i * args.seq:(i + 1) * args.seq] for i in range(min(args.segments, len(ids) // args.seq))]

    out_dir = Path("results") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.configs.split(","):
        out = out_dir / f"{name}.json"
        if out.exists():
            print(f"skip {name}")
            continue
        if name in SPLITS:
            unit.default, unit.per_layer = SPLITS[name]
        else:
            unit.default, unit.per_layer = CONFIGS[name], {}
        t0 = time.time()
        rows = [segment_nll(model, s.unsqueeze(0).cuda()) for s in segs]
        nll, n = sum(r[0] for r in rows), sum(r[1] for r in rows)
        res = {"config": SPLITS.get(name, CONFIGS.get(name)), "seq": args.seq, "segment_nll": [r[0] for r in rows],
               "segment_tokens": [r[1] for r in rows], "ppl": ppl(nll, n), "seconds": time.time() - t0}
        out.write_text(json.dumps(res))
        print(f"{name:12s} ppl={res['ppl']:.4f}  ({res['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
