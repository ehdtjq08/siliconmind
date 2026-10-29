"""Precompute the demo slider: chat answers from Llama-3.2-1B-Instruct running on each accumulator design.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/gen_demo.py --points 24:x:rne,10:10:rne,6:6:rne

A point is P:Q:mode (Q = x for an exact block product, P = x for exact accumulation). Each design point
writes results/demo/gen/<point>.json and is skipped if that file exists.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from siliconmind.llm import UnitConfig, patch_model

PROMPTS = [
    "Explain why the sky is blue in two sentences.",
    "What is 17 times 23? Show your work.",
    "Write a haiku about a tiny computer chip.",
    "Give me three tips for staying focused while studying.",
    "Translate into French: The cat is sleeping on the warm keyboard.",
]


def parse(point):
    p, q, mode = point.split(":")
    if p == "x":
        return {"kind": "nvfp4", "p": None}
    return {"kind": "nvfp4", "p": int(p), "q": None if q == "x" else int(q), "mode": mode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Llama-3.2-1B-Instruct")
    ap.add_argument("--points", required=True)
    ap.add_argument("--max-new", type=int, default=96)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    unit = UnitConfig({"kind": "bf16"})
    patch_model(model, unit)
    out_dir = Path("results/demo/gen")
    out_dir.mkdir(parents=True, exist_ok=True)

    for point in a.points.split(","):
        f = out_dir / f"{point.replace(':', '_')}.json"
        if f.exists():
            continue
        unit.default = {"kind": "bf16"} if point == "bf16" else parse(point)
        rows, t0 = [], time.time()
        for prompt in PROMPTS:
            ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=a.max_new, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            rows.append({"prompt": prompt, "answer": tok.decode(gen[0, ids["input_ids"].shape[1]:],
                                                                skip_special_tokens=True)})
        f.write_text(json.dumps({"point": point, "config": unit.default, "rows": rows,
                                 "seconds": time.time() - t0}, indent=1))
        print(f"{point:12s} {time.time() - t0:5.0f}s  {rows[1]['answer'][:70]!r}", flush=True)


if __name__ == "__main__":
    main()
