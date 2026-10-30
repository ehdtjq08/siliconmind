"""Capture real NVFP4 operand streams from Llama-3.2-1B for gate-level power simulation.

Hooks record the input of a few decoder Linears on one WikiText-2 segment. Random (token, output row) pairs are then
turned into the block sequence the dot-product unit would see for that output, one block per clock: the E2M1 codes of
both operands, packed as in rtl/fp4_block_dot.v, and the two E4M3 block scales.

    PYTHONPATH=. python scripts/capture_operands.py --model models/Llama-3.2-1B
"""
import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from siliconmind.numerics import BLOCK, e2m1_code, e4m3_code, quantize_nvfp4

LAYERS = ["model.layers.0.self_attn.q_proj", "model.layers.4.self_attn.v_proj", "model.layers.7.self_attn.o_proj",
          "model.layers.9.mlp.gate_proj", "model.layers.12.mlp.up_proj", "model.layers.15.mlp.down_proj"]


def pack(codes):
    return f"{sum(c << (4 * i) for i, c in enumerate(codes)):016x}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Llama-3.2-1B")
    ap.add_argument("--segment", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=8, help="(token, row) dot products per layer")
    ap.add_argument("--out", default="data/llama_block_stream.json")
    a = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    seg = json.loads(Path("data/wikitext2_seg1024.json").read_text())[a.segment]
    inputs = {}
    hooks = [model.get_submodule(n).register_forward_hook(lambda m, i, o, n=n: inputs.__setitem__(n, i[0][0]))
             for n in LAYERS]
    with torch.no_grad():
        model(torch.tensor([seg]))
    for h in hooks:
        h.remove()

    rng = random.Random(0)
    streams = []
    for n in LAYERS:
        qx, sx, _ = quantize_nvfp4(inputs[n])
        qw, sw, _ = quantize_nvfp4(model.get_submodule(n).weight.data)
        for _ in range(a.pairs):
            t, r = rng.randrange(qx.shape[0]), rng.randrange(qw.shape[0])
            blocks = [[pack([e2m1_code(v) for v in qx[t, k * BLOCK:(k + 1) * BLOCK].tolist()]),
                       pack([e2m1_code(v) for v in qw[r, k * BLOCK:(k + 1) * BLOCK].tolist()]),
                       e4m3_code(sx[t, k].item()), e4m3_code(sw[r, k].item())] for k in range(sx.shape[1])]
            streams.append({"layer": n, "token": t, "row": r, "blocks": blocks})
    Path(a.out).write_text(json.dumps({"model": Path(a.model).name, "segment": a.segment, "streams": streams}))
    print(f"wrote {a.out}: {len(streams)} dot products, {sum(len(s['blocks']) for s in streams)} blocks")


if __name__ == "__main__":
    main()
