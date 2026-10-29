"""Evaluate accumulator configs in parallel on Modal GPUs.

    ~/.venvs/modal/bin/modal run modal_app.py --configs p8_rne,p6_rne --tag modal_test

Model weights live in the Modal volume "siliconmind-models" (uploaded once with `modal volume put`),
so no Hugging Face token leaves this machine. Results are written to results/<tag>/ locally.
"""
import json
from pathlib import Path

import modal

app = modal.App("siliconmind")
models = modal.Volume.from_name("siliconmind-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", "transformers==5.14.1", "safetensors", "numpy")
    .add_local_python_source("siliconmind")
)

MODEL_DIR = "/models/Llama-3.2-1B"


@app.function(image=image, gpu="H100", volumes={"/models": models}, timeout=3600)
def eval_config(name: str, cfg, per_layer: dict, segments: list[list[int]]):
    import time

    import torch
    from transformers import AutoModelForCausalLM

    from siliconmind.llm import UnitConfig, patch_model, segment_nll

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).cuda().eval()
    unit = UnitConfig(cfg)
    unit.per_layer = per_layer
    patch_model(model, unit)
    t0 = time.time()
    rows = [segment_nll(model, torch.tensor([s], device="cuda")) for s in segments]
    return {"name": name, "rows": rows, "seconds": time.time() - t0}


@app.local_entrypoint()
def main(configs: str, tag: str = "modal", segments: int = 16, seq: int = 1024, shards: int = 4):
    import math

    from siliconmind.configs import CONFIGS, SPLITS

    segs = json.loads(Path(f"data/wikitext2_seg{seq}.json").read_text())[:segments]

    jobs = []
    for name in configs.split(","):
        default, per_layer = SPLITS[name] if name in SPLITS else (CONFIGS[name], {})
        for k in range(shards):                                   # shard segments across containers
            jobs.append((f"{name}#{k}", default, per_layer, segs[k::shards]))

    out = {}
    for r in eval_config.starmap(jobs):
        name, k = r["name"].split("#")
        out.setdefault(name, {})[int(k)] = r
    out_dir = Path("results") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, shards_res in out.items():
        rows = [None] * segments
        for k, r in shards_res.items():                           # undo the k::shards interleave
            for j, row in enumerate(r["rows"]):
                rows[k + j * shards] = row
        nll, n = sum(r[0] for r in rows), sum(r[1] for r in rows)
        default, per_layer = SPLITS[name] if name in SPLITS else (CONFIGS[name], {})
        res = {"config": SPLITS.get(name, CONFIGS.get(name)), "seq": seq, "segment_nll": [r[0] for r in rows],
               "segment_tokens": [r[1] for r in rows], "ppl": math.exp(nll / n),
               "seconds": max(r["seconds"] for r in shards_res.values())}
        (out_dir / f"{name}.json").write_text(json.dumps(res))
        print(f"{name:22s} ppl={res['ppl']:.4f}  ({res['seconds']:.0f}s wall)")
