"""GPU worker: keep the emulated model loaded and evaluate configs dropped into a job directory.

A job is jobs/pending/<id>.json = {"config": {...}, "per_layer": {...}, "segments": [i, ...]}.
The worker claims it by renaming into jobs/running/ and writes jobs/done/<id>.json with per-segment NLL.

    CUDA_VISIBLE_DEVICES=0 python scripts/eval_worker.py --model models/Llama-3.2-1B
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from siliconmind.llm import UnitConfig, patch_model, segment_nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Llama-3.2-1B")
    ap.add_argument("--jobs", default="jobs")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--block-chunk", type=int, default=32,
                    help="blocks per emulation step; lower it to fit bigger models (results are identical)")
    args = ap.parse_args()

    jobs = Path(args.jobs)
    for d in ("pending", "running", "done"):
        (jobs / d).mkdir(parents=True, exist_ok=True)
    segs = json.loads(Path(f"data/wikitext2_seg{args.seq}.json").read_text())

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    unit = UnitConfig({"kind": "bf16"})
    patch_model(model, unit, args.block_chunk)
    me = f"{os.uname().nodename}:gpu{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}"
    print(f"worker {me} ready", flush=True)

    while True:
        claimed = None
        for f in sorted((jobs / "pending").glob("*.json")):
            try:
                claimed = f.rename(jobs / "running" / f.name)
                break
            except FileNotFoundError:                     # another worker took it
                continue
        if claimed is None:
            if (jobs / "STOP").exists():                  # graceful shutdown once the queue is drained
                print(f"worker {me} stopping", flush=True)
                return
            time.sleep(0.5)
            continue
        t0 = time.time()
        try:
            job = json.loads(claimed.read_text())
            unit.default, unit.per_layer = job["config"], job.get("per_layer", {})
            rows = [segment_nll(model, torch.tensor([segs[i]], device="cuda")) for i in job["segments"]]
            job.update(segment_nll=[r[0] for r in rows], segment_tokens=[r[1] for r in rows],
                       seconds=time.time() - t0, worker=me)
        except Exception as e:                            # report the failure to the client instead of dying
            job = {"error": f"{type(e).__name__}: {e}", "worker": me, "seconds": time.time() - t0, "config": None}
        tmp = jobs / "done" / (claimed.name + ".tmp")
        tmp.write_text(json.dumps(job))
        tmp.rename(jobs / "done" / claimed.name)
        claimed.unlink()
        torch.cuda.empty_cache()                          # don't sit on ~18 GB of cached blocks while idle
        print(f"{claimed.stem} {job['config']} {job['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
