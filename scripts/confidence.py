"""Paired bootstrap confidence intervals for every confirmed design's perplexity change.

Each design and the exact-accumulation reference were run on the same 32 held-out segments, so segments are resampled
together (paired) and the perplexity change is recomputed per resample. Reads only the local evaluation cache; nothing
is sent to the GPU workers.

    PYTHONPATH=. python scripts/confidence.py
"""
import argparse
import glob
import json
import uuid
from pathlib import Path

import numpy as np

from siliconmind.agent import CACHE, CONFIRM_SEGMENTS, ROOT, _unit_config


def cached_nll(config, segments):
    key = json.dumps([config, segments], sort_keys=True)
    cf = CACHE / (uuid.uuid5(uuid.NAMESPACE_OID, key).hex + ".json")
    return json.loads(cf.read_text()) if cf.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--budget", type=float, default=1.0)
    a = ap.parse_args()

    ref = cached_nll(_unit_config(None, None, None), CONFIRM_SEGMENTS)
    ref_nll, tok = np.array(ref["segment_nll"]), np.array(ref["segment_tokens"])
    idx = np.random.default_rng(0).integers(0, len(tok), (a.resamples, len(tok)))
    rows = []
    for f in sorted(glob.glob(str(CACHE / "*.json"))):           # every design confirmed by anyone: agent, baselines, sweep
        res = json.loads(Path(f).read_text())
        c = res["config"]
        if res["segments"] != CONFIRM_SEGMENTS or c.get("p") is None:
            continue
        diff = np.array(res["segment_nll"]) - ref_nll
        boot = 100 * np.expm1(diff[idx].sum(1) / tok[idx].sum(1))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"p": c["p"], "q": c.get("q"), "mode": c["mode"],
                     "delta_ppl_percent": 100 * float(np.expm1(diff.sum() / tok.sum())),
                     "ci95": [float(lo), float(hi)], "p_within_budget": float((boot <= a.budget).mean())})
    out = ROOT / "results/exhaustive/confidence.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}: {len(rows)} designs")


if __name__ == "__main__":
    main()
