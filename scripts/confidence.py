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

from siliconmind.agent import CACHE, CONFIRM_SEGMENTS, ROOT, _unit_config
from siliconmind.stats import paired_bootstrap


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
    rows = []
    for f in sorted(glob.glob(str(CACHE / "*.json"))):           # every design confirmed by anyone: agent, baselines, sweep
        res = json.loads(Path(f).read_text())
        c = res["config"]
        if res["segments"] != CONFIRM_SEGMENTS or c.get("p") is None:
            continue
        rows.append({"p": c["p"], "q": c.get("q"), "mode": c["mode"],
                     **paired_bootstrap(res["segment_nll"], ref["segment_nll"], ref["segment_tokens"], a.budget,
                                        a.resamples)})
    out = ROOT / "results/exhaustive/confidence.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}: {len(rows)} designs")


if __name__ == "__main__":
    main()
