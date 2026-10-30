"""Collect everything the demo page shows into demo/data.js (window.DATA = {...}).

Missing inputs (generations, baselines, exhaustive results not run yet) are exported as empty, so the page can
be built and checked while experiments are still running.
"""
import glob
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCREEN_SEGMENTS = list(range(0, 4))          # mirrors siliconmind/agent.py without importing it
CONFIRM_SEGMENTS = list(range(16, 48))


def load(path, default=None):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else default


def agent_events(path):
    """Agent transcript as a flat event list: what it said, what it called, what came back."""
    rows = [json.loads(l) for l in open(path)]
    t0 = rows[0]["t"]
    ev = []
    for r in rows:
        if r["kind"] == "assistant":
            if r.get("content"):
                ev.append({"t": r["t"] - t0, "kind": "say", "text": r["content"]})
        else:
            res = r["result"]
            keep = ("delta_ppl_percent", "segments", "accumulator_transistors", "mismatches", "vectors", "error")
            ev.append({"t": r["t"] - t0, "kind": "tool", "name": r["name"], "args": r["args"],
                       "result": {k: res[k] for k in keep if k in res}})
    return ev


def cached_deltas(segments):
    """Delta ppl vs exact accumulation for every config the workers evaluated on exactly these segments."""
    runs = [json.loads(Path(f).read_text()) for f in glob.glob(str(ROOT / "results/agent_cache/*.json"))]
    runs = [r for r in runs if r["segments"] == segments]
    ref = next((r for r in runs if r["config"].get("p") is None), None)
    if ref is None:
        return []
    tok, ref_nll = sum(ref["segment_tokens"]), sum(ref["segment_nll"])
    return [{"p": r["config"]["p"], "q": r["config"].get("q"), "mode": r["config"]["mode"],
             "ppl": math.exp(sum(r["segment_nll"]) / tok),
             "delta_ppl_percent": 100 * (math.exp((sum(r["segment_nll"]) - ref_nll) / tok) - 1)}
            for r in runs if r["config"].get("p") is not None]


def main():
    areas = load("results/exhaustive/areas.json", [])
    block = statistics.median(r["block_product_transistors"] for r in areas) if areas else None
    runs = sorted(glob.glob(str(ROOT / "results/agent_runs/*.jsonl")))
    data = {
        "block_product_transistors": block,
        "areas": [{k: r[k] for k in ("p", "q", "mode", "accumulator_transistors")} for r in areas],
        "exhaustive_done": (ROOT / "results/exhaustive/confirm.json").exists(),
        "exhaustive_max_area": 14000,
        "ci": [{k: r[k] for k in ("p", "q", "mode", "ci95", "p_within_budget")}
               for r in load("results/exhaustive/confidence.json", [])],
        "sky130": [{k: r[k] for k in ("p", "q", "mode", "accumulator_um2")}
                   for r in load("results/exhaustive/areas_sky130.json", [])],
        "bigger_model": {Path(f).stem: load(Path(f).relative_to(ROOT))
                         for f in sorted(glob.glob(str(ROOT / "results/model_check/*.json")))},
        "timing": [{k: r[k] for k in ("p", "q", "mode", "acc_loop_ns")} for r in load("results/ppa/timing.json", [])],
        "power": load("results/ppa/power.json", []),
        "confirm": cached_deltas(CONFIRM_SEGMENTS),
        "screen": cached_deltas(SCREEN_SEGMENTS),
        "agent_runs": [{"name": Path(f).stem, "events": agent_events(f)} for f in runs],
        "baselines": [load(Path(f).relative_to(ROOT)) for f in sorted(glob.glob(str(ROOT / "results/baselines/*.json")))],
        "generations": [load(Path(f).relative_to(ROOT)) for f in sorted(glob.glob(str(ROOT / "results/demo/gen/*.json")))],
    }
    out = ROOT / "demo/data.js"
    out.parent.mkdir(exist_ok=True)
    out.write_text("window.DATA = " + json.dumps(data) + ";\n")
    print(f"wrote {out}: {len(data['areas'])} areas, {len(data['confirm'])} confirmed, {len(data['screen'])} screened, "
          f"{len(data['agent_runs'])} agent runs, {len(data['baselines'])} baselines, "
          f"{len(data['generations'])} generation sets")


if __name__ == "__main__":
    main()
