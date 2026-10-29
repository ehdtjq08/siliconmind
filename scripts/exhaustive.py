"""Ground truth for the agent benchmark: every (P, Q, mode) in the agent's design space.

Synthesis is cheap, so all configs are synthesized (in parallel on CPU). Accuracy is then measured on the
held-out confirmation segments in increasing accumulator area, up to --max-area: any config larger than a
confirmed in-budget one cannot be the optimum. The cheapest in-budget config is the true answer to the agent's
task, and every evaluated point goes into the reference Pareto front for the demo.

    PYTHONPATH=. python scripts/exhaustive.py --synth-only
    PYTHONPATH=. python scripts/exhaustive.py --max-area 14000
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from siliconmind.agent import ROOT, SPACE, evaluate_accuracy, synthesize_area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-only", action="store_true")
    ap.add_argument("--max-area", type=int, default=14000)
    ap.add_argument("--synth-jobs", type=int, default=32)
    ap.add_argument("--eval-jobs", type=int, default=3)     # each confirm eval is split over both GPU workers
    ap.add_argument("--also", default="24:x:rne,24:x:rz,12:x:rne,12:x:rz",
                    help="extra P:Q:mode points to confirm regardless of area (demo slider points)")
    a = ap.parse_args()

    with ThreadPoolExecutor(a.synth_jobs) as ex:
        areas = list(ex.map(lambda c: synthesize_area(*c), SPACE))
    out = ROOT / "results/exhaustive"
    out.mkdir(parents=True, exist_ok=True)
    (out / "areas.json").write_text(json.dumps(areas, indent=1))
    also = {(int(p), 19 if q == "x" else int(q), m) for p, q, m in (s.split(":") for s in a.also.split(",") if s)}
    todo = sorted((r for r in areas if r["accumulator_transistors"] <= a.max_area or (r["p"], r["q"], r["mode"]) in also),
                  key=lambda r: r["accumulator_transistors"])
    print(f"{len(areas)} configs synthesized, {len(todo)} at or below {a.max_area} accumulator transistors")
    if a.synth_only:
        return

    def run(r):
        q = None if r["q"] == 19 else r["q"]
        res = {**evaluate_accuracy(r["p"], q, r["mode"], confirm=True), **r, "q": q}
        print(f"P{res['p']:2d} Q{q or 'x':>2} {res['mode']:3s} acc={res['accumulator_transistors']:6d} "
              f"dppl={res['delta_ppl_percent']:+.2f}%", flush=True)
        return res

    with ThreadPoolExecutor(a.eval_jobs) as ex:
        rows = list(ex.map(run, todo))
    (out / "confirm.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
