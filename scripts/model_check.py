"""Does the design chosen on Llama-3.2-1B hold on a bigger model? Re-measure a few designs on another Llama.

The other model's workers take jobs from their own directory on the worker box, and results are cached separately:

    CUDA_VISIBLE_DEVICES=0 python scripts/eval_worker.py --model models/Llama-3.2-3B --jobs jobs_3b   # one per GPU
    PYTHONPATH=. python scripts/model_check.py --model Llama-3.2-3B --jobs jobs_3b
"""
import argparse
import json

from siliconmind.agent import CONFIRM_SEGMENTS, ROOT, _run_nll, _unit_config, synthesize_area
from siliconmind.stats import paired_bootstrap

# full size, the chosen design, the agent's other pick, a borderline P8 design, the best baselines
DESIGNS = "24:x:rne,9:7:rne,8:8:rne,8:5:rne,10:8:rne,12:7:rz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Llama-3.2-3B")
    ap.add_argument("--jobs", default="jobs_3b")
    ap.add_argument("--designs", default=DESIGNS)
    ap.add_argument("--budget", type=float, default=1.0)
    a = ap.parse_args()

    cache = ROOT / "results/model_check" / a.model / "cache"
    run = lambda cfg: _run_nll(cfg, CONFIRM_SEGMENTS, jobs=a.jobs, cache=cache)
    ref = run(_unit_config(None, None, None))
    rows = []
    for s in a.designs.split(","):
        p, q, mode = s.split(":")
        p, q = int(p), None if q == "x" else int(q)
        res = run(_unit_config(p, q, mode))
        row = {"p": p, "q": q, "mode": mode, "segments": len(CONFIRM_SEGMENTS),
               "accumulator_transistors": synthesize_area(p, q, mode)["accumulator_transistors"],
               **paired_bootstrap(res["segment_nll"], ref["segment_nll"], ref["segment_tokens"], a.budget)}
        rows.append(row)
        print(f"P{p:>2} Q{q or 'x':>2} {mode:3s} acc={row['accumulator_transistors']:6d} "
              f"dppl={row['delta_ppl_percent']:+.2f}% ci=[{row['ci95'][0]:+.2f}, {row['ci95'][1]:+.2f}]", flush=True)
        out = ROOT / "results/model_check" / f"{a.model}.json"
        out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
