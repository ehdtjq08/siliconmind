"""Non-agent baselines with the same accuracy-evaluation budget as the agent.

  random  uniform samples from the (P, Q, mode) space
  grid    a coarse grid ordered from cheap to expensive area
  tpe     Bayesian optimization (Optuna TPE): minimize area subject to the accuracy constraint

Each picks the smallest-area config whose screening delta is within budget, then confirms it on held-out
segments (the confirmation counts against the budget, as it does for the agent).
"""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from siliconmind.agent import MODES, P_RANGE, Q_CHOICES, ROOT, SPACE, evaluate_accuracy, synthesize_area


def tpe_candidates(n, tol, seed):
    """Sequential, adaptive: each design is chosen from the results so far. Repeats of an already evaluated design are
    answered from memory and do not count against the budget."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=3,
                                         constraints_func=lambda t: [t.user_attrs["violation"]])
    study = optuna.create_study(direction="minimize", sampler=sampler)
    seen = {}
    while len(seen) < n and len(study.trials) < 1000:
        t = study.ask()
        c = (t.suggest_int("p", P_RANGE.start, P_RANGE.stop - 1), t.suggest_categorical("q", Q_CHOICES),
             t.suggest_categorical("mode", MODES))
        if c not in seen:
            seen[c] = {**evaluate_accuracy(*c), **synthesize_area(*c)}
        t.set_user_attr("violation", seen[c]["delta_ppl_percent"] - tol)
        study.tell(t, seen[c]["accumulator_transistors"])
    return list(seen.values())


def search(kind, budget, tol, seed):
    rng = random.Random(seed)
    if kind == "tpe":
        rows = tpe_candidates(budget - 1, tol, seed)
    elif kind == "random":
        cands = rng.sample(SPACE, budget - 1)
    else:
        grid = [(p, q, m) for p in (6, 8, 10, 12, 16) for q in (None, 8, 12) for m in MODES]
        cands = sorted(grid, key=lambda c: synthesize_area(*c)["accumulator_transistors"])[:budget - 1]
    if kind != "tpe":
        # Candidates are fixed up front (no adaptivity), so evaluating them concurrently changes nothing but wall time.
        with ThreadPoolExecutor(2) as ex:
            rows = list(ex.map(lambda c: {**evaluate_accuracy(*c), **synthesize_area(*c)}, cands))
    ok = [r for r in rows if r["delta_ppl_percent"] <= tol]
    best = min(ok, key=lambda r: r["accumulator_transistors"]) if ok else None
    confirmed = evaluate_accuracy(best["p"], best["q"], best["mode"], confirm=True) if best else None
    return {"kind": kind, "budget": budget, "evaluated": rows, "chosen": best, "confirmed": confirmed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["random", "grid", "tpe"], default="random")
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    res = search(a.kind, a.budget, a.tol, a.seed)
    out = ROOT / f"results/baselines/{a.kind}_b{a.budget}_s{a.seed}_{time.strftime('%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("chosen", "confirmed")}, indent=1))


if __name__ == "__main__":
    main()
