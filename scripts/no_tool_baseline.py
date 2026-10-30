"""Ablation: Muse Spark without tools. Same task and design space, but it must commit to a design from its own
knowledge in one reply. Its pick is then confirmed on the held-out segments (one accuracy evaluation).

    PYTHONPATH=. python scripts/no_tool_baseline.py --runs 4
"""
import argparse
import json
import re
import time

from siliconmind.agent import ROOT, SYSTEM, chat, evaluate_accuracy, synthesize_area

PROMPT = SYSTEM.split("Budget:")[0] + """You have no tools in this session: no accuracy evaluation, no synthesis and no
RTL verification. Choose the design from what you know. P is 5-24, Q is 5-18 or null for exact, mode is rne or rz.
End your reply with one line of JSON: {"p": <int>, "q": <int or null>, "mode": "rne" | "rz"}"""


def ask(reasoning_effort):
    resp = chat([{"role": "system", "content": PROMPT}, {"role": "user", "content": "Choose the design."}],
                reasoning_effort=reasoning_effort)
    text = resp["choices"][0]["message"]["content"]
    pick = json.loads(re.findall(r"\{[^{}]*\"p\"[^{}]*\}", text)[-1])
    return text, pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--effort", default="medium")
    a = ap.parse_args()
    for i in range(a.runs):
        text, pick = ask(a.effort)
        p, q, mode = pick["p"], pick.get("q"), pick["mode"]
        chosen = {"p": p, "q": q, "mode": mode, **synthesize_area(p, q, mode)}
        res = {"kind": "notools", "budget": 1, "evaluated": [], "chosen": chosen, "reply": text,
               "confirmed": evaluate_accuracy(p, q, mode, confirm=True)}
        out = ROOT / f"results/baselines/notools_s{i}_{time.strftime('%H%M%S')}.json"
        out.write_text(json.dumps(res, indent=1))
        print(f"run {i}: P{p} Q{q} {mode} acc={chosen['accumulator_transistors']} "
              f"confirmed {res['confirmed']['delta_ppl_percent']:+.2f}%", flush=True)


if __name__ == "__main__":
    main()
