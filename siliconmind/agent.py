"""Muse Spark design agent: explores NVFP4 accumulator configs with real tools.

Tools
  evaluate_accuracy  Llama-3.2-1B perplexity under the emulated unit (GPU workers on the 4090 box)
  synthesize_area    Yosys CMOS transistor estimate of rtl/fp4_block_dot.v for (P, Q, rounding)
  verify_rtl         RTL vs exact reference on random + corner-case vectors
After every round the agent also receives the current Pareto plot as an image.
"""
import base64
import io
import json
import math
import os
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# GPU box running scripts/eval_worker.py: environment, else ~/.config/siliconmind/worker.json {"host", "dir"}
_WORKER_CFG = Path("~/.config/siliconmind/worker.json").expanduser()
_worker = json.loads(_WORKER_CFG.read_text()) if _WORKER_CFG.exists() else {}
WORKER_HOST = os.environ.get("SILICONMIND_WORKER_HOST", _worker.get("host", "localhost"))
WORKER_DIR = os.environ.get("SILICONMIND_WORKER_DIR", _worker.get("dir", str(ROOT)))
SCREEN_SEGMENTS = list(range(0, 4))
CONFIRM_SEGMENTS = list(range(16, 48))       # held out from screening
CACHE = ROOT / "results" / "agent_cache"
P_RANGE = range(5, 25)                       # the design space described to the agent in TOOLS
Q_CHOICES = [None] + list(range(5, 19))
MODES = ["rne", "rz"]
SPACE = [(p, q, m) for p in P_RANGE for q in Q_CHOICES for m in MODES]
API = "https://api.meta.ai/v1/chat/completions"


# ---------------------------------------------------------------- Muse Spark client

def chat(messages, tools=None, model="muse-spark-1.3", reasoning_effort=None, retries=3):
    key = Path("~/.config/meta/api_key").expanduser().read_text().strip()
    body = {"model": model, "messages": messages}
    if tools:
        body["tools"] = tools
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if attempt == retries - 1 or e.code < 500:
                raise RuntimeError(f"Meta API {e.code}: {e.read().decode()[:300]}")
            time.sleep(5 * (attempt + 1))


# ---------------------------------------------------------------- tools

def _ssh(cmd, stdin=None):
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}   # anaconda's OpenSSL breaks ssh
    return subprocess.run(["ssh", "-o", "BatchMode=yes", WORKER_HOST, cmd], input=stdin, text=True,
                          capture_output=True, check=True, env=env).stdout


def _unit_config(p, q, mode):
    return {"kind": "nvfp4", "p": p, "q": q, "mode": mode} if p else {"kind": "nvfp4", "p": None}


def _run_nll(config, segments, workers=2):
    """Per-segment NLL on the worker box, cached locally by (config, segments).

    Long segment lists are split into contiguous chunks so each GPU worker takes one."""
    key = json.dumps([config, segments], sort_keys=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / (uuid.uuid5(uuid.NAMESPACE_OID, key).hex + ".json")
    if cf.exists():
        return json.loads(cf.read_text())
    n = workers if len(segments) >= 8 else 1
    chunks = [segments[k * len(segments) // n:(k + 1) * len(segments) // n] for k in range(n)]
    jids = [uuid.uuid4().hex for _ in chunks]
    for jid, segs in zip(jids, chunks):
        _ssh(f"cat > {WORKER_DIR}/jobs/pending/{jid}.json", stdin=json.dumps({"config": config, "segments": segs}))
    done = {}
    while len(done) < len(jids):
        time.sleep(2)
        for jid in jids:
            if jid not in done:
                out = _ssh(f"cat {WORKER_DIR}/jobs/done/{jid}.json 2>/dev/null || true")
                if out.strip():
                    done[jid] = json.loads(out)
    parts = [done[j] for j in jids]
    res = {"config": config, "segments": segments, "segment_nll": sum((p["segment_nll"] for p in parts), []),
           "segment_tokens": sum((p["segment_tokens"] for p in parts), []),
           "seconds": max(p["seconds"] for p in parts), "workers": [p["worker"] for p in parts]}
    cf.write_text(json.dumps(res))
    return res


def evaluate_accuracy(p, q=None, mode="rne", confirm=False):
    segs = CONFIRM_SEGMENTS if confirm else SCREEN_SEGMENTS
    ref = _run_nll(_unit_config(None, None, None), segs)
    res = _run_nll(_unit_config(p, q, mode), segs)
    tok = sum(ref["segment_tokens"])
    d = math.exp((sum(res["segment_nll"]) - sum(ref["segment_nll"])) / tok) - 1
    return {"p": p, "q": q, "mode": mode, "segments": len(segs), "ppl": math.exp(sum(res["segment_nll"]) / tok),
            "ppl_exact_accumulation": math.exp(sum(ref["segment_nll"]) / tok), "delta_ppl_percent": 100 * d}


def synthesize_area(p, q=None, mode="rne"):
    q = q or 19
    rne = int(mode == "rne")
    log = ROOT / f"results/synth/p{p}_q{q}_rne{rne}.log"
    if not log.exists():
        subprocess.run([str(ROOT / "scripts/synth_sweep.sh"), str(p), str(q), str(rne)], cwd=ROOT,
                       capture_output=True, check=True)
    import re
    mods = {name: int(m.group(1)) for name, body in re.findall(r"=== (\S+) ===(.*?)(?====|\Z)", log.read_text(), re.S)
            if (m := re.search(r"Estimated number of transistors:\s+(\d+)", body))}
    prod = next(v for k, v in mods.items() if "fp4_block_prod" in k)
    acc = next(v for k, v in mods.items() if "fp_acc_add" in k)
    return {"p": p, "q": q, "mode": mode, "block_product_transistors": prod, "accumulator_transistors": acc,
            "total_transistors": prod + acc}


def verify_rtl(p, q=None, mode="rne", n=1000):
    import sys
    sys.path.insert(0, str(ROOT))
    from tests.rtl_check import run
    total, bad = run(p, mode, n, seed=hash((p, q, mode)) % 1000, q=q)
    return {"p": p, "q": q, "mode": mode, "vectors": total, "mismatches": len(bad)}


TOOLS = [
    {"type": "function", "function": {
        "name": "evaluate_accuracy",
        "description": "Run Llama-3.2-1B on WikiText-2 with every decoder Linear computed by the emulated NVFP4 "
                       "dot-product unit (bit-exact to the RTL). Returns perplexity and its % change vs exact "
                       "accumulation. Screening uses 4 segments (noise roughly +/-1.5%); confirm=true uses 32 "
                       "held-out segments and costs 8x more.",
        "parameters": {"type": "object", "properties": {
            "p": {"type": "integer", "description": "accumulator significand bits incl. hidden bit (5-24)"},
            "q": {"type": "integer", "description": "pre-round the 19-bit block product to q bits before the add "
                                                    "(5-18); omit for exact"},
            "mode": {"type": "string", "enum": ["rne", "rz"], "description": "rounding for both roundings"},
            "confirm": {"type": "boolean"}}, "required": ["p", "mode"]}}},
    {"type": "function", "function": {
        "name": "synthesize_area",
        "description": "Synthesize the unit with Yosys (generic CMOS gates) and return transistor estimates for "
                       "the fixed block-product part and for the accumulator adder.",
        "parameters": {"type": "object", "properties": {
            "p": {"type": "integer"}, "q": {"type": "integer"}, "mode": {"type": "string", "enum": ["rne", "rz"]}},
            "required": ["p", "mode"]}}},
    {"type": "function", "function": {
        "name": "verify_rtl",
        "description": "Check the RTL for (p, q, mode) against the exact reference model on 1000 random and "
                       "corner-case vectors. Must report 0 mismatches before a design is final.",
        "parameters": {"type": "object", "properties": {
            "p": {"type": "integer"}, "q": {"type": "integer"}, "mode": {"type": "string", "enum": ["rne", "rz"]}},
            "required": ["p", "mode"]}}},
]
IMPL = {"evaluate_accuracy": evaluate_accuracy, "synthesize_area": synthesize_area, "verify_rtl": verify_rtl}


# ---------------------------------------------------------------- pareto plot

def pareto_png(points):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for pt in points:
        c = "tab:blue" if pt["mode"] == "rne" else "tab:red"
        ax.scatter(pt["accumulator_transistors"], pt["delta_ppl_percent"], c=c, s=30)
        ax.annotate(f"P{pt['p']}/Q{pt['q'] or 'x'}", (pt["accumulator_transistors"], pt["delta_ppl_percent"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(1.0, ls="--", c="gray", lw=0.8)
    ax.set_xlabel("accumulator transistors (Yosys estimate)")
    ax.set_ylabel("perplexity change vs exact (%)")
    ax.set_yscale("symlog", linthresh=2)
    ax.set_title("blue = RNE, red = RZ; dashed = 1% budget")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- loop

SYSTEM = """You are a hardware architect designing the accumulator of an NVFP4 dot-product unit for LLM inference.
Each step the unit computes R = sum of 16 E2M1 products (exact) times two E4M3 block scales, then
acc <- round_P(acc + round_Q(R)). You choose P (accumulator significand bits), Q (pre-rounding of the 19-bit
block product; omit for exact) and the rounding mode (rne or rz).

Goal: the smallest accumulator transistor count whose Llama-3.2-1B perplexity change vs exact accumulation
stays within +1.0%. Screening evaluations use 4 segments and are noisy (about +/-1.5%), so confirm finalists
with confirm=true before deciding. Every final design must pass verify_rtl with 0 mismatches.

Budget: at most {budget} evaluate_accuracy calls (confirm calls count as one). synthesize_area and verify_rtl
are cheap. Think about which knob actually drives area before spending accuracy evaluations. After each round
you will see the current Pareto plot. When done, reply with a short final report: chosen design, its area
saving vs the P=24 RNE exact-Q baseline, its confirmed perplexity change, and what you learned."""


def run_agent(budget=10, log_path=None, reasoning_effort="medium", max_turns=30):
    log_path = Path(log_path or ROOT / f"results/agent_runs/{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")
    messages = [{"role": "system", "content": SYSTEM.format(budget=budget)},
                {"role": "user", "content": "Start the design exploration."}]
    points, areas, used = [], {}, 0

    def record(kind, **kw):
        log.write(json.dumps({"t": time.time(), "kind": kind, **kw}) + "\n")
        log.flush()

    for turn in range(max_turns):
        resp = chat(messages, tools=TOOLS, reasoning_effort=reasoning_effort)
        msg = resp["choices"][0]["message"]
        record("assistant", content=msg.get("content"), tool_calls=msg.get("tool_calls"), usage=resp.get("usage"))
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content"), log_path
        for call in calls:
            name, args = call["function"]["name"], json.loads(call["function"]["arguments"] or "{}")
            if name == "evaluate_accuracy" and used >= budget:
                result = {"error": "accuracy evaluation budget exhausted; finish with what you have"}
            else:
                try:
                    result = IMPL[name](**args)
                except Exception as e:                                   # report tool errors to the agent
                    result = {"error": f"{type(e).__name__}: {e}"}
                if name == "evaluate_accuracy" and "error" not in result:
                    used += 1
                    area = synthesize_area(result["p"], result["q"], result["mode"])
                    points.append({**result, **area})
                if name == "synthesize_area" and "error" not in result:
                    areas[(result["p"], result["q"], result["mode"])] = result
            record("tool", name=name, args=args, result=result)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
        if points:
            messages.append({"role": "user", "content": [
                {"type": "text", "text": f"Current Pareto plot ({used}/{budget} accuracy evaluations used)."},
                {"type": "image_url", "image_url": {"url": pareto_png(points)}}]})
    return None, log_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--effort", default="medium")
    a = ap.parse_args()
    report, path = run_agent(budget=a.budget, reasoning_effort=a.effort)
    print(report)
    print("log:", path)
