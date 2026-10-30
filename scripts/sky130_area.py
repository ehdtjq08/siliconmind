"""Cross-check the area metric: synthesize every design in the space onto sky130 standard cells.

The agent and the benchmark rank designs by the Yosys CMOS transistor estimate. This maps the same RTL onto
sky130_fd_sc_hd cells (the library the chosen tile is laid out in) and reports cell area per module, so the ranking can
be checked in real silicon area.

    SKY130_LIB=.../sky130_fd_sc_hd__tt_025C_1v80.lib PYTHONPATH=. python scripts/sky130_area.py
"""
import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

from siliconmind.agent import ROOT, SPACE


def sky130_area(p, q=None, mode="rne"):
    q = q or 19
    rne = int(mode == "rne")
    log = ROOT / f"results/synth_sky130/p{p}_q{q}_rne{rne}.log"
    if not log.exists():
        subprocess.run([str(ROOT / "scripts/synth_sky130.sh"), str(p), str(q), str(rne)], cwd=ROOT,
                       capture_output=True, check=True)
    mods = dict(re.findall(r"Chip area for module '\S*?\\?(fp4_block_prod|fp_acc_add)\S*': ([\d.]+)", log.read_text()))
    return {"p": p, "q": q, "mode": mode, "block_product_um2": float(mods["fp4_block_prod"]),
            "accumulator_um2": float(mods["fp_acc_add"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=64)
    a = ap.parse_args()
    with ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(lambda c: sky130_area(*c), SPACE))
    out = ROOT / "results/exhaustive/areas_sky130.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}: {len(rows)} designs")


if __name__ == "__main__":
    main()
