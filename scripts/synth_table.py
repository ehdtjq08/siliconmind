"""Summarize results/synth/*.log (Yosys stat -tech cmos) per module."""
import re
import sys
from pathlib import Path

rows = []
for f in Path("results/synth").glob("p*_q*_rne*.log"):
    p, q, r = map(int, re.match(r"p(\d+)_q(\d+)_rne(\d)", f.stem).groups())
    mods = {}
    for name, body in re.findall(r"=== (\S+) ===(.*?)(?====|\Z)", f.read_text(), re.S):
        m = re.search(r"Estimated number of transistors:\s+(\d+)", body)
        if m:
            mods[name] = int(m.group(1))
    prod = next(v for k, v in mods.items() if "fp4_block_prod" in k)
    acc = next(v for k, v in mods.items() if "fp_acc_add" in k)
    rows.append((p, q, r, prod, acc))
for p, q, r, prod, acc in sorted(rows):
    print(f"P={p:2d} Q={q:2d} {'RNE' if r else 'RZ '}  prod={prod:6d}  acc_add={acc:6d}  total={prod + acc:6d}")
