#!/bin/bash
# Synthesize fp4_block_dot for one (P, Q, RNE) and report Yosys CMOS transistor estimates per module.
P=$1; Q=$2; RNE=$3; OUT=results/synth/p${P}_q${Q}_rne${RNE}.log
yosys -q -p "read_verilog -sv rtl/fp4_block_dot.v; chparam -set P $P -set Q $Q -set RNE $RNE fp4_block_dot; \
  synth -top fp4_block_dot; abc -g cmos2; opt_clean; tee -o $OUT stat -tech cmos" >/dev/null 2>&1
python - "$OUT" "$P" "$Q" "$RNE" <<'PY'
import re, sys
txt = open(sys.argv[1]).read()
mods = {}
for name, body in re.findall(r"=== (\S+) ===(.*?)(?====|\Z)", txt, re.S):
    m = re.search(r"Estimated number of transistors:\s+(\d+)", body)
    if m: mods[name] = int(m.group(1))
prod = next(v for k, v in mods.items() if "prod" in k)
acc = next(v for k, v in mods.items() if "acc_add" in k)
print(f"P={sys.argv[2]:>2} Q={sys.argv[3]:>2} RNE={sys.argv[4]}  prod={prod:6d}  acc_add={acc:6d}  total={prod+acc:6d}")
PY
