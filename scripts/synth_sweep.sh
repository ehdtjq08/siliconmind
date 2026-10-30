#!/bin/bash
# Synthesize fp4_block_dot for one (P, Q, RNE) and report Yosys CMOS transistor estimates per module.
# Parameters go in as Verilog literals through a wrapper, as in src/project.v. Yosys chparam would make them unsigned,
# which turns the signed exponent arithmetic in fp_acc_add unsigned and synthesizes a different circuit.
P=$1; Q=$2; RNE=$3; OUT=results/synth/p${P}_q${Q}_rne${RNE}.log
TOP=$(mktemp --suffix=.v); trap 'rm -f $TOP' EXIT
cat > $TOP <<V
module synth_top (input wire [63:0] a, input wire [63:0] b, input wire [7:0] sa, input wire [7:0] sb,
    input wire acc_s, input wire signed [9:0] acc_e, input wire [$((P - 1)):0] acc_m,
    output wire out_s, output wire signed [9:0] out_e, output wire [$((P - 1)):0] out_m);
    fp4_block_dot #(.P($P), .Q($Q), .RNE($RNE)) u (.a(a), .b(b), .sa(sa), .sb(sb), .acc_s(acc_s), .acc_e(acc_e),
        .acc_m(acc_m), .out_s(out_s), .out_e(out_e), .out_m(out_m));
endmodule
V
yosys -q -p "read_verilog -sv rtl/fp4_block_dot.v $TOP; synth -top synth_top; abc -g cmos2; opt_clean; \
  tee -o $OUT stat -tech cmos" >/dev/null 2>&1
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
