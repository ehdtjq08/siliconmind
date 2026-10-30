"""Timing and power on sky130 standard cells, beyond the area metric.

timing  Every design's accumulate stage (fp_acc_add) is synthesized onto sky130_fd_sc_hd and timed with OpenSTA. The
        acc -> adder -> acc loop is the recurrence a pipeline cannot cut, so its delay sets the unit's peak rate.
power   For a few designs, a MAC wrapper (fp4_block_dot plus the accumulator register) is synthesized, the gate-level
        netlist is simulated with Icarus on real Llama-3.2-1B operand streams (data/llama_block_stream.json, from
        scripts/capture_operands.py), and OpenSTA turns the switching activity into power. The gate-level results are
        checked against the RTL on the same streams.

    SKY130_LIB=.../sky130_fd_sc_hd__tt_025C_1v80.lib STA=.../sta PYTHONPATH=. python scripts/ppa.py timing
    SKY130_LIB=... STA=... PYTHONPATH=. python scripts/ppa.py power
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from siliconmind.agent import ROOT, SPACE

LIB = os.environ.get("SKY130_LIB", "")
STA = os.environ.get("STA", "sta")
RTL = ROOT / "rtl/fp4_block_dot.v"
DONT_USE = "-dont_use *__lpflow_* -dont_use *__probe_* -dont_use *__clk* -dont_use *__dly*"
PERIOD_NS = 100                                    # 10 MHz, the Tiny Tapeout clock; energy per step = power * period
POWER_DESIGNS = "24:x:rne,9:7:rne,9:6:rne,8:8:rne,8:6:rne,10:8:rne,12:7:rz"

# Parameters are Verilog literals in the generated source, as in src/project.v: Yosys chparam would make them unsigned
# and synthesize different exponent arithmetic from what Icarus simulates.
MAC = """
module fp4_mac #(parameter P = P_VAL, parameter Q = Q_VAL, parameter RNE = RNE_VAL, parameter EW = 10) (
    input wire clk, input wire clr, input wire [63:0] a, input wire [63:0] b, input wire [7:0] sa, input wire [7:0] sb,
    output reg acc_s, output reg signed [EW-1:0] acc_e, output reg [P-1:0] acc_m);
    wire s; wire signed [EW-1:0] e; wire [P-1:0] m;
    fp4_block_dot #(.P(P), .Q(Q), .RNE(RNE), .EW(EW)) u (.a(a), .b(b), .sa(sa), .sb(sb),
        .acc_s(acc_s), .acc_e(acc_e), .acc_m(acc_m), .out_s(s), .out_e(e), .out_m(m));
    always @(posedge clk) {acc_s, acc_e, acc_m} <= clr ? 0 : {s, e, m};
endmodule
"""

ACC_TOP = """
module acc_top (input wire acc_s, input wire signed [9:0] acc_e, input wire [P_VAL-1:0] acc_m, input wire r_s,
    input wire signed [13:0] r_e, input wire [18:0] r_m, output wire out_s, output wire signed [9:0] out_e,
    output wire [P_VAL-1:0] out_m);
    fp_acc_add #(.P(P_VAL), .Q(Q_VAL), .RNE(RNE_VAL)) u (.acc_s(acc_s), .acc_e(acc_e), .acc_m(acc_m), .r_s(r_s),
        .r_e(r_e), .r_m(r_m), .out_s(out_s), .out_e(out_e), .out_m(out_m));
endmodule
"""

TB = """
`timescale 1ns/1ps
module tb;
    reg clk = 0, clr = 1; reg [63:0] a = 0, b = 0; reg [7:0] sa = 0, sb = 0;
    wire acc_s; wire [9:0] acc_e; wire [{P}-1:0] acc_m;
    fp4_mac dut (.clk(clk), .clr(clr), .a(a), .b(b), .sa(sa), .sb(sb), .acc_s(acc_s), .acc_e(acc_e), .acc_m(acc_m));
    always #{half} clk = ~clk;
    integer fin, fout, n, last;
    reg [63:0] va, vb; reg [7:0] vsa, vsb;
    initial begin
        fin = $fopen("{stim}", "r"); fout = $fopen("{out}", "w");
        if ({vcd}) begin $dumpfile("{vcdfile}"); $dumpvars(2, tb.dut); end
        while (!$feof(fin)) begin
            n = $fscanf(fin, "%h %h %h %h %h\\n", last, va, vb, vsa, vsb);
            if (n == 5) begin
                @(negedge clk); clr = 0; a = va; b = vb; sa = vsa; sb = vsb;
                if (last) begin
                    @(negedge clk); $fwrite(fout, "%h %h %h\\n", acc_s, acc_e, acc_m);
                    clr = 1; a = 0; b = 0; sa = 0; sb = 0;
                end
            end
        end
        @(negedge clk); $fclose(fout); $finish;
    end
endmodule
"""


def run(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, capture_output=True, text=True, **kw).stdout


def parse(s):
    p, q, mode = s.split(":")
    return int(p), (19 if q == "x" else int(q)), mode


def yosys_map(script_head, top, out_v):
    run(["yosys", "-q", "-p", f"{script_head}; synth -top {top} -flatten; dfflibmap -liberty {LIB}; "
                              f"abc -liberty {LIB} {DONT_USE}; opt_clean; write_verilog -noattr {out_v}; "
                              f"tee -o {out_v}.stat stat -liberty {LIB}"])
    return float(re.search(r"Chip area for (?:top )?module '\\?" + top + r"': ([\d.]+)", Path(f"{out_v}.stat").read_text())[1])


def acc_timing(p, q, mode):
    with tempfile.TemporaryDirectory() as d:
        net = f"{d}/acc.v"
        top = Path(f"{d}/acc_top.v")
        top.write_text(ACC_TOP.replace("P_VAL", str(p)).replace("Q_VAL", str(q)).replace("RNE_VAL", str(int(mode == "rne"))))
        area = yosys_map(f"read_verilog -sv {RTL} {top}", "acc_top", net)
        tcl = Path(f"{d}/t.tcl")
        tcl.write_text(f"read_liberty {LIB}\nread_verilog {net}\nlink_design acc_top\n"
                       f"create_clock -name vclk -period {PERIOD_NS}\nset_input_delay 0 -clock vclk [all_inputs]\n"
                       f"set_output_delay 0 -clock vclk [all_outputs]\nreport_worst_slack -max -digits 4\nexit\n")
        slack = float(re.search(r"worst slack(?: max)? (-?[\d.]+)", run([STA, "-no_init", "-no_splash", str(tcl)]))[1])
    return {"p": p, "q": q, "mode": mode, "acc_stage_um2": area, "acc_loop_ns": PERIOD_NS - slack}


def mac_power(p, q, mode, stim, n_steps, work):
    tag = f"p{p}_q{q}_{mode}"
    net, wrap = work / f"{tag}_net.v", work / f"{tag}_mac.v"
    wrap.write_text(MAC.replace("P_VAL", str(p)).replace("Q_VAL", str(q)).replace("RNE_VAL", str(int(mode == "rne"))))
    area = yosys_map(f"read_verilog -sv {RTL} {wrap}", "fp4_mac", net)
    outs, vcd = {}, work / f"{tag}.vcd"
    for kind, srcs in (("gate", [work / "cells.v", net]), ("rtl", [RTL, wrap])):
        tb = work / f"{tag}_{kind}_tb.v"
        tb.write_text(TB.format(P=p, half=PERIOD_NS // 2, stim=stim, out=work / f"{tag}_{kind}.out",
                                vcd=int(kind == "gate"), vcdfile=vcd))
        sim = work / f"{tag}_{kind}.vvp"
        run(["iverilog", "-g2012", "-o", str(sim), str(tb), *map(str, srcs)])
        run(["vvp", "-n", str(sim)])
        outs[kind] = (work / f"{tag}_{kind}.out").read_text()
    tcl = work / f"{tag}_p.tcl"
    tcl.write_text(f"read_liberty {LIB}\nread_verilog {net}\nlink_design fp4_mac\n"
                   f"create_clock -name clk -period {PERIOD_NS} [get_ports clk]\n"
                   f"read_vcd -scope tb/dut {vcd}\nreport_power -digits 6\nexit\n")
    rep = run([STA, "-no_init", "-no_splash", str(tcl)])
    internal, switching, leakage, total = map(float, re.search(r"^Total\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", rep, re.M).groups())
    vcd.unlink()
    return {"p": p, "q": q, "mode": mode, "mac_um2": area, "power_w": total, "internal_w": internal,
            "switching_w": switching, "leakage_w": leakage, "energy_pj_per_step": total * PERIOD_NS * 1e3,
            "steps": n_steps, "gate_matches_rtl": outs["gate"] == outs["rtl"] and bool(outs["gate"].strip())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["timing", "power"])
    ap.add_argument("--jobs", type=int, default=64)
    ap.add_argument("--designs", default=POWER_DESIGNS)
    a = ap.parse_args()
    assert LIB, "set SKY130_LIB to sky130_fd_sc_hd__tt_025C_1v80.lib"
    out_dir = ROOT / "results/ppa"
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.what == "timing":
        with ThreadPoolExecutor(a.jobs) as ex:
            rows = list(ex.map(lambda c: acc_timing(c[0], c[1] or 19, c[2]), SPACE))
        (out_dir / "timing.json").write_text(json.dumps(rows, indent=1))
        print(f"timed {len(rows)} accumulate stages")
        return

    streams = json.loads((ROOT / "data/llama_block_stream.json").read_text())["streams"]
    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        run(["yosys", "-q", "-p", f"read_liberty -ignore_miss_func -ignore_miss_dir {LIB}; "
                                  f"write_verilog -noattr {work / 'cells.v'}"])
        stim = work / "stim.txt"
        stim.write_text("".join(f"{int(i == len(s['blocks']) - 1)} {x} {w} {sx:02x} {sw:02x}\n"
                                for s in streams for i, (x, w, sx, sw) in enumerate(s["blocks"])))
        n_steps = sum(len(s["blocks"]) for s in streams)
        with ThreadPoolExecutor(len(a.designs.split(","))) as ex:
            rows = list(ex.map(lambda s: mac_power(*parse(s), stim, n_steps, work), a.designs.split(",")))
    (out_dir / "power.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"P{r['p']:>2} Q{r['q']:>2} {r['mode']:3s} mac={r['mac_um2']:8.0f} um2  "
              f"{r['energy_pj_per_step']:.3f} pJ/step  gate==rtl {r['gate_matches_rtl']}")


if __name__ == "__main__":
    main()
