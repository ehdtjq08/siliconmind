"""Place and route the whole multiply-accumulate unit with LibreLane (sky130_fd_sc_hd), one run directory per design.

The unit is registered on both sides: input registers, one NVFP4 block step (fp4_block_dot), accumulator register.

    python scripts/pnr_runs.py make 24:x:rne 9:7:rne       # P:Q:mode, Q = x for exact -> results/pnr/p9_q7_rne/, ...
    cd results/pnr/p9_q7_rne && librelane --dockerized --pdk sky130A config.json      # LibreLane 3.0.17
    python scripts/pnr_runs.py summary                     # area, slack and signoff checks -> results/pnr/summary.json
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pnr"

MAC = """// Registered multiply-accumulate unit for place and route: input registers, one NVFP4 block step, accumulator register.
module mac_top (input wire clk, input wire clr, input wire [63:0] a_in, input wire [63:0] b_in,
                input wire [7:0] sa_in, input wire [7:0] sb_in,
                output reg acc_s, output reg signed [9:0] acc_e, output reg [{pm1}:0] acc_m);
    reg [63:0] a, b; reg [7:0] sa, sb; reg c;
    wire s; wire signed [9:0] e; wire [{pm1}:0] m;
    fp4_block_dot #(.P({p}), .Q({q}), .RNE({rne})) u (.a(a), .b(b), .sa(sa), .sb(sb),
        .acc_s(acc_s), .acc_e(acc_e), .acc_m(acc_m), .out_s(s), .out_e(e), .out_m(m));
    always @(posedge clk) begin
        {{a, b, sa, sb, c}} <= {{a_in, b_in, sa_in, sb_in, clr}};
        {{acc_s, acc_e, acc_m}} <= c ? 0 : {{s, e, m}};
    end
endmodule
"""
CONFIG = {"DESIGN_NAME": "mac_top", "VERILOG_FILES": ["dir::src/fp4_block_dot.v", "dir::src/mac_top.v"],
          "CLOCK_PORT": "clk", "CLOCK_PERIOD": 40, "FP_CORE_UTIL": 45, "PL_TARGET_DENSITY_PCT": 55}
METRICS = {"cells": "design__instance__count__stdcell", "area_um2": "design__instance__area__stdcell",
           "slack_tt_ns": "timing__setup__ws__corner:nom_tt_025C_1v80", "drc": "magic__drc_error__count",
           "lvs": "design__lvs_error__count", "antenna": "antenna__violating__nets"}


def make(specs):
    for spec in specs:
        p, q, mode = spec.split(":")
        d = OUT / f"p{p}_q{q}_{mode}"
        (d / "src").mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "rtl/fp4_block_dot.v", d / "src")
        (d / "src/mac_top.v").write_text(MAC.format(p=p, pm1=int(p) - 1, q=19 if q == "x" else int(q),
                                                    rne=int(mode == "rne")))
        (d / "config.json").write_text(json.dumps(CONFIG, indent=1))
        print(d)


def summary():
    rows = {}
    for d in sorted(OUT.glob("p*_q*_*")):
        runs = sorted(d.glob("runs/*/final/metrics.json"))
        if runs:                                                    # the latest run of each design
            m = json.loads(runs[-1].read_text())
            rows[d.name] = {k: m[v] for k, v in METRICS.items()}
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1))
    for name, r in sorted(rows.items(), key=lambda kv: kv[1]["area_um2"]):
        print(f"{name:12s} {r['cells']:6d} cells {r['area_um2']:9.0f} um2  tt slack {r['slack_tt_ns']:+6.2f} ns  "
              f"drc {r['drc']} lvs {r['lvs']} antenna {r['antenna']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "summary"])
    ap.add_argument("specs", nargs="*")
    a = ap.parse_args()
    make(a.specs) if a.cmd == "make" else summary()
