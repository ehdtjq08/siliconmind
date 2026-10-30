#!/bin/bash
# Synthesize fp4_block_dot for one (P, Q, RNE) onto sky130_fd_sc_hd standard cells and report cell area per module.
# SKY130_LIB points at sky130_fd_sc_hd__tt_025C_1v80.lib. Clock, delay, probe and low-power cells are left out, as in
# the OpenLane flow. Parameters go in through a literal wrapper, as in scripts/synth_sweep.sh.
P=$1; Q=$2; RNE=$3; OUT=results/synth_sky130/p${P}_q${Q}_rne${RNE}.log
LIB=${SKY130_LIB:?set SKY130_LIB to sky130_fd_sc_hd__tt_025C_1v80.lib}
mkdir -p results/synth_sky130
TOP=$(mktemp --suffix=.v); trap 'rm -f $TOP' EXIT
cat > $TOP <<V
module synth_top (input wire [63:0] a, input wire [63:0] b, input wire [7:0] sa, input wire [7:0] sb,
    input wire acc_s, input wire signed [9:0] acc_e, input wire [$((P - 1)):0] acc_m,
    output wire out_s, output wire signed [9:0] out_e, output wire [$((P - 1)):0] out_m);
    fp4_block_dot #(.P($P), .Q($Q), .RNE($RNE)) u (.a(a), .b(b), .sa(sa), .sb(sb), .acc_s(acc_s), .acc_e(acc_e),
        .acc_m(acc_m), .out_s(out_s), .out_e(out_e), .out_m(out_m));
endmodule
V
yosys -q -p "read_verilog -sv rtl/fp4_block_dot.v $TOP; synth -top synth_top; abc -liberty $LIB -dont_use *__lpflow_* \
  -dont_use *__probe_* -dont_use *__clk* -dont_use *__dly*; opt_clean; tee -o $OUT stat -liberty $LIB" >/dev/null 2>&1
