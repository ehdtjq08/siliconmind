#!/bin/bash
# Synthesize fp4_block_dot for one (P, Q, RNE) onto sky130_fd_sc_hd standard cells and report cell area per module.
# SKY130_LIB points at sky130_fd_sc_hd__tt_025C_1v80.lib. Clock, delay, probe and low-power cells are left out, as in
# the OpenLane flow.
P=$1; Q=$2; RNE=$3; OUT=results/synth_sky130/p${P}_q${Q}_rne${RNE}.log
LIB=${SKY130_LIB:?set SKY130_LIB to sky130_fd_sc_hd__tt_025C_1v80.lib}
mkdir -p results/synth_sky130
yosys -q -p "read_verilog -sv rtl/fp4_block_dot.v; chparam -set P $P -set Q $Q -set RNE $RNE fp4_block_dot; \
  synth -top fp4_block_dot; abc -liberty $LIB -dont_use *__lpflow_* -dont_use *__probe_* -dont_use *__clk* \
  -dont_use *__dly*; opt_clean; tee -o $OUT stat -liberty $LIB" >/dev/null 2>&1
