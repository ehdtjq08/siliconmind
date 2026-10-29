/*
 * SiliconMind NVFP4 dot-product unit for Tiny Tapeout.
 * SPDX-License-Identifier: MIT
 *
 * Byte-serial wrapper around fp4_block_dot: load one 16-element NVFP4 block (two E2M1 vectors and two E4M3
 * block scales), then step the accumulator. P, Q and RNE select the accumulator design.
 *
 *   uio_in[0]    we     write ui_in into register addr on the rising clock edge
 *   uio_in[5:1]  addr   0-7: a (byte i = elements 2i, 2i+1), 8-15: b, 16: sa, 17: sb
 *   uio_in[6]    step   acc <= round_P(acc + round_Q(a.b * sa * sb))
 *   uio_in[7]    clear  acc <= 0
 *   uo_out              byte addr[2:0] of the accumulator {sign, exponent[9:0], significand[P-1:0]}
 */

`default_nettype none

module tt_um_ehdtjq08_siliconmind #(
    parameter P   = 9,
    parameter Q   = 7,
    parameter RNE = 1
) (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);
    localparam EW = 10;

    reg  [63:0]          a, b;
    reg  [7:0]           sa, sb;
    reg                  acc_s;
    reg  signed [EW-1:0] acc_e;
    reg  [P-1:0]         acc_m;
    wire                 nxt_s;
    wire signed [EW-1:0] nxt_e;
    wire [P-1:0]         nxt_m;

    fp4_block_dot #(.P(P), .Q(Q), .RNE(RNE), .EW(EW)) u_dot (
        .a(a), .b(b), .sa(sa), .sb(sb), .acc_s(acc_s), .acc_e(acc_e), .acc_m(acc_m),
        .out_s(nxt_s), .out_e(nxt_e), .out_m(nxt_m));

    wire       we    = uio_in[0];
    wire [4:0] addr  = uio_in[5:1];
    wire       step  = uio_in[6];
    wire       clear = uio_in[7];

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            acc_s <= 1'b0;
            acc_e <= {EW{1'b0}};
            acc_m <= {P{1'b0}};
        end else if (step) begin
            acc_s <= nxt_s;
            acc_e <= nxt_e;
            acc_m <= nxt_m;
        end
        if (we) begin
            if (addr < 5'd8)        a[{addr[2:0], 3'b000} +: 8] <= ui_in;
            else if (addr < 5'd16)  b[{addr[2:0], 3'b000} +: 8] <= ui_in;
            else if (addr == 5'd16) sa <= ui_in;
            else if (addr == 5'd17) sb <= ui_in;
        end
    end

    wire [63:0] acc_bits = {{(63 - EW - P){1'b0}}, acc_s, acc_e, acc_m};   // 64 wide so every addr is in range
    assign uo_out  = acc_bits[{addr[2:0], 3'b000} +: 8];
    assign uio_out = 8'h00;
    assign uio_oe  = 8'h00;

    wire _unused = &{ena, 1'b0};
endmodule
