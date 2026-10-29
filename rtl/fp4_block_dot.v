// NVFP4 block dot-product + accumulate step (combinational).
//
//   R       = sum_{i<16} a_i*b_i * sa * sb                     (exact, fp4_block_prod)
//   acc_out = round_P(acc_in + round_Q(R))                      (fp_acc_add)
//
// a_i, b_i: E2M1 codes; sa, sb: E4M3FN block scales; accumulator: (-1)^s * m * 2^e with
// m a normalized P-bit significand (or zero). Q >= 19 keeps R exact. Rounding is RNE or toward
// zero, matching siliconmind/numerics.py (ref_block_step).

// ------------------------------------------------------------------ exact block product
module fp4_block_prod #(
    parameter XW = 14
) (
    input  wire [63:0]          a,      // element i at [4i+3:4i], {sign, exp[1:0], man}
    input  wire [63:0]          b,
    input  wire [7:0]           sa,     // {sign(ignored), exp[3:0], man[2:0]}
    input  wire [7:0]           sb,
    output wire                 r_s,
    output wire signed [XW-1:0] r_e,    // R = (-1)^r_s * r_m * 2^r_e
    output wire [18:0]          r_m
);
    function [3:0] e2m1_mag;            // magnitude in units of 1/2
        input [2:0] c;
        case (c)
            3'd0: e2m1_mag = 4'd0;  3'd1: e2m1_mag = 4'd1;
            3'd2: e2m1_mag = 4'd2;  3'd3: e2m1_mag = 4'd3;
            3'd4: e2m1_mag = 4'd4;  3'd5: e2m1_mag = 4'd6;
            3'd6: e2m1_mag = 4'd8;  3'd7: e2m1_mag = 4'd12;
        endcase
    endfunction

    reg signed [13:0] psum;             // in units of 1/4, |psum| <= 2304
    reg [7:0] prod;
    integer i;
    always @* begin
        psum = 14'sd0;
        for (i = 0; i < 16; i = i + 1) begin
            prod = e2m1_mag(a[4*i +: 3]) * e2m1_mag(b[4*i +: 3]);
            if (a[4*i+3] ^ b[4*i+3]) psum = psum - $signed({6'd0, prod});
            else                     psum = psum + $signed({6'd0, prod});
        end
    end

    wire       sa_sub = (sa[6:3] == 4'd0);
    wire       sb_sub = (sb[6:3] == 4'd0);
    wire [3:0] siga   = {~sa_sub, sa[2:0]};
    wire [3:0] sigb   = {~sb_sub, sb[2:0]};
    wire signed [XW-1:0] ea = sa_sub ? -9 : $signed({{(XW-4){1'b0}}, sa[6:3]}) - 10;
    wire signed [XW-1:0] eb = sb_sub ? -9 : $signed({{(XW-4){1'b0}}, sb[6:3]}) - 10;
    wire [12:0] pmag = psum[13] ? -psum[12:0] : psum[12:0];

    assign r_s = psum[13];
    assign r_m = pmag * siga * sigb;
    assign r_e = ea + eb - 2;
endmodule

// ------------------------------------------------------------------ accumulator adder
module fp_acc_add #(
    parameter P   = 12,   // accumulator significand bits, incl. hidden one
    parameter Q   = 19,   // R is rounded to Q bits before the add (>= 19: exact)
    parameter RNE = 1,    // 1: round to nearest even, 0: round toward zero (both roundings)
    parameter EW  = 10,
    parameter XW  = 14
) (
    input  wire                 acc_s,
    input  wire signed [EW-1:0] acc_e,
    input  wire [P-1:0]         acc_m,
    input  wire                 r_s,
    input  wire signed [XW-1:0] r_e,
    input  wire [18:0]          r_m,
    output reg                  out_s,
    output reg  signed [EW-1:0] out_e,
    output reg  [P-1:0]         out_m
);
    localparam RW = 19;
    localparam QW = (Q < RW) ? Q : RW;          // width of R after pre-rounding
    localparam SW = (P > QW) ? P : QW;          // widest significand entering the add
    localparam L  = SW + 3;                     // alignment window below the top bit
    localparam F  = L + 3;                      // frame: sticky, L+1 window bits, carry

    reg  [RW-1:0] rn;
    reg  [QW:0]   rq;
    reg           rg, rs_, ri;
    reg  signed [XW-1:0] r_top, x_top, top, idx;
    reg  [F-1:0]  fx, fy;
    reg  signed [F+1:0] sum;
    reg  [F:0]    mag;
    reg  [P:0]    kept;
    reg           guard, sticky, inc;
    integer       k, lz, msb;

    // place significand s (width w, MSB at absolute position stop) into the frame whose
    // index 0 (sticky) sits at absolute position t-L-1; shifted-out bits OR into index 0
    function [F-1:0] place;
        input [SW-1:0] s;
        input integer w;
        input signed [XW-1:0] stop;
        input signed [XW-1:0] t;
        reg signed [XW-1:0] lsb_idx, amt;
        begin
            lsb_idx = stop - w + 2 - t + L;
            if (lsb_idx >= 1) begin
                place = s << lsb_idx;
            end else begin
                amt = 1 - lsb_idx;
                if (amt >= w)
                    place = {{(F-1){1'b0}}, |s};
                else
                    place = ((s >> amt) << 1) | {{(F-1){1'b0}}, |(s & ~({SW{1'b1}} << amt))};
            end
        end
    endfunction

    always @* begin
        // normalize R, then pre-round it to QW bits
        lz = 0;
        for (k = 0; k < RW; k = k + 1)
            if (r_m[k]) lz = RW - 1 - k;
        rn    = r_m << lz;
        r_top = r_e + RW - 1 - lz;
        rq    = {1'b0, rn[RW-1 -: QW]};
        rg = 0; rs_ = 0; ri = 0;
        if (QW < RW) begin
            rg  = rn[RW-1-QW];
            rs_ = (QW < RW - 1) ? |(rn & ~({RW{1'b1}} << (RW - 1 - QW))) : 1'b0;
            ri  = RNE ? (rg & (rs_ | rq[0])) : 1'b0;
            rq  = rq + ri;
            if (rq[QW]) begin
                rq    = rq >> 1;
                r_top = r_top + 1;
            end
        end
        x_top = acc_e + P - 1;

        out_s = acc_s; out_e = acc_e; out_m = acc_m;
        fx = 0; fy = 0; sum = 0; mag = 0; kept = 0; guard = 0; sticky = 0; inc = 0; msb = 0;
        top = 0; idx = 0;

        if (r_m != 0) begin
            top = (acc_m == 0 || r_top > x_top) ? r_top : x_top;
            fx  = (acc_m == 0) ? {F{1'b0}} : place(acc_m, P, x_top, top);
            fy  = place(rq[QW-1:0], QW, r_top, top);
            sum = (acc_s ? -$signed({2'b00, fx}) : $signed({2'b00, fx}))
                + (r_s   ? -$signed({2'b00, fy}) : $signed({2'b00, fy}));
            out_s = sum[F+1];
            mag   = sum[F+1] ? -sum[F:0] : sum[F:0];
            if (mag == 0) begin
                out_s = 1'b0; out_e = 0; out_m = 0;
            end else begin
                for (k = 0; k <= F; k = k + 1)
                    if (mag[k]) msb = k;
                if (msb >= P) begin
                    kept   = mag >> (msb - P + 1);
                    guard  = mag[msb - P];
                    sticky = |(mag & ~({(F+1){1'b1}} << (msb - P)));
                end else begin
                    kept   = mag << (P - 1 - msb);
                end
                inc  = RNE ? (guard & (sticky | kept[0])) : 1'b0;
                kept = kept + inc;
                idx  = top - L - 1 + msb;       // absolute position of the result MSB
                if (kept[P]) begin
                    out_m = kept[P:1];
                    out_e = idx - P + 2;
                end else begin
                    out_m = kept[P-1:0];
                    out_e = idx - P + 1;
                end
            end
        end
    end
endmodule

// ------------------------------------------------------------------ top
module fp4_block_dot #(
    parameter P   = 12,
    parameter Q   = 19,
    parameter RNE = 1,
    parameter EW  = 10
) (
    input  wire [63:0]          a,
    input  wire [63:0]          b,
    input  wire [7:0]           sa,
    input  wire [7:0]           sb,
    input  wire                 acc_s,
    input  wire signed [EW-1:0] acc_e,
    input  wire [P-1:0]         acc_m,
    output wire                 out_s,
    output wire signed [EW-1:0] out_e,
    output wire [P-1:0]         out_m
);
    localparam XW = EW + 4;
    wire                 r_s;
    wire signed [XW-1:0] r_e;
    wire [18:0]          r_m;

    fp4_block_prod #(.XW(XW)) u_prod (.a(a), .b(b), .sa(sa), .sb(sb), .r_s(r_s), .r_e(r_e), .r_m(r_m));
    fp_acc_add #(.P(P), .Q(Q), .RNE(RNE), .EW(EW), .XW(XW)) u_acc (
        .acc_s(acc_s), .acc_e(acc_e), .acc_m(acc_m), .r_s(r_s), .r_e(r_e), .r_m(r_m),
        .out_s(out_s), .out_e(out_e), .out_m(out_m));
endmodule
