// Reads stimulus lines "a b sa sb acc_s acc_e acc_m" (hex) and writes "out_s out_e out_m" (hex).
module tb;
    parameter P = 12;
    parameter Q = 19;
    parameter RNE = 1;
    parameter EW = 10;

    reg  [63:0] a, b;
    reg  [7:0]  sa, sb;
    reg         acc_s;
    reg  [EW-1:0] acc_e;
    reg  [P-1:0]  acc_m;
    wire        out_s;
    wire [EW-1:0] out_e;
    wire [P-1:0]  out_m;

    fp4_block_dot #(.P(P), .Q(Q), .RNE(RNE), .EW(EW)) dut (
        .a(a), .b(b), .sa(sa), .sb(sb), .acc_s(acc_s), .acc_e(acc_e), .acc_m(acc_m),
        .out_s(out_s), .out_e(out_e), .out_m(out_m));

    integer fin, fout, n;
    reg [1023:0] vin, vout;
    initial begin
        if (!$value$plusargs("in=%s", vin)) $finish;
        if (!$value$plusargs("out=%s", vout)) $finish;
        fin = $fopen(vin, "r");
        fout = $fopen(vout, "w");
        while (!$feof(fin)) begin
            n = $fscanf(fin, "%h %h %h %h %h %h %h\n", a, b, sa, sb, acc_s, acc_e, acc_m);
            if (n == 7) begin
                #1 $fwrite(fout, "%h %h %h\n", out_s, out_e, out_m);
            end
        end
        $fclose(fout);
        $finish;
    end
endmodule
