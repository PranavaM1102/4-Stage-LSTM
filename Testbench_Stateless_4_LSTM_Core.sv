`timescale 1ns/1ps

module tb_lstm_4step_pipeline;

    localparam WIDTH = 18;
    localparam FRAC  = 11;

    reg clk, rst;

    // Inputs
    reg signed [WIDTH-1:0] x1, x2, x3, x4;
    reg signed [WIDTH-1:0] c0, h0;

    // Parsed weights from your image (in Q6.11)
    reg signed [WIDTH-1:0] W_fx = 18'sd3333;   // 1.63
    reg signed [WIDTH-1:0] W_fh = 18'sd5529;   // 2.70
    reg signed [WIDTH-1:0] b_f  = 18'sd3318;   // 1.62

    reg signed [WIDTH-1:0] W_ix = 18'sd3379;   // 1.65
    reg signed [WIDTH-1:0] W_ih = 18'sd4096;   // 2.00
    reg signed [WIDTH-1:0] b_i  = 18'sd1269;   // 0.62

    reg signed [WIDTH-1:0] W_gx = 18'sd1923;   // 0.94
    reg signed [WIDTH-1:0] W_gh = 18'sd2891;   // 1.41
    reg signed [WIDTH-1:0] b_g  = -18'sd655;   // -0.32

    reg signed [WIDTH-1:0] W_ox = -18'sd389;   // -0.19
    reg signed [WIDTH-1:0] W_oh = 18'sd8970;   // 4.38
    reg signed [WIDTH-1:0] b_o  = 18'sd1208;   // 0.59

    wire signed [WIDTH-1:0] c4_out, h4_out;

    // DUT
    lstm_4step_pipeline DUT(
        .clk(clk), .rst(rst),
        .x1(x1), .x2(x2), .x3(x3), .x4(x4),
        .c0(c0), .h0(h0),
        .W_fx(W_fx), .W_fh(W_fh), .b_f(b_f),
        .W_ix(W_ix), .W_ih(W_ih), .b_i(b_i),
        .W_gx(W_gx), .W_gh(W_gh), .b_g(b_g),
        .W_ox(W_ox), .W_oh(W_oh), .b_o(b_o),
        .c4_out(c4_out), .h4_out(h4_out)
    );

    // Clock
    always #5 clk = ~clk;

    // Float → Q6.11
    function signed [WIDTH-1:0] Q(real r);
        Q = r * (1 << FRAC);
    endfunction

    initial begin
        $display("\n=== PIPELINED LSTM TEST ===\n");

        clk = 0;
        rst = 1;

        // Inputs from your earlier Day example
        x1 = Q(1.0);
        x2 = Q(0.5);
        x3 = Q(0.25);
        x4 = Q(1.0);

        c0 = Q(0.0);
        h0 = Q(0.0);

        #20 rst = 0;

        repeat(8) @(posedge clk);

        $display("Final C4 (Day-5 C): %f", c4_out / 2048.0);
        $display("Final H4 (Day-5 H): %f", h4_out / 2048.0);

        $finish;
    end

endmodule
