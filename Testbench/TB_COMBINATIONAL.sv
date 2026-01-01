`timescale 1ns/1ps

module tb_lstm_lut;

    localparam WIDTH = 18;
    localparam FRAC  = 11;
    localparam real TOL = 0.03;     // tolerance (3%)

    reg clk;
    reg rst;

    // inputs
    reg  signed [WIDTH-1:0] x_t;
    reg  signed [WIDTH-1:0] c_prev;
    reg  signed [WIDTH-1:0] h_prev;

    // outputs
    wire signed [WIDTH-1:0] c_t;
    wire signed [WIDTH-1:0] h_t;

    // ------------------------------------------------------------
    // StatQuest weights (Q6.11)
    // ------------------------------------------------------------
    reg signed [WIDTH-1:0] W_fx = 18'sd3338;   // 1.63
    reg signed [WIDTH-1:0] W_fh = 18'sd5529;   // 2.70
    reg signed [WIDTH-1:0] b_f  = 18'sd3318;   // 1.62

    reg signed [WIDTH-1:0] W_ix = 18'sd3379;   // 1.65
    reg signed [WIDTH-1:0] W_ih = 18'sd4096;   // 2.00
    reg signed [WIDTH-1:0] b_i  = 18'sd1269;   // 0.62

    reg signed [WIDTH-1:0] W_gx = 18'sd1926;   // 0.94
    reg signed [WIDTH-1:0] W_gh = 18'sd2889;   // 1.41
    reg signed [WIDTH-1:0] b_g  = -18'sd655;   // -0.32

    reg signed [WIDTH-1:0] W_ox = -18'sd389;   // -0.19
    reg signed [WIDTH-1:0] W_oh = 18'sd8970;   // 4.38
    reg signed [WIDTH-1:0] b_o  = 18'sd1209;   // 0.59

    // ------------------------------------------------------------
    // DUT
    // ------------------------------------------------------------
    lstm_cell_q6_11 dut (
        .clk(clk),
        .rst(rst),

        .x_t(x_t),
        .c_prev(c_prev),
        .h_prev(h_prev),

        .W_fx(W_fx), .W_fh(W_fh), .b_f(b_f),
        .W_ix(W_ix), .W_ih(W_ih), .b_i(b_i),
        .W_gx(W_gx), .W_gh(W_gh), .b_g(b_g),
        .W_ox(W_ox), .W_oh(W_oh), .b_o(b_o),

        .c_t(c_t),
        .h_t(h_t)
    );

    // Clock
    always #5 clk = ~clk;

    // Convert Q6.11 → real
    function real q2r (input signed [WIDTH-1:0] v);
        q2r = v / 2048.0;
    endfunction

    // Pass/Fail judge
    task check(string tag, real expC, real expH);
        real aC, aH;
        real dc, dh;
    begin
        aC = q2r(c_t);
        aH = q2r(h_t);

        dc = (aC > expC) ? (aC - expC) : (expC - aC);
        dh = (aH > expH) ? (aH - expH) : (expH - aH);

        $display("\n--- %s ---", tag);
        $display("Expected C = %f   Actual C = %f", expC, aC);
        $display("Expected H = %f   Actual H = %f", expH, aH);

        if (dc < TOL && dh < TOL)
            $display("RESULT: PASS\n");
        else
            $display("RESULT: FAIL (dC=%f  dH=%f)\n", dc, dh);
    end
    endtask

    // ------------------------------------------------------------
    // Python EXPECTED Values
    // ------------------------------------------------------------
    real EXP_C1 = 0.499521;
    real EXP_H1 = 0.276438;

    real EXP_C2 = 0.913569;
    real EXP_H2 = 0.611733;

    // ------------------------------------------------------------
    // Test Sequence
    // ------------------------------------------------------------
    initial begin
        $display("================ LSTM LUT TESTBENCH ================");

        clk = 0;
        rst = 1;
        c_prev = 0;
        h_prev = 0;

        // --------------------------------------------------------
        // STEP 1: x = 1.0
        // --------------------------------------------------------
        x_t = 18'sd2048; // 1.0 Q6.11

        #20 rst = 0;
        @(posedge clk); #1;

        check("STEP 1", EXP_C1, EXP_H1);

        // store results for step2
        c_prev = c_t;
        h_prev = h_t;

        // --------------------------------------------------------
        // STEP 2: x = 0.5
        // --------------------------------------------------------
        x_t = 18'sd1024; // 0.5 Q6.11

        @(posedge clk); #1;

        check("STEP 2", EXP_C2, EXP_H2);

        $display("================  TEST COMPLETE  ===================");

        #10 $finish;
    end

endmodule
