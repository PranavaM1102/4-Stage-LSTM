//=================================================================
//  Stateless LSTM Core (Q6.11, 18-bit signed)
//  SAME MATH AS lstm_cell_q6_11 BUT WITHOUT INTERNAL REGISTERS
//=================================================================
module lstm_cell_q6_11_core #(
    parameter WIDTH = 18,
    parameter FRAC  = 11
)(
    input  wire signed [WIDTH-1:0]  x_t,
    input  wire signed [WIDTH-1:0]  c_prev,
    input  wire signed [WIDTH-1:0]  h_prev,

    // Forget gate weights
    input  wire signed [WIDTH-1:0]  W_fx,
    input  wire signed [WIDTH-1:0]  W_fh,
    input  wire signed [WIDTH-1:0]  b_f,

    // Input gate weights
    input  wire signed [WIDTH-1:0]  W_ix,
    input  wire signed [WIDTH-1:0]  W_ih,
    input  wire signed [WIDTH-1:0]  b_i,

    // Candidate gate weights
    input  wire signed [WIDTH-1:0]  W_gx,
    input  wire signed [WIDTH-1:0]  W_gh,
    input  wire signed [WIDTH-1:0]  b_g,

    // Output gate weights
    input  wire signed [WIDTH-1:0]  W_ox,
    input  wire signed [WIDTH-1:0]  W_oh,
    input  wire signed [WIDTH-1:0]  b_o,

    // NEW state after this time-step
    output wire signed [WIDTH-1:0]  c_new,
    output wire signed [WIDTH-1:0]  h_new
);
    // -------------------------------------------------------------
    // 1) Gate pre-activations
    // -------------------------------------------------------------
    wire signed [2*WIDTH-1:0] f_mul_x = x_t    * W_fx;
    wire signed [2*WIDTH-1:0] f_mul_h = h_prev * W_fh;
    wire signed [WIDTH-1:0]   f_pre   = (f_mul_x >>> FRAC) +
                                        (f_mul_h >>> FRAC) + b_f;

    wire signed [2*WIDTH-1:0] i_mul_x = x_t    * W_ix;
    wire signed [2*WIDTH-1:0] i_mul_h = h_prev * W_ih;
    wire signed [WIDTH-1:0]   i_pre   = (i_mul_x >>> FRAC) +
                                        (i_mul_h >>> FRAC) + b_i;

    wire signed [2*WIDTH-1:0] g_mul_x = x_t    * W_gx;
    wire signed [2*WIDTH-1:0] g_mul_h = h_prev * W_gh;
    wire signed [WIDTH-1:0]   g_pre   = (g_mul_x >>> FRAC) +
                                        (g_mul_h >>> FRAC) + b_g;

    wire signed [2*WIDTH-1:0] o_mul_x = x_t    * W_ox;
    wire signed [2*WIDTH-1:0] o_mul_h = h_prev * W_oh;
    wire signed [WIDTH-1:0]   o_pre   = (o_mul_x >>> FRAC) +
                                        (o_mul_h >>> FRAC) + b_o;

    // -------------------------------------------------------------
    // 2) Activations (σ and tanh LUTs from your original code)
    // -------------------------------------------------------------
    wire signed [WIDTH-1:0] f_gate;
    wire signed [WIDTH-1:0] i_gate;
    wire signed [WIDTH-1:0] g_gate;
    wire signed [WIDTH-1:0] o_gate;

    sigmoid_lut u_sig_f (.x(f_pre), .y(f_gate));
    sigmoid_lut u_sig_i (.x(i_pre), .y(i_gate));
    sigmoid_lut u_sig_o (.x(o_pre), .y(o_gate));
    tanh_lut    u_tanh_g(.x(g_pre), .y(g_gate));

    // -------------------------------------------------------------
    // 3) Cell update: C_t = f_t * C_prev + i_t * g_t
    // -------------------------------------------------------------
    wire signed [2*WIDTH-1:0] fC_mul = f_gate * c_prev;
    wire signed [2*WIDTH-1:0] iG_mul = i_gate * g_gate;

    wire signed [WIDTH-1:0] c_int =
        (fC_mul >>> FRAC) + (iG_mul >>> FRAC);

    // -------------------------------------------------------------
    // 4) Short-term memory: h_t = o_t * tanh(C_t)
    // -------------------------------------------------------------
    wire signed [WIDTH-1:0] c_tanh;
    tanh_lut u_tanh_c (.x(c_int), .y(c_tanh));

    wire signed [2*WIDTH-1:0] oC_mul = o_gate * c_tanh;
    wire signed [WIDTH-1:0]   h_int  = oC_mul >>> FRAC;

    assign c_new = c_int;
    assign h_new = h_int;

endmodule
//=================================================================
//  4-Step LSTM - PIPELINED VERSION
//  Uses 4 lstm_cell_q6_11_core blocks with pipeline registers
//=================================================================
module lstm_4step_pipeline #(
    parameter WIDTH = 18,
    parameter FRAC  = 11
)(
    input  wire                     clk,
    input  wire                     rst,

    // Inputs for 4 time-steps (Day 1..4)
    input  wire signed [WIDTH-1:0]  x1,
    input  wire signed [WIDTH-1:0]  x2,
    input  wire signed [WIDTH-1:0]  x3,
    input  wire signed [WIDTH-1:0]  x4,

    // Initial state (Day 0): usually 0
    input  wire signed [WIDTH-1:0]  c0,
    input  wire signed [WIDTH-1:0]  h0,

    // Shared weights for all time-steps
    input  wire signed [WIDTH-1:0]  W_fx,
    input  wire signed [WIDTH-1:0]  W_fh,
    input  wire signed [WIDTH-1:0]  b_f,
    input  wire signed [WIDTH-1:0]  W_ix,
    input  wire signed [WIDTH-1:0]  W_ih,
    input  wire signed [WIDTH-1:0]  b_i,
    input  wire signed [WIDTH-1:0]  W_gx,
    input  wire signed [WIDTH-1:0]  W_gh,
    input  wire signed [WIDTH-1:0]  b_g,
    input  wire signed [WIDTH-1:0]  W_ox,
    input  wire signed [WIDTH-1:0]  W_oh,
    input  wire signed [WIDTH-1:0]  b_o,

    // Predicted "Day 5" state (after 4 steps)
    output reg  signed [WIDTH-1:0]  c4_out,
    output reg  signed [WIDTH-1:0]  h4_out    // you can treat this as y_5
);

    // ---- Stage 0: Day 1 ------------------------------------------------
    wire signed [WIDTH-1:0] c1_w, h1_w;

    lstm_cell_q6_11_core #(.WIDTH(WIDTH), .FRAC(FRAC)) S0 (
        .x_t (x1),
        .c_prev (c0),
        .h_prev (h0),
        .W_fx(W_fx), .W_fh(W_fh), .b_f(b_f),
        .W_ix(W_ix), .W_ih(W_ih), .b_i(b_i),
        .W_gx(W_gx), .W_gh(W_gh), .b_g(b_g),
        .W_ox(W_ox), .W_oh(W_oh), .b_o(b_o),
        .c_new(c1_w),
        .h_new(h1_w)
    );

    // Pipeline registers for state going into Day 2
    reg signed [WIDTH-1:0] c1_p, h1_p;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            c1_p <= 0;
            h1_p <= 0;
        end else begin
            c1_p <= c1_w;
            h1_p <= h1_w;
        end
    end

    // ---- Stage 1: Day 2 ------------------------------------------------
    wire signed [WIDTH-1:0] c2_w, h2_w;

    lstm_cell_q6_11_core #(.WIDTH(WIDTH), .FRAC(FRAC)) S1 (
        .x_t (x2),
        .c_prev (c1_p),
        .h_prev (h1_p),
        .W_fx(W_fx), .W_fh(W_fh), .b_f(b_f),
        .W_ix(W_ix), .W_ih(W_ih), .b_i(b_i),
        .W_gx(W_gx), .W_gh(W_gh), .b_g(b_g),
        .W_ox(W_ox), .W_oh(W_oh), .b_o(b_o),
        .c_new(c2_w),
        .h_new(h2_w)
    );

    reg signed [WIDTH-1:0] c2_p, h2_p;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            c2_p <= 0;
            h2_p <= 0;
        end else begin
            c2_p <= c2_w;
            h2_p <= h2_w;
        end
    end

    // ---- Stage 2: Day 3 ------------------------------------------------
    wire signed [WIDTH-1:0] c3_w, h3_w;

    lstm_cell_q6_11_core #(.WIDTH(WIDTH), .FRAC(FRAC)) S2 (
        .x_t (x3),
        .c_prev (c2_p),
        .h_prev (h2_p),
        .W_fx(W_fx), .W_fh(W_fh), .b_f(b_f),
        .W_ix(W_ix), .W_ih(W_ih), .b_i(b_i),
        .W_gx(W_gx), .W_gh(W_gh), .b_g(b_g),
        .W_ox(W_ox), .W_oh(W_oh), .b_o(b_o),
        .c_new(c3_w),
        .h_new(h3_w)
    );

    reg signed [WIDTH-1:0] c3_p, h3_p;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            c3_p <= 0;
            h3_p <= 0;
        end else begin
            c3_p <= c3_w;
            h3_p <= h3_w;
        end
    end

    // ---- Stage 3: Day 4 ------------------------------------------------
    wire signed [WIDTH-1:0] c4_w, h4_w;

    lstm_cell_q6_11_core #(.WIDTH(WIDTH), .FRAC(FRAC)) S3 (
        .x_t (x4),
        .c_prev (c3_p),
        .h_prev (h3_p),
        .W_fx(W_fx), .W_fh(W_fh), .b_f(b_f),
        .W_ix(W_ix), .W_ih(W_ih), .b_i(b_i),
        .W_gx(W_gx), .W_gh(W_gh), .b_g(b_g),
        .W_ox(W_ox), .W_oh(W_oh), .b_o(b_o),
        .c_new(c4_w),
        .h_new(h4_w)
    );

    // Final pipeline register: "Day 5" prediction
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            c4_out <= 0;
            h4_out <= 0;
        end else begin
            c4_out <= c4_w;
            h4_out <= h4_w;
        end
    end

endmodule
