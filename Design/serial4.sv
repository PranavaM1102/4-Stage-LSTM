//=================================================================
//  4-Step LSTM - SERIAL VERSION
//  Reuses ONE lstm_cell_q6_11_core over 4 cycles
//=================================================================
module lstm_4step_serial #(
    parameter WIDTH = 18,
    parameter FRAC  = 11
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     start,   // pulse to start a new 4-step run

    // Inputs for 4 time-steps (Day 1..4)
    input  wire signed [WIDTH-1:0]  x1,
    input  wire signed [WIDTH-1:0]  x2,
    input  wire signed [WIDTH-1:0]  x3,
    input  wire signed [WIDTH-1:0]  x4,

    // Initial state (Day 0)
    input  wire signed [WIDTH-1:0]  c0,
    input  wire signed [WIDTH-1:0]  h0,

    // Shared weights
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

    // Output after 4 steps (Day 5 prediction)
    output reg  signed [WIDTH-1:0]  c4_out,
    output reg  signed [WIDTH-1:0]  h4_out,
    output reg                      done    // 1-cycle pulse when result ready
);

    // FSM step counter: 0 = idle, 1..4 = Day1..Day4
    reg [2:0] step;

    // Current state being fed into the core
    reg signed [WIDTH-1:0] c_reg, h_reg;

    // Select which day's x_t we are processing
    reg signed [WIDTH-1:0] x_sel;
    always @* begin
        case (step)
            3'd1: x_sel = x1;  // Day 1
            3'd2: x_sel = x2;  // Day 2
            3'd3: x_sel = x3;  // Day 3
            3'd4: x_sel = x4;  // Day 4
            default: x_sel = 0;
        endcase
    end

    // One LSTM core reused each cycle
    wire signed [WIDTH-1:0] c_next, h_next;

    lstm_cell_q6_11_core #(.WIDTH(WIDTH), .FRAC(FRAC)) CORE (
        .x_t   (x_sel),
        .c_prev(c_reg),
        .h_prev(h_reg),
        .W_fx(W_fx), .W_fh(W_fh), .b_f(b_f),
        .W_ix(W_ix), .W_ih(W_ih), .b_i(b_i),
        .W_gx(W_gx), .W_gh(W_gh), .b_g(b_g),
        .W_ox(W_ox), .W_oh(W_oh), .b_o(b_o),
        .c_new(c_next),
        .h_new(h_next)
    );

    // FSM + state registers
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            step   <= 3'd0;
            done   <= 1'b0;
            c_reg  <= {WIDTH{1'b0}};
            h_reg  <= {WIDTH{1'b0}};
            c4_out <= {WIDTH{1'b0}};
            h4_out <= {WIDTH{1'b0}};
        end else begin
            done <= 1'b0;   // default

            case (step)
                3'd0: begin  // IDLE
                    if (start) begin
                        // Load initial state Day 0
                        c_reg <= c0;
                        h_reg <= h0;
                        step  <= 3'd1;  // go to Day 1
                    end
                end

                3'd1, 3'd2, 3'd3, 3'd4: begin
                    // One time-step per clock: feed through core
                    c_reg <= c_next;
                    h_reg <= h_next;

                    if (step == 3'd4) begin
                        // Finished Day 4 → output Day-5 prediction
                        c4_out <= c_next;
                        h4_out <= h_next;
                        done   <= 1'b1;
                        step   <= 3'd0;    // back to IDLE
                    end else begin
                        step <= step + 3'd1;
                    end
                end

                default: step <= 3'd0;
            endcase
        end
    end

endmodule
