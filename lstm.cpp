//#include <stdint.h>

#include "ap_int.h"
#include "ap_fixed.h"
#include "lstm.hpp"

#define INPUT_SEQUENTIAL_D 120
#define INPUT_FEATURE_D 7
#define HIDDEN_FEATURE_D 32
#define PARTITION_DIM
#define PARTITION_FACTOR
// naiive
t_feature sigmoid(t_feature input)
{
    if (input > 6)
        return 1.0;
    if (input < -6)
        return 0.0;
    else
        return sigmoid_lut[(int)(input + 6) * 1024];
}

t_feature tanh(t_feature input)
{
    if (input > 4)
        return 1.0;
    if (input < -4)
        return -1.0;
    else
        return tanh_lut[(int)(input + 4) * 1024];
}

void input_to_hidden_dot_product(t_feature x[INPUT_FEATURE_D], t_feature W[INPUT_FEATURE_D][HIDDEN_FEATURE_D], t_feature output[HIDDEN_FEATURE_D])
{
DOT_ITH_1:
    for (int i = 0; i < INPUT_FEATURE_D; ++i)
    {
DOT_ITH_2:
        for (int j = 0; j < HIDDEN_FEATURE_D; ++j)
        {
            output[j] += x[i] * W[i][j];
        }
    }
}

void hidden_to_hidden_dot_product(t_feature h[HIDDEN_FEATURE_D], t_feature U[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D], t_feature output[HIDDEN_FEATURE_D])
{
DOT_HTH_1:
    for (int i = 0; i < HIDDEN_FEATURE_D; ++i)
    {
DOT_HTH_2:
        for (int j = 0; j < HIDDEN_FEATURE_D; ++j)
        {
            output[i] += h[i] * U[i][j];
        }
    }
}
/*
    input gate/ forget gate/ output gate share the same function
*/
void unified_gate(t_feature W[INPUT_FEATURE_D][HIDDEN_FEATURE_D], t_feature x[INPUT_FEATURE_D],
                  t_feature U[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D], t_feature h[HIDDEN_FEATURE_D],
                  t_feature bias[HIDDEN_FEATURE_D],
                  t_feature output[HIDDEN_FEATURE_D])
{
    t_feature output_x[HIDDEN_FEATURE_D];
    t_feature output_h[HIDDEN_FEATURE_D];

    input_to_hidden_dot_product(x, W, output_x);
    hidden_to_hidden_dot_product(h, U, output_h);
UNIFIED_LOOP:
    for (int i = 0; i < HIDDEN_FEATURE_D; ++i)
    {
        t_feature tmp_sum = output_x[i] + output_h[i] + bias[i];
        output[i] = sigmoid(tmp_sum);
    }
}

void g_gate(t_feature W[INPUT_FEATURE_D][HIDDEN_FEATURE_D], t_feature x[INPUT_FEATURE_D],
                t_feature U[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D], t_feature h[HIDDEN_FEATURE_D],
                t_feature bias[HIDDEN_FEATURE_D],
                t_feature output[HIDDEN_FEATURE_D])
{
    t_feature output_x[HIDDEN_FEATURE_D];
    t_feature output_h[HIDDEN_FEATURE_D];

    input_to_hidden_dot_product(x, W, output_x);
    hidden_to_hidden_dot_product(h, U, output_h);
G_GATE_LOOP:
    for (int i = 0; i < HIDDEN_FEATURE_D; ++i)
    {
        t_feature tmp_sum = output_x[i] + output_h[i] + bias[i];
        output[i] = tanh(tmp_sum);
    }
}

void cell_out(t_feature forget[HIDDEN_FEATURE_D], t_feature last_c[HIDDEN_FEATURE_D], t_feature input_t[HIDDEN_FEATURE_D], t_feature g_gate[HIDDEN_FEATURE_D],
               t_feature output[HIDDEN_FEATURE_D])
{
CELL_LOOP:
    for (int i = 0; i < HIDDEN_FEATURE_D; ++i)
    {
        output[i] = forget[i] * last_c[i] + input_t[i] * g_gate[i];
    }
}

void next_hidden(t_feature output_t[HIDDEN_FEATURE_D], t_feature cell_out_t[HIDDEN_FEATURE_D], t_feature output[HIDDEN_FEATURE_D])
{
NEXT_HIDDEN_LOOP:
    for (int i = 0; i < HIDDEN_FEATURE_D; ++i)
    {
        output[i] = output_t[i] * tanh(cell_out_t[i]);
    }
}

// 32 unit
static void LSTM_cell(t_feature input[INPUT_FEATURE_D],
                      t_feature hidden_state[HIDDEN_FEATURE_D],
                      t_feature cell_state[HIDDEN_FEATURE_D],
                      t_weight weight_matrix_hf[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D],
                      t_weight weight_matrix_xf[INPUT_FEATURE_D][HIDDEN_FEATURE_D],
                      t_weight weight_matrix_hi[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D],
                      t_weight weight_matrix_xi[INPUT_FEATURE_D][HIDDEN_FEATURE_D],
                      t_weight weight_matrix_ho[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D],
                      t_weight weight_matrix_xo[INPUT_FEATURE_D][HIDDEN_FEATURE_D],
                      t_weight weight_matrix_hg[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D],
                      t_weight weight_matrix_xg[INPUT_FEATURE_D][HIDDEN_FEATURE_D],
                      t_weight bias_f[HIDDEN_FEATURE_D],
                      t_weight bias_i[HIDDEN_FEATURE_D],
                      t_weight bias_o[HIDDEN_FEATURE_D],
                      t_weight bias_c[HIDDEN_FEATURE_D],
                      t_feature next_hidden_state[HIDDEN_FEATURE_D],
                      t_feature next_cell_state[HIDDEN_FEATURE_D])
{
    t_feature output_i[HIDDEN_FEATURE_D];
    t_feature output_f[HIDDEN_FEATURE_D];
    t_feature output_g[HIDDEN_FEATURE_D];
    t_feature output_o[HIDDEN_FEATURE_D];

#pragma HLS inline recursive
    // input gate
    unified_gate(weight_matrix_xi, input, weight_matrix_hi, hidden_state, bias_i, output_i);
    // forget gate
    unified_gate(weight_matrix_xf, input, weight_matrix_hf, hidden_state, bias_f, output_f);
    // g gate
    g_gate(weight_matrix_xg, input, weight_matrix_hg, hidden_state, bias_c, output_g);
    // output gate
    unified_gate(weight_matrix_xo, input, weight_matrix_ho, hidden_state, bias_o, output_o);
    // cell out
    cell_out(output_f, cell_state, output_i, output_g, next_cell_state);
    // next_hidden state
    next_hidden(output_o, next_cell_state, next_hidden_state);
}

// wrapper lstm

int wrapper_flow(t_feature input_seq[INPUT_SEQUENTIAL_D][INPUT_FEATURE_D],
                 t_feature hidden_state_seq[INPUT_SEQUENTIAL_D][HIDDEN_FEATURE_D],
                 t_weight weight_matrix_hf[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D],
                 t_weight weight_matrix_xf[INPUT_FEATURE_D][HIDDEN_FEATURE_D],
                 t_weight weight_matrix_hi[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D],
                 t_weight weight_matrix_xi[INPUT_FEATURE_D][HIDDEN_FEATURE_D],
                 t_weight weight_matrix_ho[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D],
                 t_weight weight_matrix_xo[INPUT_FEATURE_D][HIDDEN_FEATURE_D],
                 t_weight weight_matrix_hg[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D],
                 t_weight weight_matrix_xg[INPUT_FEATURE_D][HIDDEN_FEATURE_D],
                 t_weight bias_f[HIDDEN_FEATURE_D],
                 t_weight bias_i[HIDDEN_FEATURE_D],
                 t_weight bias_o[HIDDEN_FEATURE_D],
                 t_weight bias_c[HIDDEN_FEATURE_D])
{

    t_feature cell_state_seq[INPUT_SEQUENTIAL_D][HIDDEN_FEATURE_D];
    t_feature dummy_hidden[32] = {0};
    t_feature dummy_cell[32] = {0};
//#pragma HLS dataflow
#pragma HLS inline off
#pragma HLS array_partition variable = weight_matrix_hf cyclic dim = 2 factor = 32
#pragma HLS array_partition variable = weight_matrix_xf cyclic dim = 2 factor = 32
#pragma HLS array_partition variable = weight_matrix_hi cyclic dim = 2 factor = 32
#pragma HLS array_partition variable = weight_matrix_xi cyclic dim = 2 factor = 32
#pragma HLS array_partition variable = weight_matrix_ho cyclic dim = 2 factor = 32
#pragma HLS array_partition variable = weight_matrix_xo cyclic dim = 2 factor = 32
#pragma HLS array_partition variable = weight_matrix_hg cyclic dim = 2 factor = 32
#pragma HLS array_partition variable = weight_matrix_xg cyclic dim = 2 factor = 32
#pragma HLS array_partition variable = hidden_state_seq cyclic dim = 2 factor = 32

#pragma HLS array_partition variable = bias_f complete factor = 32
#pragma HLS array_partition variable = bias_i complete factor = 32
#pragma HLS array_partition variable = bias_o complete factor = 32
#pragma HLS array_partition variable = bias_c complete factor = 32
// version keep 2 cells 
    LSTM_cell(
        input_seq[0],
		dummy_hidden,
		dummy_cell,
        weight_matrix_hf,
        weight_matrix_xf,
        weight_matrix_hi,
        weight_matrix_xi,
        weight_matrix_ho,
        weight_matrix_xo,
        weight_matrix_hg,
        weight_matrix_xg,
        bias_f,
        bias_i,
        bias_o,
        bias_c,
        hidden_state_seq[0],  // next
        cell_state_seq[0]    // next
    );
LOOP_ALL:
    for (int i = 1; i < INPUT_SEQUENTIAL_D; i+2) {
#pragma HLS dataflow
            // LSTM cell 1
            LSTM_cell(
                input_seq[i],
                hidden_state_seq[i-1],
                cell_state_seq[i-1],
                weight_matrix_hf,
                weight_matrix_xf,
                weight_matrix_hi,
                weight_matrix_xi,
                weight_matrix_ho,
                weight_matrix_xo,
                weight_matrix_hg,
                weight_matrix_xg,
                bias_f,
                bias_i,
                bias_o,
                bias_c,
                hidden_state_seq[i],  // next
                cell_state_seq[i]    // next
            );
            // LSTM cell 2
            LSTM_cell(
                input_seq[i + 1],
                hidden_state_seq[i],
                cell_state_seq[i],
                weight_matrix_hf,
                weight_matrix_xf,
                weight_matrix_hi,
                weight_matrix_xi,
                weight_matrix_ho,
                weight_matrix_xo,
                weight_matrix_hg,
                weight_matrix_xg,
                bias_f,
                bias_i,
                bias_o,
                bias_c,
                hidden_state_seq[i + 1],  // next
                cell_state_seq[i + 1]    // next
            );
    }
    return 0;
}
