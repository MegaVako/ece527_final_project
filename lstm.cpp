//#include <stdint.h>

#include "ap_int.h"
#include "ap_fixed.h"
#include "lstm.hpp"


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

#define I_TO_H_UNROLL_RATIO 8
void input_to_hidden_dot_product(t_feature x[INPUT_FEATURE_D], t_weight W[INPUT_FEATURE_D][HIDDEN_FEATURE_D], t_feature output[HIDDEN_FEATURE_D])
{
	I_TO_H : for (int i = 0; i < INPUT_FEATURE_D; ++i)
	{
#pragma HLS PIPELINE
		I_TO_2 : for (int j = 0; j < HIDDEN_FEATURE_D / I_TO_H_UNROLL_RATIO; ++j)
		{
    		I_TO_H_3 : for (int k = 0; k < I_TO_H_UNROLL_RATIO; k++) {
#pragma HLS UNROLL
    			output[j*I_TO_H_UNROLL_RATIO + k] += x[i] * W[i][j*I_TO_H_UNROLL_RATIO + k];
    		}
        }
    }
}

#define H_TO_H_UNROLL_RATIO 8
void hidden_to_hidden_dot_product(t_feature h[HIDDEN_FEATURE_D], t_weight U[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D], t_feature output[HIDDEN_FEATURE_D])
{
	H_TO_H : for (int j = 0; j < HIDDEN_FEATURE_D; ++j)
    {
#pragma HLS PIPELINE rewind
		H_TO_H_2 : for (int i = 0; i < HIDDEN_FEATURE_D / H_TO_H_UNROLL_RATIO; ++i)
        {
			H_TO_H_3 : for (int k = 0; k < H_TO_H_UNROLL_RATIO; k++) {
#pragma HLS UNROLL
	            output[i*H_TO_H_UNROLL_RATIO + k] += h[i*H_TO_H_UNROLL_RATIO + k] * U[i][j];
			}
        }
    }
}
/*
    input gate/ forget gate/ output gate share the same function
*/
void unified_gate(t_weight W[INPUT_FEATURE_D][HIDDEN_FEATURE_D], t_feature x[INPUT_FEATURE_D],
		t_weight U[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D], t_feature h[HIDDEN_FEATURE_D],
		t_weight bias[HIDDEN_FEATURE_D],
                  t_feature output[HIDDEN_FEATURE_D])
{
    t_feature output_x[HIDDEN_FEATURE_D];
    t_feature output_h[HIDDEN_FEATURE_D];

#pragma HLS array_partition variable = output_h cyclic factor = 8
#pragma HLS array_partition variable = output_x cyclic factor = 8

    input_to_hidden_dot_product(x, W, output_x);
    hidden_to_hidden_dot_product(h, U, output_h);

    UNI_GATE : for (int i = 0; i < HIDDEN_FEATURE_D; ++i)
    {
#pragma HLS UNROLL factor = 8
        t_feature tmp_sum = output_x[i] + output_h[i] + bias[i];
        output[i] = sigmoid(tmp_sum);
    }
}

void g_gate(t_weight W[INPUT_FEATURE_D][HIDDEN_FEATURE_D], t_feature x[INPUT_FEATURE_D],
		t_weight U[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D], t_feature h[HIDDEN_FEATURE_D],
		t_weight bias[HIDDEN_FEATURE_D],
                t_feature output[HIDDEN_FEATURE_D])
{
    t_feature output_x[HIDDEN_FEATURE_D];
    t_feature output_h[HIDDEN_FEATURE_D];

#pragma HLS array_partition variable = output_h cyclic factor = 8
#pragma HLS array_partition variable = output_x cyclic factor = 8

    input_to_hidden_dot_product(x, W, output_x);
    hidden_to_hidden_dot_product(h, U, output_h);

    G_GATE : for (int i = 0; i < HIDDEN_FEATURE_D; ++i)
    {
#pragma HLS UNROLL factor = 8
        t_feature tmp_sum = output_x[i] + output_h[i] + bias[i];
        output[i] = tanh(tmp_sum);
    }
}

void cell_out(t_feature forget[HIDDEN_FEATURE_D], t_feature last_c[HIDDEN_FEATURE_D], t_feature input_t[HIDDEN_FEATURE_D], t_feature g_gate[HIDDEN_FEATURE_D],
               t_feature output[HIDDEN_FEATURE_D])
{
    CELL_OUT : for (int i = 0; i < HIDDEN_FEATURE_D; ++i)
    {
#pragma HLS UNROLL
        output[i] = forget[i] * last_c[i] + input_t[i] * g_gate[i];
    }
}

void next_hidden(t_feature output_t[HIDDEN_FEATURE_D], t_feature cell_out_t[HIDDEN_FEATURE_D], t_feature output[HIDDEN_FEATURE_D])
{
	NEXT_HIDDEN : for (int i = 0; i < HIDDEN_FEATURE_D; ++i)
    {
#pragma HLS UNROLL
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

#pragma HLS array_partition variable = output_i cyclic factor = 8
#pragma HLS array_partition variable = output_f cyclic factor = 8
#pragma HLS array_partition variable = output_g cyclic factor = 8
#pragma HLS array_partition variable = output_o cyclic factor = 8

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
#define CELL_UNROLL_RATIO 2
int wrapper_flow(t_feature input_seq[INPUT_SEQUENTIAL_D][INPUT_FEATURE_D],
                 t_feature hidden_state_seq[INPUT_SEQUENTIAL_D][HIDDEN_FEATURE_D])
{

    t_feature cell_state_seq[INPUT_SEQUENTIAL_D][HIDDEN_FEATURE_D];
    t_feature dummy_hidden[32] = {0};
    t_feature dummy_cell[32] = {0};

#pragma HLS array_partition variable = weight_matrix_hf cyclic dim = 2 factor = 8
#pragma HLS array_partition variable = weight_matrix_xf cyclic dim = 2 factor = 8
#pragma HLS array_partition variable = weight_matrix_hi cyclic dim = 2 factor = 8
#pragma HLS array_partition variable = weight_matrix_xi cyclic dim = 2 factor = 8
#pragma HLS array_partition variable = weight_matrix_ho cyclic dim = 2 factor = 8
#pragma HLS array_partition variable = weight_matrix_xo cyclic dim = 2 factor = 8
#pragma HLS array_partition variable = weight_matrix_hg cyclic dim = 2 factor = 8
#pragma HLS array_partition variable = weight_matrix_xg cyclic dim = 2 factor = 8
#pragma HLS array_partition variable = hidden_state_seq cyclic dim = 2 factor = 8

#pragma HLS array_partition variable = bias_f cyclic factor = 8
#pragma HLS array_partition variable = bias_i cyclic factor = 8
#pragma HLS array_partition variable = bias_o cyclic factor = 8
#pragma HLS array_partition variable = bias_c cyclic factor = 8

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
    LSTM_cell(
        input_seq[1],
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
        hidden_state_seq[2],  // next
        cell_state_seq[2]    // next
    );
	CELL_LOOP_INNER : for (int i = 2; i < INPUT_SEQUENTIAL_D; i++) {
		int curr_idx = i - 1;
		int next_idx = curr_idx + 1;
		LSTM_cell(
			input_seq[curr_idx],
			hidden_state_seq[curr_idx],
			cell_state_seq[curr_idx],
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
			hidden_state_seq[next_idx],  // next
			cell_state_seq[next_idx]    // next
		);
	}
//    CELL_LOOP_OUTER : for (int j = 1; j < INPUT_SEQUENTIAL_D / CELL_UNROLL_RATIO; j++) {
//		CELL_LOOP_INNER : for (int i = 0; i < CELL_UNROLL_RATIO; i++) {
//			int curr_idx = i + (j*CELL_UNROLL_RATIO) - 1;
//			int next_idx = curr_idx + 1;
//			LSTM_cell(
//				input_seq[curr_idx],
//				hidden_state_seq[curr_idx],
//				cell_state_seq[curr_idx],
//				weight_matrix_hf,
//				weight_matrix_xf,
//				weight_matrix_hi,
//				weight_matrix_xi,
//				weight_matrix_ho,
//				weight_matrix_xo,
//				weight_matrix_hg,
//				weight_matrix_xg,
//				bias_f,
//				bias_i,
//				bias_o,
//				bias_c,
//				hidden_state_seq[next_idx],  // next
//				cell_state_seq[next_idx]    // next
//			);
//		}
//    }
    return 0;
}
