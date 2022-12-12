#include <stdint.h>

#include "ap_int.h"
#include "ap_fixed.h"
#include "lstm.hpp"

#define INPUT_SEQUENTIAL_D 120
#define INPUT_FEATURE_D 7
#define HIDDEN_FEATURE_D 32
#define PARTITION_DIM
#define PARTITION_FACTOR
// naiive 
t_feature sigmoid(t_feature input){
    if (input > 6) 
        return 1.0;
    if (input < -6)
        return 0.0;
    else 
        return sigmoid_lut[(int)(input+6)*1024];
}

t_feature tanh(t_feature input){
    if (input > 4)
        return 1.0;
    if (input < -4)
        return -1.0;
    else
        return tanh_lut[(int)(input+4)*1024];
}

void input_to_hidden_dot_product(t_feature x[INPUT_FEATURE_D], t_feature W[INPUT_FEATURE_D][HIDDEN_FEATURE_D], t_feature output[HIDDEN_FEATURE_D]) {
    for (int i = 0; i < INPUT_FEATURE_D; ++i){
        for (int j = 0; j < HIDDEN_FEATURE_D; ++j){
            output[j] += x[i] * W[i][j];
        }
    }
}

void hidden_to_hidden_dot_product(t_feature h[HIDDEN_FEATURE_D], t_feature U[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D], t_feature output[HIDDEN_FEATURE_D]) {
    for (int i = 0; i < HIDDEN_FEATURE_D; ++i){
        for (int j = 0; j < HIDDEN_FEATURE_D; ++j){
            output[j] += h[i][j] * U[i][j];
        }
    }
}
/*
    input gate/ forget gate/ output gate share the same function 
*/
void unified_gate(t_feature W[INPUT_FEATURE_D][HIDDEN_FEATURE_D], t_feature x[INPUT_FEATURE_D], \
t_feature U[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D], t_feature h[HIDDEN_FEATURE_D], \
t_feature bias[HIDDEN_FEATURE_D],\
t_feature output[HIDDEN_FEATURE_D]){
    t_feature output_x[HIDDEN_FEATURE_D];
    t_feature output_h[HIDDEN_FEATURE_D];

    input_to_hidden_dot_product(x, W, output_x);
    hidden_to_hidden_dot_product(h, U, output_h);

    for (int i = 0; i < HIDDEN_FEATURE_D; ++i){
    	t_feature tmp_sum = output_x[i] + output_h[i] + bias[i];
        output[i] = sigmoid(tmp_sum);
    }
}


void carry_gate(t_feature W[INPUT_FEATURE_D][HIDDEN_FEATURE_D], t_feature x[INPUT_FEATURE_D], \
t_feature U[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D], t_feature h[HIDDEN_FEATURE_D], \
t_feature bias[HIDDEN_FEATURE_D],\
t_feature output[HIDDEN_FEATURE_D]){
    t_feature output_x[HIDDEN_FEATURE_D];
    t_feature output_h[HIDDEN_FEATURE_D];

    input_to_hidden_dot_product(x, W, output_x);
    hidden_to_hidden_dot_product(h, U, output_h);

    for (int i = 0; i < HIDDEN_FEATURE_D; ++i){
    	t_feature tmp_sum = output_x[i] + output_h[i] + bias[i];
        output[i] = tanh(tmp_sum);
    }
 }
 
void carry_out(t_feature forget[HIDDEN_FEATURE_D], t_feature last_c[HIDDEN_FEATURE_D], t_feature input_t[HIDDEN_FEATURE_D], t_feature carry_t_hat[HIDDEN_FEATURE_D], \
t_feature output[HIDDEN_FEATURE_D]){
    for (int i = 0; i < HIDDEN_FEATURE_D; ++i){
        output[i] = forget[i]*last_c[i] + input_t[i]*carry_t_hat[i];
    }
}

void next_hidden_state(t_feature output_t[HIDDEN_FEATURE_D], t_feature carry_out_t[HIDDEN_FEATURE_D], t_feature output[HIDDEN_FEATURE_D]){
    for (int i = 0; i < HIDDEN_FEATURE_D; ++i){
        output[i] = output_t[i] * tanh(carry_out_t[i]);
    }
}


// 32 unit 
static void LSTM_cell(t_feature input[INPUT_FEATURE_D], \
t_feature hidden_state[HIDDEN_FEATURE_D], \
t_weight weight_matrix_hf[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D], \
t_weight weight_matrix_xf[INPUT_FEATURE_D][HIDDEN_FEATURE_D], \
t_weight weight_matrix_hi[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D], \
t_weight weight_matrix_xi[INPUT_FEATURE_D][HIDDEN_FEATURE_D], \
t_weight weight_matrix_ho[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D], \
t_weight weight_matrix_xo[INPUT_FEATURE_D][HIDDEN_FEATURE_D], \
t_weight weight_matrix_hc[HIDDEN_FEATURE_D][HIDDEN_FEATURE_D], \
t_weight weight_matrix_xc[INPUT_FEATURE_D][HIDDEN_FEATURE_D], \
t_weight bias_f[HIDDEN_FEATURE_D], \
t_weight bias_i[HIDDEN_FEATURE_D], \
t_weight bias_o[HIDDEN_FEATURE_D], \
t_weight bias_c[HIDDEN_FEATURE_D]
){
    // input gate
    unified_gate();
    // forget gate
    unified_gate();
    // carry_out gate
    carry_gate();
    // output gate
    unified_gate();
    // carry out
    carry_out();
    // next_hidden state
    next_hidden_state();
}

// wrapper lstm


int wrapper_flow(t_feature input[INPUT_ROW_D][INPUT_COL_D][INPUT__SEQUENTIAL_D], \
t_feature hidden_state[HIDDEN_ROW_D][HIDDEN_COL_D][INPUT__SEQUENTIAL_D], \
t_weight weight_matrix_f[INPUT_ROW_D][INPUT_COL_D][INPUT__SEQUENTIAL_D], \
t_weight weight_matrix_o[INPUT_ROW_D][INPUT_COL_D][INPUT__SEQUENTIAL_D], \
t_weight weight_matrix_c[INPUT_ROW_D][INPUT_COL_D][INPUT__SEQUENTIAL_D])
{
    #pragma HLS dataflow 
    #pragma HLS array_partition variable=weight_matrix_f cyclic dim=1 factor=16
    #pragma HLS array_partition variable=weight_matrix_o cyclic dim=1 factor=16
    #pragma HLS array_partition variable=weight_matrix_c cyclic dim=1 factor=16
    #pragma HLS array_partition variable=hidden_state cyclic dim=1 factor=16




}
