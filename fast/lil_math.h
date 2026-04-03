#pragma once

#include "lil_gguf.h"

#define fp16_to_fp32(H) (fp1632_lut[H])

extern float fp1632_lut[65536];

void f32_to_f16_row(const float* x, gguf_half* y, int n);

void fp1632_init();

void dequantize_row_q8_0(const block_q8_0* x, float* y, int64_t k);
void dequantize_row_q1_0(const block_q1_0* x, float* y, int64_t k);
void dequantize_row_q1_G(const block_q1_G* x, float* y, int64_t k);
void dequantize_row_q2_K(const block_q2_K* x, float* y, int64_t k);
void dequantize_row_q3_K(const block_q3_K* x, float* y, int64_t k);
void dequantize_row_q4_K(const block_q4_K* x, float* y, int64_t k);
void dequantize_row_q5_K(const block_q5_K* x, float* y, int64_t k);
void dequantize_row_q6_K(const block_q6_K* x, float* y, int64_t k);
void dequantize_row(gguf_type t, const void* in, float* out, int64_t n);

float vec_dot_f32(const float* a, const float* b, int size);
float vec_dot_f16_f32(const gguf_half* a, const float* b, int size);

void quantize_q8_0(qtensor &y, const ftensor &x);
void quantize_q8_K(ktensor &y, const ftensor &x);

void matmul(ftensor &out, const ftensor* src, const ktensor* qxk, const qtensor* qx8, wtensor w, int n, int d);
void matmul2(ftensor &out0, ftensor &out1, const ftensor* src, const ktensor* qxk, const qtensor* qx8, wtensor w0, wtensor w1, int n, int d);
void matmul3(ftensor &out0, ftensor &out1, ftensor &out2, const ftensor* src, const ktensor* qxk, const qtensor* qx8, wtensor w0, wtensor w1, wtensor w2, int n, int d0, int d1, int d2);

void rope(ftensor& x, int n_heads, int pos, const gguf_model &mdl, const ftensor &rope_freq);
void rmsnorm(ftensor &out, const ftensor &x, float* w, int size, float epsilon);
void rmsnorm_inplace(float* x, float* w, int size, float epsilon);
void layernorm(ftensor &out, const ftensor &x, const float* w, const float* b, int size, float epsilon);
void l2norm(ftensor &x);
void softmax(float* x, int size);
void gelu(ftensor &x);
