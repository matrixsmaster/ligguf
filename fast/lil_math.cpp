/* LiGGUF - a small, dependency-free LLaMA inference engine with direct GGUF support
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025-2026
 * */

#include <assert.h>
#include "common.h"
#include "lil_math.h"

using namespace std;

float fp1632_lut[65536];

static qtensor m_xq8;

// Acceleration stuff
#if defined(__AVX2__)
#include "arch_avx2.h"
#endif
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include "arch_neon.h"
#endif

static inline float fp32_from_bits(uint32_t w)
{
    float r;
    memcpy(&r,&w,4);
    return r;
}

static inline uint32_t fp32_to_bits(float f)
{
    uint32_t r;
    memcpy(&r,&f,4);
    return r;
}

static float fp16_to_fp32_impl(uint16_t h)
{
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));

    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline uint16_t fp32_to_fp16(float f)
{
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));

    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) bias = UINT32_C(0x71000000);

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

void fp1632_init()
{
    for (int i = 0; i < 65536; i++) fp1632_lut[i] = fp16_to_fp32_impl((uint16_t)i);
}

void f32_to_f16_row(const float* x, gguf_half* y, int n)
{
    for (int i = 0; i < n; i++) y[i] = fp32_to_fp16(x[i]);
}

static inline void f16_to_f32_row(const gguf_half* x, float* y, int n)
{
    for (int i = 0; i < n; i++) y[i] = fp16_to_fp32(x[i]);
}

void dequantize_row_q8_0(const block_q8_0* x, float* y, int64_t k)
{
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_0; j++) y[i*QK8_0 + j] = x[i].qs[j] * d;
    }
}

void dequantize_row_q1_0(const block_q1_0* x, float* y, int64_t k)
{
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_0; j++) {
            int b = (x[i].qs[j >> 3] >> (j & 7)) & 1;
            y[i*QK8_0 + j] = d * (b? 1.0f : -1.0f);
        }
    }
}

void dequantize_row_q1_G(const block_q1_G* x, float* y, int64_t k)
{
    const int nb = k / 128;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < 128; j++) {
            int b = (x[i].qs[j >> 3] >> (j & 7)) & 1;
            y[i*128 + j] = d * (b? 1.0f : -1.0f);
        }
    }
}

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4);
    }
}

void dequantize_row_q2_K(const block_q2_K* x, float* y, int64_t k)
{
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);
        const uint8_t* q = x[i].qs;

        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = x[i].scales[is++];
                float dl = d * (sc & 0xF), ml = min * (sc >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                sc = x[i].scales[is++];
                dl = d * (sc & 0xF), ml = min * (sc >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

void dequantize_row_q3_K(const block_q3_K* x, float* y, int64_t k)
{
    // Q3_K stores 256 values as:
    // - 2-bit payload in qs[64]
    // - one extra sign/high-bit plane in hmask[32]
    // - 16 signed scales packed into scales[12]
    // - one fp16 block scale d
    const int nb = k / QK_K;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t* scales = (const int8_t*)aux;

    for (int i = 0; i < nb; i++) {
        const float d_all = fp16_to_fp32(x[i].d);
        const uint8_t* q = x[i].qs;
        const uint8_t* hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; l++) *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4));

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; l++) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

void dequantize_row_q4_K(const block_q4_K* x, float* y, int64_t k)
{
    // Q4_K stores 256 values as 4-bit payloads in qs[128], plus 8 scale/min pairs
    // packed into scales[12]. d scales the values, dmin scales the per-subblock offsets.
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t* q = x[i].qs;
        const float d = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

void dequantize_row_q5_K(const block_q5_K* x, float* y, int64_t k)
{
    // Q5_K is Q4_K plus one extra bit plane in qh[32], so each value is 5 bits
    // with the same packed scale/min layout and shared block scales d/dmin.
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t* ql = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const float d = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

void dequantize_row_q6_K(const block_q6_K* x, float* y, int64_t k)
{
    // Q6_K stores low 4 bits in ql[128], high 2 bits in qh[64], one signed scale
    // per 16 values in scales[16], and a shared fp16 block scale d.
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void dequantize_row(gguf_type t, const void* in, float* out, int64_t n)
{
    switch (t) {
    case F32: memcpy(out,in,n*sizeof(float)); break;
    case F16: f16_to_f32_row((const gguf_half*)in,out,n); break;
    case Q8_0: dequantize_row_q8_0((block_q8_0*)in,out,n); break;
    case Q1_0: dequantize_row_q1_0((block_q1_0*)in,out,n); break;
    case Q1_G: dequantize_row_q1_G((block_q1_G*)in,out,n); break;
    case Q2_K: dequantize_row_q2_K((block_q2_K*)in,out,n); break;
    case Q3_K: dequantize_row_q3_K((block_q3_K*)in,out,n); break;
    case Q4_K: dequantize_row_q4_K((block_q4_K*)in,out,n); break;
    case Q5_K: dequantize_row_q5_K((block_q5_K*)in,out,n); break;
    case Q6_K: dequantize_row_q6_K((block_q6_K*)in,out,n); break;
    default:
        ERR("Unsupported tensor type %d",t);
        abort();
    }
}

float vec_dot_f32(const float* a, const float* b, int size)
{
    return dot_f32(a,b,size);
}

float vec_dot_f16_f32(const gguf_half* a, const float* b, int size)
{
    return dot_f16_f32(a,b,size);
}

void quantize_q8_0(qtensor &y, const ftensor &x)
{
    const int nb = x.size() / QK8_0;
    if ((int)y.size() < nb) y.resize(nb);

    MULTITHREAD
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = max(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = fp32_to_fp16(d);

        for (int j = 0; j < QK8_0; j++)
            y[i].qs[j] = roundf(x[i * QK8_0 + j] * id);
    }
}

static inline int nearest_int(float f)
{
    return f < 0 ? (int)(f - 0.5f) : (int)(f + 0.5f);
}

void quantize_q8_K(ktensor &y, const ftensor &x)
{
    assert(x.size() % QK_K == 0);
    const int nb = x.size() / QK_K;

    MULTITHREAD
    for (int i = 0; i < nb; i++) {
        const float* p = x.data() + i * QK_K;
        float max = 0.0f;
        float amax = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            float ax = fabsf(p[j]);
            if (ax > amax) {
                amax = ax;
                max = p[j];
            }
        }
        if (!amax) {
            y[i].d = 0.0f;
            memset(y[i].qs,0,QK_K);
            memset(y[i].bsums,0,sizeof(y[i].bsums));
            continue;
        }

        const float iscale = -127.0f / max;
        for (int j = 0; j < QK_K; j++) {
            int v = nearest_int(iscale * p[j]);
            y[i].qs[j] = min(127,v);
        }
        for (int j = 0; j < QK_K/16; j++) {
            int sum = 0;
            for (int ii = 0; ii < 16; ii++) sum += y[i].qs[j*16 + ii];
            y[i].bsums[j] = sum;
        }
        y[i].d = 1.0f / iscale;
    }
}

#ifndef ARCH_ACCEL
float inline dot_f32(const float* a, const float* b, int size)
{
    float sum = 0.0f;
    for (int i = 0; i < size; i++) sum += a[i] * b[i];
    return sum;
}

float inline dot_f16_f32(const gguf_half* a, const float* b, int size)
{
    float sum = 0.0f;
    for (int i = 0; i < size; i++) sum += fp16_to_fp32(a[i]) * b[i];
    return sum;
}

float inline dot_q8_0_q8_0(const block_q8_0* x, const block_q8_0* y, int n)
{
    const int nb = n / QK8_0;
    float acc = 0.0f;
    for (int s,j,i = 0; i < nb; i++) {
        s = 0;
        for (j = 0; j < QK8_0; j++) s += x[i].qs[j] * y[i].qs[j];
        acc += fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d) * s;
    }
    return acc;
}

float inline dot_q8_0_q1_0(const block_q8_0* y, const block_q1_0* x, int n)
{
    const int nb = n / QK8_0;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; j++) {
            int xi = ((x[i].qs[j >> 3] >> (j & 7)) & 1)? 1 : -1;
            sumi += xi * y[i].qs[j];
        }
        sumf += fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d) * sumi;
    }

    return sumf;
}

float inline dot_q8_0_q1_G(const block_q8_0* y, const block_q1_G* x, int n)
{
    const int nb = n / 128;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d0 = fp16_to_fp32(x[i].d);
        float sumi = 0.0f;

        for (int k = 0; k < 4; k++) {
            const block_q8_0* yb = y + i * 4 + k;
            int sumib = 0;
            for (int j = 0; j < QK8_0; j++) {
                int bit = k * QK8_0 + j;
                int xi = ((x[i].qs[bit >> 3] >> (bit & 7)) & 1)? 1 : -1;
                sumib += xi * yb->qs[j];
            }
            sumi += fp16_to_fp32(yb->d) * sumib;
        }

        sumf += d0 * sumi;
    }

    return sumf;
}

float inline dot_q8_K_q3_K(const block_q8_K* y, const block_q3_K* x, int n)
{
    // Rebuild the signed Q3 values into aux8[], expand the packed per-16 scales,
    // then accumulate the dot in 16-value chunks against the runtime Q8_K activation.
    const int nb = n / QK_K;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    int8_t aux8[QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    memset(sums,0,sizeof(sums));
    uint32_t auxs[4];
    const int8_t* scales = (const int8_t*)auxs;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t* q3 = x[i].qs;
        const uint8_t* hm = x[i].hmask;
        const int8_t* q8 = y[i].qs;
        memset(aux32,0,sizeof(aux32));
        int8_t* a = aux8;
        uint8_t m = 1;
        // Expand the 2-bit payload plus high-bit mask into signed 3-bit values.
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; l++) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; l++) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32;
            m <<= 1;
            for (int l = 0; l < 32; l++) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; l++) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32;
            m <<= 1;
            for (int l = 0; l < 32; l++) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; l++) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32;
            m <<= 1;
            for (int l = 0; l < 32; l++) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; l++) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32;
            m <<= 1;
            q3 += 32;
        }

        a = aux8;
        memcpy(auxs,x[i].scales,12);

        // Expand the packed sub-block scales to one signed scale per 16 values.
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        for (int j = 0; j < QK_K/16; j++) {
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8;
            a += 8;
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8;
            a += 8;
        }

        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; l++) sums[l] += d * aux32[l];
    }

    for (int l = 0; l < 8; l++) sumf += sums[l];
    return sumf;
}

float inline dot_q8_K_q2_K(const block_q8_K* y, const block_q2_K* x, int n)
{
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t* q2 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        const uint8_t* sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < QK_K/16; j++) summs += y[i].bsums[j] * (sc[j] >> 4);

        const float dall = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = y[i].d * fp16_to_fp32(x[i].dmin);

        int isum = 0;
        int is = 0;
        for (int k = 0; k < QK_K/128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l = 0; l < 16; l++) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;

                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; l++) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;

                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }

        sumf += dall * isum - dmin * summs;
    }

    return sumf;
}

float inline dot_q8_K_q4_K(const block_q8_K* y, const block_q4_K* x, int n)
{
    // Walk the two 32-value halves inside each 64-value chunk, dot the activation
    // against each nibble stream, then apply the chunk scale and min correction.
    const int nb = n / QK_K;
    float acc = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t* q = x[i].qs;
        const int8_t* a = y[i].qs;
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        const float dmin = fp16_to_fp32(x[i].dmin) * y[i].d;
        int is = 0;

        // Each 64-value chunk is two 32-value dots with separate scale/min pairs.
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0,x[i].scales,&sc,&m);
            int dot1 = 0;
            for (int l = 0; l < 32; l++) dot1 += a[l] * (q[l] & 0xF);
            acc += d * sc * dot1 - dmin * m * (y[i].bsums[is*2 + 0] + y[i].bsums[is*2 + 1]);

            get_scale_min_k4(is + 1,x[i].scales,&sc,&m);
            int dot2 = 0;
            for (int l = 0; l < 32; l++) dot2 += a[l + 32] * (q[l] >> 4);
            acc += d * sc * dot2 - dmin * m * (y[i].bsums[is*2 + 2] + y[i].bsums[is*2 + 3]);

            q += 32;
            a += 64;
            is += 2;
        }
    }

    return acc;
}

float inline dot_q8_K_q5_K(const block_q8_K* y, const block_q5_K* x, int n)
{
    // Rebuild the 5-bit payload into aux8[], expand the packed scale/min tables,
    // accumulate scaled dot products, then subtract the min contribution via bsums.
    const int nb = n / QK_K;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    const uint8_t* scales = (const uint8_t*)&utmp[0];
    const uint8_t* mins = (const uint8_t*)&utmp[2];
    int8_t aux8[QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    memset(sums,0,sizeof(sums));
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t* q4 = x[i].qs;
        const uint8_t* hm = x[i].qh;
        const int8_t* q8 = y[i].qs;
        memset(aux32,0,sizeof(aux32));
        int8_t* a = aux8;
        uint8_t m = 1;

        // Rebuild the 5-bit values before applying scale and min corrections.
        for (int j = 0; j < QK_K/64; j++) {
            for (int l = 0; l < 32; l++) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; l++) a[l] += (hm[l] & m ? 16 : 0);
            a += 32;
            m <<= 1;
            for (int l = 0; l < 32; l++) a[l] = (int8_t)(q4[l] >> 4);
            for (int l = 0; l < 32; l++) a[l] += (hm[l] & m ? 16 : 0);
            a += 32;
            m <<= 1;
            q4 += 32;
        }

        memcpy(utmp,x[i].scales,12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; j++) sumi += y[i].bsums[j] * mins[j/2];

        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; j++) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }

        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; l++) sums[l] += d * aux32[l];
        sumf -= fp16_to_fp32(x[i].dmin) * y[i].d * sumi;
    }

    for (int l = 0; l < 8; l++) sumf += sums[l];
    return sumf;
}

float inline dot_q8_K_q6_K(const block_q8_K* y, const block_q6_K* x, int n)
{
    const int nb = n / QK_K;
    int8_t aux8[QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    memset(sums,0,sizeof(sums));
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t* q4 = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* q8 = y[i].qs;
        memset(aux32,0,sizeof(aux32));
        int8_t* a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; l++) {
                a[l + 0] = (int8_t)((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a += 128;
            q4 += 64;
            qh += 32;
        }

        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; j++) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }

        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; l++) sums[l] += d * aux32[l];
    }

    for (int l = 0; l < 8; l++) sumf += sums[l];
    return sumf;
}
#endif /*ARCH_ACCEL*/

// out: destination vector with d rows
// src: float activation to quantize on demand when the weight is Q8_0
// qxk/qx8: pre-quantized activation in Q8_K or Q8_0 form
// w: source weight tensor row layout
// n: input width, d: output width
void matmul(ftensor &out, const ftensor* src, const ktensor* qxk, const qtensor* qx8, wtensor w, int n, int d)
{
    const block_q8_0* q8 = qx8 ? qx8->data() : NULL;
    if ((w.type == Q8_0 || w.type == Q1_0 || w.type == Q1_G) && !q8) {
        assert(src);
        quantize_q8_0(m_xq8,*src);
        q8 = m_xq8.data();
    }

    MULTITHREAD
    for (int r = 0; r < d; r++) {
        uint8_t* wp = w.ptr + r * w.rsz;
        switch (w.type) {
            case F32: out[r] = dot_f32((float*)wp,src->data(),n); break;
            case F16: out[r] = dot_f16_f32((gguf_half*)wp,src->data(),n); break;
            case Q8_0: out[r] = dot_q8_0_q8_0(q8,(block_q8_0*)wp,n); break;
            case Q1_0: out[r] = dot_q8_0_q1_0(q8,(block_q1_0*)wp,n); break;
            case Q1_G: out[r] = dot_q8_0_q1_G(q8,(block_q1_G*)wp,n); break;
            case Q2_K: out[r] = dot_q8_K_q2_K(qxk->data(),(block_q2_K*)wp,n); break;
            case Q3_K: out[r] = dot_q8_K_q3_K(qxk->data(),(block_q3_K*)wp,n); break;
            case Q4_K: out[r] = dot_q8_K_q4_K(qxk->data(),(block_q4_K*)wp,n); break;
            case Q5_K: out[r] = dot_q8_K_q5_K(qxk->data(),(block_q5_K*)wp,n); break;
            case Q6_K: out[r] = dot_q8_K_q6_K(qxk->data(),(block_q6_K*)wp,n); break;
            default:
                ERR("Unsupported matmul type %d",w.type);
                abort();
        }
    }
}

// Same contract as matmul(), but computes two same-shape outputs in one row loop.
void matmul2(ftensor &out0, ftensor &out1, const ftensor* src, const ktensor* qxk, const qtensor* qx8, wtensor w0, wtensor w1, int n, int d)
{
    const block_q8_0* q8 = qx8 ? qx8->data() : NULL;
    if ((w0.type == Q8_0 || w0.type == Q1_0 || w0.type == Q1_G || w1.type == Q8_0 || w1.type == Q1_0 || w1.type == Q1_G) && !q8) {
        assert(src);
        quantize_q8_0(m_xq8,*src);
        q8 = m_xq8.data();
    }

    MULTITHREAD
    for (int r = 0; r < d; r++) {
        uint8_t* wp0 = w0.ptr + r * w0.rsz;
        uint8_t* wp1 = w1.ptr + r * w1.rsz;

        switch (w0.type) {
            case F32: out0[r] = dot_f32((float*)wp0,src->data(),n); break;
            case F16: out0[r] = dot_f16_f32((gguf_half*)wp0,src->data(),n); break;
            case Q8_0: out0[r] = dot_q8_0_q8_0(q8,(block_q8_0*)wp0,n); break;
            case Q1_0: out0[r] = dot_q8_0_q1_0(q8,(block_q1_0*)wp0,n); break;
            case Q1_G: out0[r] = dot_q8_0_q1_G(q8,(block_q1_G*)wp0,n); break;
            case Q2_K: out0[r] = dot_q8_K_q2_K(qxk->data(),(block_q2_K*)wp0,n); break;
            case Q3_K: out0[r] = dot_q8_K_q3_K(qxk->data(),(block_q3_K*)wp0,n); break;
            case Q4_K: out0[r] = dot_q8_K_q4_K(qxk->data(),(block_q4_K*)wp0,n); break;
            case Q5_K: out0[r] = dot_q8_K_q5_K(qxk->data(),(block_q5_K*)wp0,n); break;
            case Q6_K: out0[r] = dot_q8_K_q6_K(qxk->data(),(block_q6_K*)wp0,n); break;
            default:
                ERR("Unsupported matmul type %d",w0.type);
                abort();
        }

        switch (w1.type) {
            case F32: out1[r] = dot_f32((float*)wp1,src->data(),n); break;
            case F16: out1[r] = dot_f16_f32((gguf_half*)wp1,src->data(),n); break;
            case Q8_0: out1[r] = dot_q8_0_q8_0(q8,(block_q8_0*)wp1,n); break;
            case Q1_0: out1[r] = dot_q8_0_q1_0(q8,(block_q1_0*)wp1,n); break;
            case Q1_G: out1[r] = dot_q8_0_q1_G(q8,(block_q1_G*)wp1,n); break;
            case Q2_K: out1[r] = dot_q8_K_q2_K(qxk->data(),(block_q2_K*)wp1,n); break;
            case Q3_K: out1[r] = dot_q8_K_q3_K(qxk->data(),(block_q3_K*)wp1,n); break;
            case Q4_K: out1[r] = dot_q8_K_q4_K(qxk->data(),(block_q4_K*)wp1,n); break;
            case Q5_K: out1[r] = dot_q8_K_q5_K(qxk->data(),(block_q5_K*)wp1,n); break;
            case Q6_K: out1[r] = dot_q8_K_q6_K(qxk->data(),(block_q6_K*)wp1,n); break;
            default:
                ERR("Unsupported matmul type %d",w1.type);
                abort();
        }
    }
}

void matmul3(ftensor &out0, ftensor &out1, ftensor &out2, const ftensor* src, const ktensor* qxk, const qtensor* qx8, wtensor w0, wtensor w1, wtensor w2, int n, int d0, int d1, int d2)
{
    const block_q8_0* q8 = qx8 ? qx8->data() : NULL;
    if ((w0.type == Q8_0 || w0.type == Q1_0 || w0.type == Q1_G || w1.type == Q8_0 || w1.type == Q1_0 || w1.type == Q1_G || w2.type == Q8_0 || w2.type == Q1_0 || w2.type == Q1_G) && !q8) {
        assert(src);
        quantize_q8_0(m_xq8,*src);
        q8 = m_xq8.data();
    }

    int d = d0;
    if (d1 > d) d = d1;
    if (d2 > d) d = d2;

    MULTITHREAD
    for (int r = 0; r < d; r++) {
        if (r < d0) {
            uint8_t* wp0 = w0.ptr + r * w0.rsz;
            switch (w0.type) {
                case F32: out0[r] = dot_f32((float*)wp0,src->data(),n); break;
                case F16: out0[r] = dot_f16_f32((gguf_half*)wp0,src->data(),n); break;
                case Q8_0: out0[r] = dot_q8_0_q8_0(q8,(block_q8_0*)wp0,n); break;
                case Q1_0: out0[r] = dot_q8_0_q1_0(q8,(block_q1_0*)wp0,n); break;
                case Q1_G: out0[r] = dot_q8_0_q1_G(q8,(block_q1_G*)wp0,n); break;
                case Q2_K: out0[r] = dot_q8_K_q2_K(qxk->data(),(block_q2_K*)wp0,n); break;
                case Q3_K: out0[r] = dot_q8_K_q3_K(qxk->data(),(block_q3_K*)wp0,n); break;
                case Q4_K: out0[r] = dot_q8_K_q4_K(qxk->data(),(block_q4_K*)wp0,n); break;
                case Q5_K: out0[r] = dot_q8_K_q5_K(qxk->data(),(block_q5_K*)wp0,n); break;
                case Q6_K: out0[r] = dot_q8_K_q6_K(qxk->data(),(block_q6_K*)wp0,n); break;
                default:
                    ERR("Unsupported matmul type %d",w0.type);
                    abort();
            }
        }

        if (r < d1) {
            uint8_t* wp1 = w1.ptr + r * w1.rsz;
            switch (w1.type) {
                case F32: out1[r] = dot_f32((float*)wp1,src->data(),n); break;
                case F16: out1[r] = dot_f16_f32((gguf_half*)wp1,src->data(),n); break;
                case Q8_0: out1[r] = dot_q8_0_q8_0(q8,(block_q8_0*)wp1,n); break;
                case Q1_0: out1[r] = dot_q8_0_q1_0(q8,(block_q1_0*)wp1,n); break;
                case Q1_G: out1[r] = dot_q8_0_q1_G(q8,(block_q1_G*)wp1,n); break;
                case Q2_K: out1[r] = dot_q8_K_q2_K(qxk->data(),(block_q2_K*)wp1,n); break;
                case Q3_K: out1[r] = dot_q8_K_q3_K(qxk->data(),(block_q3_K*)wp1,n); break;
                case Q4_K: out1[r] = dot_q8_K_q4_K(qxk->data(),(block_q4_K*)wp1,n); break;
                case Q5_K: out1[r] = dot_q8_K_q5_K(qxk->data(),(block_q5_K*)wp1,n); break;
                case Q6_K: out1[r] = dot_q8_K_q6_K(qxk->data(),(block_q6_K*)wp1,n); break;
                default:
                    ERR("Unsupported matmul type %d",w1.type);
                    abort();
            }
        }

        if (r < d2) {
            uint8_t* wp2 = w2.ptr + r * w2.rsz;
            switch (w2.type) {
                case F32: out2[r] = dot_f32((float*)wp2,src->data(),n); break;
                case F16: out2[r] = dot_f16_f32((gguf_half*)wp2,src->data(),n); break;
                case Q8_0: out2[r] = dot_q8_0_q8_0(q8,(block_q8_0*)wp2,n); break;
                case Q1_0: out2[r] = dot_q8_0_q1_0(q8,(block_q1_0*)wp2,n); break;
                case Q1_G: out2[r] = dot_q8_0_q1_G(q8,(block_q1_G*)wp2,n); break;
                case Q2_K: out2[r] = dot_q8_K_q2_K(qxk->data(),(block_q2_K*)wp2,n); break;
                case Q3_K: out2[r] = dot_q8_K_q3_K(qxk->data(),(block_q3_K*)wp2,n); break;
                case Q4_K: out2[r] = dot_q8_K_q4_K(qxk->data(),(block_q4_K*)wp2,n); break;
                case Q5_K: out2[r] = dot_q8_K_q5_K(qxk->data(),(block_q5_K*)wp2,n); break;
                case Q6_K: out2[r] = dot_q8_K_q6_K(qxk->data(),(block_q6_K*)wp2,n); break;
                default:
                    ERR("Unsupported matmul type %d",w2.type);
                    abort();
            }
        }
    }
}

void rope(ftensor& x, int n_heads, int pos, const gguf_model &mdl, const ftensor &rope_freq)
{
    assert((int)x.size() == n_heads * mdl.head_dim);
    int rope_dim = mdl.rope_dim;
    if (rope_dim > mdl.head_dim) rope_dim = mdl.head_dim;
    if (rope_dim & 1) rope_dim--;

    for (int h = 0; h < n_heads; h++) {
        float* v = x.data() + h * mdl.head_dim;
        for (int i = 0; i < rope_dim; i += 2) {
            const int m = i >> 1;
            const float ang = pos * rope_freq[m];
            const float c = cosf(ang), s = sinf(ang);
            const float x0 = v[i + 0];
            const float x1 = v[i + 1];
            v[i + 0] = x0 * c - x1 * s;
            v[i + 1] = x0 * s + x1 * c;
        }
    }
}

void rmsnorm(ftensor &out, const ftensor &x, float* w, int size, float epsilon)
{
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = ss / (float)size + epsilon;
    ss = 1.0f / sqrtf(ss);
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * w[i];
}

void rmsnorm_inplace(float* x, float* w, int size, float epsilon)
{
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / (float)size + epsilon);
    for (int i = 0; i < size; i++) x[i] = x[i] * ss * w[i];
}

void l2norm(ftensor &x)
{
    float ss = 0.0f;
    for (int i = 0; i < (int)x.size(); i++) ss += x[i] * x[i];
    if (ss <= 0.0f) return;
    ss = 1.0f / sqrtf(ss);
    for (int i = 0; i < (int)x.size(); i++) x[i] *= ss;
}

void layernorm(ftensor &out, const ftensor &x, const float* w, const float* b, int size, float epsilon)
{
    double mean = 0.0;
    double var = 0.0;
    for (int i = 0; i < size; i++) mean += x[i];
    mean /= size;
    for (int i = 0; i < size; i++) {
        const double d = x[i] - mean;
        var += d * d;
    }
    var /= size;

    const float scale = 1.0f / sqrtf(var + epsilon);
    for (int i = 0; i < size; i++) out[i] = (x[i] - mean) * scale * w[i] + b[i];
}

void softmax(float* x, int size)
{
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < size; i++) x[i] /= sum;
}

void gelu(ftensor &x)
{
    for (int i = 0; i < (int)x.size(); i++) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}
