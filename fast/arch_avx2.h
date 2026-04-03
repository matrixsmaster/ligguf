/* LiGGUF - a small, dependency-free LLaMA inference engine with direct GGUF support
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025-2026
 * */

#include <immintrin.h>

#define ARCH_ACCEL

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

static inline float hsum_float_8(__m256 x)
{
    __m128 res = _mm256_extractf128_ps(x,1);
    res = _mm_add_ps(res,_mm256_castps256_ps128(x));
    res = _mm_add_ps(res,_mm_movehl_ps(res,res));
    res = _mm_add_ss(res,_mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline __m256 mul_sum_i8_pairs_float(__m256i x, __m256i y)
{
    __m256i ax = _mm256_sign_epi8(x,x);
    __m256i sy = _mm256_sign_epi8(y,x);
    __m256i dot = _mm256_maddubs_epi16(ax,sy);
    __m256i ones = _mm256_set1_epi16(1);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(ones,dot));
}

static inline __m256i get_scale_shuffle_q3k(int i)
{
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

static inline __m256i get_scale_shuffle_k4(int i)
{
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

static inline __m128i get_scale_shuffle(int i)
{
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}

static inline __m256i q1_0_bits_to_bytes_32(const uint8_t* qs)
{
    static const uint32_t lut[16] = {
        0xFFFFFFFFu, 0xFFFFFF01u, 0xFFFF01FFu, 0xFFFF0101u,
        0xFF01FFFFu, 0xFF01FF01u, 0xFF0101FFu, 0xFF010101u,
        0x01FFFFFFu, 0x01FFFF01u, 0x01FF01FFu, 0x01FF0101u,
        0x0101FFFFu, 0x0101FF01u, 0x010101FFu, 0x01010101u,
    };
    uint32_t out[8];
    for (int i = 0; i < 8; i++) out[i] = lut[(qs[i >> 1] >> ((i & 1) * 4)) & 0xF];
    return _mm256_loadu_si256((const __m256i*)out);
}

float inline dot_f32(const float* a, const float* b, int size)
{
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(av,bv,acc);
    }
    float sum = hsum_float_8(acc);
    for (; i < size; i++) sum += a[i] * b[i];
    return sum;
}

float inline dot_f16_f32(const gguf_half* a, const float* b, int size)
{
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 av = _mm256_set_ps(fp16_to_fp32(a[i+7]),fp16_to_fp32(a[i+6]),fp16_to_fp32(a[i+5]),fp16_to_fp32(a[i+4]),fp16_to_fp32(a[i+3]),fp16_to_fp32(a[i+2]),fp16_to_fp32(a[i+1]),fp16_to_fp32(a[i+0]));
        __m256 bv = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(av,bv,acc);
    }
    float sum = hsum_float_8(acc);
    for (; i < size; i++) sum += fp16_to_fp32(a[i]) * b[i];
    return sum;
}

float inline dot_q8_0_q8_0(const block_q8_0* x, const block_q8_0* y, int n)
{
    const int nb = n / QK8_0;
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        __m256 d = _mm256_set1_ps(fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d));
        __m256i qx = _mm256_loadu_si256((const __m256i*)x[i].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i*)y[i].qs);
        acc = _mm256_fmadd_ps(d,mul_sum_i8_pairs_float(qx,qy),acc);
    }

    return hsum_float_8(acc);
}

float inline dot_q8_0_q1_0(const block_q8_0* y, const block_q1_0* x, int n)
{
    const int nb = n / QK8_0;
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        __m256 d = _mm256_set1_ps(fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d));
        __m256i qx = q1_0_bits_to_bytes_32(x[i].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i*)y[i].qs);
        acc = _mm256_fmadd_ps(d,mul_sum_i8_pairs_float(qx,qy),acc);
    }

    return hsum_float_8(acc);
}

float inline dot_q8_0_q1_G(const block_q8_0* y, const block_q1_G* x, int n)
{
    const int nb = n / 128;
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        __m256 d0 = _mm256_set1_ps(fp16_to_fp32(x[i].d));
        for (int k = 0; k < 4; k++) {
            const block_q8_0* yi = y + i * 4 + k;
            __m256 d = _mm256_mul_ps(d0,_mm256_set1_ps(fp16_to_fp32(yi->d)));
            __m256i qx = q1_0_bits_to_bytes_32(x[i].qs + k * 4);
            __m256i qy = _mm256_loadu_si256((const __m256i*)yi->qs);
            acc = _mm256_fmadd_ps(d,mul_sum_i8_pairs_float(qx,qy),acc);
        }
    }

    return hsum_float_8(acc);
}

float inline dot_q8_K_q2_K(const block_q8_K* y, const block_q2_K* x, int n)
{
    const int nb = n / QK_K;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);
        const uint8_t* q2 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        __m128i scales8 = _mm_and_si128(mins_and_scales,m4);
        __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales,4),m4);
        __m256i mins = _mm256_cvtepi8_epi16(mins8);
        __m256i prod = _mm256_madd_epi16(mins,_mm256_loadu_si256((const __m256i*)y[i].bsums));
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin),_mm256_cvtepi32_ps(prod),acc);

        __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
        __m128i l_scales = _mm256_extracti128_si256(all_scales,0);
        __m128i h_scales = _mm256_extracti128_si256(all_scales,1);
        __m256i scales[2] = { MM256_SET_M128I(l_scales,l_scales), MM256_SET_M128I(h_scales,h_scales) };
        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/128; j++) {
            __m256i q2bits = _mm256_loadu_si256((const __m256i*)q2); q2 += 32;
            __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i q2_0 = _mm256_and_si256(q2bits,m3);
            __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits,2),m3);
            __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits,4),m3);
            __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits,6),m3);
            __m256i p0 = _mm256_maddubs_epi16(q2_0,q8_0);
            __m256i p1 = _mm256_maddubs_epi16(q2_1,q8_1);
            __m256i p2 = _mm256_maddubs_epi16(q2_2,q8_2);
            __m256i p3 = _mm256_maddubs_epi16(q2_3,q8_3);
            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j],get_scale_shuffle_q3k(0)),p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j],get_scale_shuffle_q3k(1)),p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j],get_scale_shuffle_q3k(2)),p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j],get_scale_shuffle_q3k(3)),p3);
            p0 = _mm256_add_epi32(p0,p1);
            p2 = _mm256_add_epi32(p2,p3);
            sumi = _mm256_add_epi32(sumi,_mm256_add_epi32(p0,p2));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d),_mm256_cvtepi32_ps(sumi),acc);
    }

    return hsum_float_8(acc);
}

float inline dot_q8_K_q3_K(const block_q8_K* y, const block_q3_K* x, int n)
{
    const int nb = n / QK_K;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);
    __m256 acc = _mm256_setzero_ps();
    uint32_t aux[3];

    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const uint8_t* q3 = x[i].qs;
        const int8_t* q8 = y[i].qs;

        memcpy(aux,x[i].scales,12);
        __m128i scales128 = _mm_set_epi32(((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4), ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4), (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4), (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128,m32);
        __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
        __m128i l_scales = _mm256_extracti128_si256(all_scales,0);
        __m128i h_scales = _mm256_extracti128_si256(all_scales,1);
        __m256i scales[2] = { MM256_SET_M128I(l_scales,l_scales), MM256_SET_M128I(h_scales,h_scales) };
        __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].hmask);
        __m256i sumi = _mm256_setzero_si256();
        int bit = 0;

        for (int j = 0; j < QK_K/128; j++) {
            __m256i q3bits = _mm256_loadu_si256((const __m256i*)q3); q3 += 32;
            __m256i q3l_0 = _mm256_and_si256(q3bits,m3);
            __m256i q3h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits,_mm256_slli_epi16(mone,bit)),bit),2); bit++;
            __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits,2),m3);
            __m256i q3h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits,_mm256_slli_epi16(mone,bit)),bit),2); bit++;
            __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits,4),m3);
            __m256i q3h_2 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits,_mm256_slli_epi16(mone,bit)),bit),2); bit++;
            __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits,6),m3);
            __m256i q3h_3 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits,_mm256_slli_epi16(mone,bit)),bit),2); bit++;
            __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16_0 = _mm256_sub_epi16(_mm256_maddubs_epi16(q3l_0,q8_0),_mm256_maddubs_epi16(q3h_0,q8_0));
            __m256i p16_1 = _mm256_sub_epi16(_mm256_maddubs_epi16(q3l_1,q8_1),_mm256_maddubs_epi16(q3h_1,q8_1));
            __m256i p16_2 = _mm256_sub_epi16(_mm256_maddubs_epi16(q3l_2,q8_2),_mm256_maddubs_epi16(q3h_2,q8_2));
            __m256i p16_3 = _mm256_sub_epi16(_mm256_maddubs_epi16(q3l_3,q8_3),_mm256_maddubs_epi16(q3h_3,q8_3));
            p16_0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j],get_scale_shuffle_q3k(0)),p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j],get_scale_shuffle_q3k(1)),p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j],get_scale_shuffle_q3k(2)),p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j],get_scale_shuffle_q3k(3)),p16_3);
            sumi = _mm256_add_epi32(sumi,_mm256_add_epi32(_mm256_add_epi32(p16_0,p16_1),_mm256_add_epi32(p16_2,p16_3)));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d),_mm256_cvtepi32_ps(sumi),acc);
    }

    return hsum_float_8(acc);
}

float inline dot_q8_K_q4_K(const block_q8_K* y, const block_q4_K* x, int n)
{
    const int nb = n / QK_K;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);
        memcpy(utmp,x[i].scales,12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t* q4 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3],utmp[2],utmp[1],utmp[0]));
        __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums,0),_mm256_extracti128_si256(q8sums,1));
        __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales,1),q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin),_mm_cvtepi32_ps(prod),acc_m);
        __m128i sc128 = _mm256_extracti128_si256(mins_and_scales,0);
        __m256i scales = MM256_SET_M128I(sc128,sc128);
        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/64; j++) {
            __m256i scale_l = _mm256_shuffle_epi8(scales,get_scale_shuffle_k4(2*j+0));
            __m256i scale_h = _mm256_shuffle_epi8(scales,get_scale_shuffle_k4(2*j+1));
            __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            __m256i q4l = _mm256_and_si256(q4bits,m4);
            __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits,4),m4);
            __m256i q8l = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i q8h = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16l = _mm256_madd_epi16(scale_l,_mm256_maddubs_epi16(q4l,q8l));
            __m256i p16h = _mm256_madd_epi16(scale_h,_mm256_maddubs_epi16(q4h,q8h));
            sumi = _mm256_add_epi32(sumi,_mm256_add_epi32(p16l,p16h));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d),_mm256_cvtepi32_ps(sumi),acc);
    }

    acc_m = _mm_add_ps(acc_m,_mm_movehl_ps(acc_m,acc_m));
    acc_m = _mm_add_ss(acc_m,_mm_movehdup_ps(acc_m));
    return hsum_float_8(acc) + _mm_cvtss_f32(acc_m);
}

float inline dot_q8_K_q5_K(const block_q8_K* y, const block_q5_K* x, int n)
{
    const int nb = n / QK_K;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i mzero = _mm_setzero_si128();
    __m256 acc = _mm256_setzero_ps();
    float summs = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t* q5 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);

        memcpy(utmp,x[i].scales,12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3],utmp[2],utmp[1],utmp[0]));
        __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums,0),_mm256_extracti128_si256(q8sums,1));
        __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales,1),q8s);
        __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod,mzero),mzero);
        summs += dmin * _mm_extract_epi32(hsum,0);

        __m128i sc128 = _mm256_extracti128_si256(mins_and_scales,0);
        __m256i scales = MM256_SET_M128I(sc128,sc128);
        __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].qh);
        __m256i hmask = mone;
        __m256i sumi = _mm256_setzero_si256();
        int bit = 0;

        for (int j = 0; j < QK_K/64; j++) {
            __m256i scale_0 = _mm256_shuffle_epi8(scales,get_scale_shuffle_k4(2*j+0));
            __m256i scale_1 = _mm256_shuffle_epi8(scales,get_scale_shuffle_k4(2*j+1));
            __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5); q5 += 32;
            __m256i q5_0 = _mm256_add_epi8(_mm256_and_si256(q5bits,m4),_mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits,hmask),bit++),4));
            hmask = _mm256_slli_epi16(hmask,1);
            __m256i q5_1 = _mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16(q5bits,4),m4),_mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits,hmask),bit++),4));
            hmask = _mm256_slli_epi16(hmask,1);
            __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16_0 = _mm256_madd_epi16(scale_0,_mm256_maddubs_epi16(q5_0,q8_0));
            __m256i p16_1 = _mm256_madd_epi16(scale_1,_mm256_maddubs_epi16(q5_1,q8_1));
            sumi = _mm256_add_epi32(sumi,_mm256_add_epi32(p16_0,p16_1));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d),_mm256_cvtepi32_ps(sumi),acc);
    }

    return hsum_float_8(acc) + summs;
}

float inline dot_q8_K_q6_K(const block_q8_K* y, const block_q6_K* x, int n)
{
    const int nb = n / QK_K;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(3);
    const __m256i m32s = _mm256_set1_epi8(32);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const uint8_t* q4 = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* q8 = y[i].qs;
        __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        __m256i sumi = _mm256_setzero_si256();
        int is = 0;

        for (int j = 0; j < QK_K/128; j++) {
            __m128i scale_0 = _mm_shuffle_epi8(scales,get_scale_shuffle(is + 0));
            __m128i scale_1 = _mm_shuffle_epi8(scales,get_scale_shuffle(is + 1));
            __m128i scale_2 = _mm_shuffle_epi8(scales,get_scale_shuffle(is + 2));
            __m128i scale_3 = _mm_shuffle_epi8(scales,get_scale_shuffle(is + 3));
            is += 4;
            __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh); qh += 32;
            __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1,m4),_mm256_slli_epi16(_mm256_and_si256(q4bitsH,m2),4));
            __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2,m4),_mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH,2),m2),4));
            __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1,4),m4),_mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH,4),m2),4));
            __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2,4),m4),_mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH,6),m2),4));
            __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16_0 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_0,q8_0),_mm256_maddubs_epi16(m32s,q8_0));
            __m256i p16_1 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_1,q8_1),_mm256_maddubs_epi16(m32s,q8_1));
            __m256i p16_2 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_2,q8_2),_mm256_maddubs_epi16(m32s,q8_2));
            __m256i p16_3 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_3,q8_3),_mm256_maddubs_epi16(m32s,q8_3));
            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0),p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1),p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2),p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3),p16_3);
            sumi = _mm256_add_epi32(sumi,_mm256_add_epi32(_mm256_add_epi32(p16_0,p16_1),_mm256_add_epi32(p16_2,p16_3)));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d),_mm256_cvtepi32_ps(sumi),acc);
    }

    return hsum_float_8(acc);
}
