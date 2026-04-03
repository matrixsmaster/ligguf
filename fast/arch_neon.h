/* LiGGUF - a small, dependency-free LLaMA inference engine with direct GGUF support
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025-2026
 * */

#include <arm_neon.h>

#define ARCH_ACCEL

static inline int hsum_i32x4(int32x4_t x)
{
    return vaddvq_s32(x);
}

static inline int8x16x2_t q1_0_bits_to_bytes_32(const uint8_t* qs)
{
    static const int8_t lut[16][4] = {
        { -1,-1,-1,-1 }, {  1,-1,-1,-1 }, { -1, 1,-1,-1 }, {  1, 1,-1,-1 },
        { -1,-1, 1,-1 }, {  1,-1, 1,-1 }, { -1, 1, 1,-1 }, {  1, 1, 1,-1 },
        { -1,-1,-1, 1 }, {  1,-1,-1, 1 }, { -1, 1,-1, 1 }, {  1, 1,-1, 1 },
        { -1,-1, 1, 1 }, {  1,-1, 1, 1 }, { -1, 1, 1, 1 }, {  1, 1, 1, 1 },
    };
    int8_t out[32];
    for (int i = 0; i < 8; i++) memcpy(out + i * 4,lut[(qs[i >> 1] >> ((i & 1) * 4)) & 0xF],4);
    int8x16x2_t r;
    r.val[0] = vld1q_s8(out + 0);
    r.val[1] = vld1q_s8(out + 16);
    return r;
}

float inline dot_f32(const float* a, const float* b, int size)
{
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= size; i += 4) acc = vfmaq_f32(acc,vld1q_f32(a + i),vld1q_f32(b + i));
    float sum = vaddvq_f32(acc);
    for (; i < size; i++) sum += a[i] * b[i];
    return sum;
}

float inline dot_f16_f32(const gguf_half* a, const float* b, int size)
{
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= size; i += 4) {
        float aux[4] = { fp16_to_fp32(a[i]), fp16_to_fp32(a[i+1]), fp16_to_fp32(a[i+2]), fp16_to_fp32(a[i+3]) };
        acc = vfmaq_f32(acc,vld1q_f32(aux),vld1q_f32(b + i));
    }
    float sum = vaddvq_f32(acc);
    for (; i < size; i++) sum += fp16_to_fp32(a[i]) * b[i];
    return sum;
}

float inline dot_q8_0_q8_0(const block_q8_0* x, const block_q8_0* y, int n)
{
    const int nb = n / QK8_0;
    const int32x4_t vzero = vdupq_n_s32(0);
    float acc = 0.0f;

    for (int i = 0; i < nb; i++) {
        int8x16_t qx0 = vld1q_s8(x[i].qs + 0);
        int8x16_t qx1 = vld1q_s8(x[i].qs + 16);
        int8x16_t qy0 = vld1q_s8(y[i].qs + 0);
        int8x16_t qy1 = vld1q_s8(y[i].qs + 16);
        int32x4_t sumi = vdotq_s32(vdotq_s32(vzero,qx0,qy0),qx1,qy1);
        acc += fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d) * hsum_i32x4(sumi);
    }

    return acc;
}

float inline dot_q8_0_q1_0(const block_q8_0* y, const block_q1_0* x, int n)
{
    const int nb = n / QK8_0;
    const int32x4_t vzero = vdupq_n_s32(0);
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        int8x16x2_t qx = q1_0_bits_to_bytes_32(x[i].qs);
        int8x16_t qy0 = vld1q_s8(y[i].qs + 0);
        int8x16_t qy1 = vld1q_s8(y[i].qs + 16);
        int32x4_t sumi = vdotq_s32(vdotq_s32(vzero,qx.val[0],qy0),qx.val[1],qy1);
        sumf += fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d) * hsum_i32x4(sumi);
    }

    return sumf;
}

float inline dot_q8_0_q1_G(const block_q8_0* y, const block_q1_G* x, int n)
{
    const int nb = n / 128;
    const int32x4_t vzero = vdupq_n_s32(0);
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d0 = fp16_to_fp32(x[i].d);
        float sumi = 0.0f;

        for (int k = 0; k < 4; k++) {
            const block_q8_0* yi = y + i * 4 + k;
            int8x16x2_t qx = q1_0_bits_to_bytes_32(x[i].qs + k * 4);
            int8x16_t qy0 = vld1q_s8(yi->qs + 0);
            int8x16_t qy1 = vld1q_s8(yi->qs + 16);
            int32x4_t sumib = vdotq_s32(vdotq_s32(vzero,qx.val[0],qy0),qx.val[1],qy1);
            sumi += fp16_to_fp32(yi->d) * hsum_i32x4(sumib);
        }

        sumf += d0 * sumi;
    }

    return sumf;
}

float inline dot_q8_K_q2_K(const block_q8_K* y, const block_q2_K* x, int n)
{
    const int nb = n / QK_K;
    const uint8x16_t m3 = vdupq_n_u8(0x3);
    const uint8x16_t m4 = vdupq_n_u8(0xF);
    const int32x4_t vzero = vdupq_n_s32(0);
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);
        const uint8_t* q2 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        const uint8x16_t mins_and_scales = vld1q_u8(x[i].scales);
        const uint8x16_t scales = vandq_u8(mins_and_scales,m4);
        const uint8x16_t mins = vshrq_n_u8(mins_and_scales,4);
        uint8_t aux[16];
        vst1q_u8(aux,scales);

        int16x8_t mins_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins)));
        int16x8_t mins_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)));
        int16x8_t q8s_lo = vld1q_s16(y[i].bsums + 0);
        int16x8_t q8s_hi = vld1q_s16(y[i].bsums + 8);
        int32x4_t s0 = vaddq_s32(vmull_s16(vget_low_s16(mins_lo),vget_low_s16(q8s_lo)),vmull_s16(vget_high_s16(mins_lo),vget_high_s16(q8s_lo)));
        int32x4_t s1 = vaddq_s32(vmull_s16(vget_low_s16(mins_hi),vget_low_s16(q8s_hi)),vmull_s16(vget_high_s16(mins_hi),vget_high_s16(q8s_hi)));
        sum += dmin * hsum_i32x4(vaddq_s32(s0,s1));

        int isum = 0;
        int is = 0;
        for (int j = 0; j < QK_K/128; j++) {
            uint8x16x2_t q2bits = vld1q_u8_x2(q2); q2 += 32;
            int8x16x2_t q8bytes = vld1q_s8_x2(q8); q8 += 32;
            int8x16x2_t q2bytes;

            q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[0],m3));
            q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[1],m3));
            isum += hsum_i32x4(vdotq_s32(vzero,q2bytes.val[0],q8bytes.val[0])) * aux[is + 0];
            isum += hsum_i32x4(vdotq_s32(vzero,q2bytes.val[1],q8bytes.val[1])) * aux[is + 1];

            q8bytes = vld1q_s8_x2(q8); q8 += 32;
            q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[0],2),m3));
            q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[1],2),m3));
            isum += hsum_i32x4(vdotq_s32(vzero,q2bytes.val[0],q8bytes.val[0])) * aux[is + 2];
            isum += hsum_i32x4(vdotq_s32(vzero,q2bytes.val[1],q8bytes.val[1])) * aux[is + 3];

            q8bytes = vld1q_s8_x2(q8); q8 += 32;
            q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[0],4),m3));
            q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[1],4),m3));
            isum += hsum_i32x4(vdotq_s32(vzero,q2bytes.val[0],q8bytes.val[0])) * aux[is + 4];
            isum += hsum_i32x4(vdotq_s32(vzero,q2bytes.val[1],q8bytes.val[1])) * aux[is + 5];

            q8bytes = vld1q_s8_x2(q8); q8 += 32;
            q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[0],6),m3));
            q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[1],6),m3));
            isum += hsum_i32x4(vdotq_s32(vzero,q2bytes.val[0],q8bytes.val[0])) * aux[is + 6];
            isum += hsum_i32x4(vdotq_s32(vzero,q2bytes.val[1],q8bytes.val[1])) * aux[is + 7];
            is += 8;
        }

        sum += d * isum;
    }

    return sum;
}

float inline dot_q8_K_q3_K(const block_q8_K* y, const block_q3_K* x, int n)
{
    const int nb = n / QK_K;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint8x16_t m3b = vdupq_n_u8(3);
    const int32x4_t vzero = vdupq_n_s32(0);
    const uint8x16_t m0 = vdupq_n_u8(1);
    const uint8x16_t m1 = vshlq_n_u8(m0,1);
    const uint8x16_t m2 = vshlq_n_u8(m0,2);
    const uint8x16_t m3 = vshlq_n_u8(m0,3);
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        uint32_t aux[3];
        uint32_t utmp[4];
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const uint8_t* q3 = x[i].qs;
        const uint8_t* qh = x[i].hmask;
        const int8_t* q8 = y[i].qs;
        uint8x16x2_t qhbits = vld1q_u8_x2(qh);

        memcpy(aux,x[i].scales,12);
        utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
        utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
        utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
        utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);
        int8_t* scale = (int8_t*)utmp;
        for (int j = 0; j < 16; j++) scale[j] -= 32;

        int isum = 0;
        for (int j = 0; j < QK_K/128; j++) {
            uint8x16x2_t q3bits = vld1q_u8_x2(q3); q3 += 32;
            int8x16x4_t q8bytes1 = vld1q_s8_x4(q8); q8 += 64;
            int8x16x4_t q8bytes2 = vld1q_s8_x4(q8); q8 += 64;
            uint8x16x4_t q3h;
            int8x16x4_t q3bytes;

            q3h.val[0] = vshlq_n_u8(vbicq_u8(m0,qhbits.val[0]),2);
            q3h.val[1] = vshlq_n_u8(vbicq_u8(m0,qhbits.val[1]),2);
            q3h.val[2] = vshlq_n_u8(vbicq_u8(m1,qhbits.val[0]),1);
            q3h.val[3] = vshlq_n_u8(vbicq_u8(m1,qhbits.val[1]),1);
            q3bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[0],m3b)),vreinterpretq_s8_u8(q3h.val[0]));
            q3bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[1],m3b)),vreinterpretq_s8_u8(q3h.val[1]));
            q3bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0],2),m3b)),vreinterpretq_s8_u8(q3h.val[2]));
            q3bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1],2),m3b)),vreinterpretq_s8_u8(q3h.val[3]));
            isum += hsum_i32x4(vdotq_s32(vzero,q3bytes.val[0],q8bytes1.val[0])) * scale[0];
            isum += hsum_i32x4(vdotq_s32(vzero,q3bytes.val[1],q8bytes1.val[1])) * scale[1];
            isum += hsum_i32x4(vdotq_s32(vzero,q3bytes.val[2],q8bytes1.val[2])) * scale[2];
            isum += hsum_i32x4(vdotq_s32(vzero,q3bytes.val[3],q8bytes1.val[3])) * scale[3];
            scale += 4;

            q3h.val[0] = vbicq_u8(m2,qhbits.val[0]);
            q3h.val[1] = vbicq_u8(m2,qhbits.val[1]);
            q3h.val[2] = vshrq_n_u8(vbicq_u8(m3,qhbits.val[0]),1);
            q3h.val[3] = vshrq_n_u8(vbicq_u8(m3,qhbits.val[1]),1);
            q3bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0],4),m3b)),vreinterpretq_s8_u8(q3h.val[0]));
            q3bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1],4),m3b)),vreinterpretq_s8_u8(q3h.val[1]));
            q3bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0],6),m3b)),vreinterpretq_s8_u8(q3h.val[2]));
            q3bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1],6),m3b)),vreinterpretq_s8_u8(q3h.val[3]));
            isum += hsum_i32x4(vdotq_s32(vzero,q3bytes.val[0],q8bytes2.val[0])) * scale[0];
            isum += hsum_i32x4(vdotq_s32(vzero,q3bytes.val[1],q8bytes2.val[1])) * scale[1];
            isum += hsum_i32x4(vdotq_s32(vzero,q3bytes.val[2],q8bytes2.val[2])) * scale[2];
            isum += hsum_i32x4(vdotq_s32(vzero,q3bytes.val[3],q8bytes2.val[3])) * scale[3];
            scale += 4;

            if (j == 0) {
                qhbits.val[0] = vshrq_n_u8(qhbits.val[0],4);
                qhbits.val[1] = vshrq_n_u8(qhbits.val[1],4);
            }
        }

        sum += d * isum;
    }

    return sum;
}

float inline dot_q8_K_q4_K(const block_q8_K* y, const block_q4_K* x, int n)
{
    const int nb = n / QK_K;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    const uint8x16_t m4b = vdupq_n_u8(0xF);
    const int32x4_t mzero = vdupq_n_s32(0);
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = y[i].d * fp16_to_fp32(x[i].dmin);

        memcpy(utmp,x[i].scales,12);
        uint32x2_t mins8 = { 0 };
        mins8 = vset_lane_u32(utmp[1] & kmask1,mins8,0);
        mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4),mins8,1);
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums),vld1q_s16(y[i].bsums + 8));
        int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16(q8sums),vget_low_s16(mins)),vmull_s16(vget_high_s16(q8sums),vget_high_s16(mins)));
        sumf -= dmin * hsum_i32x4(prod);

        const uint8_t* scales = (const uint8_t*)utmp;
        const uint8_t* q4 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        int sumi1 = 0;
        int sumi2 = 0;

        for (int j = 0; j < QK_K/64; j++) {
            uint8x16x2_t q4bits = vld1q_u8_x2(q4); q4 += 32;
            int8x16x2_t q8a = vld1q_s8_x2(q8); q8 += 32;
            int8x16_t q40 = vreinterpretq_s8_u8(vandq_u8(q4bits.val[0],m4b));
            int8x16_t q41 = vreinterpretq_s8_u8(vandq_u8(q4bits.val[1],m4b));
            sumi1 += hsum_i32x4(vdotq_s32(vdotq_s32(mzero,q40,q8a.val[0]),q41,q8a.val[1])) * scales[2*j + 0];

            int8x16x2_t q8b = vld1q_s8_x2(q8); q8 += 32;
            q40 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0],4));
            q41 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1],4));
            sumi2 += hsum_i32x4(vdotq_s32(vdotq_s32(mzero,q40,q8b.val[0]),q41,q8b.val[1])) * scales[2*j + 1];
        }

        sumf += d * (sumi1 + sumi2);
    }

    return sumf;
}

float inline dot_q8_K_q5_K(const block_q8_K* y, const block_q5_K* x, int n)
{
    const int nb = n / QK_K;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    const uint8x16_t m4b = vdupq_n_u8(0xF);
    const uint8x16_t mone = vdupq_n_u8(1);
    const uint8x16_t mtwo = vdupq_n_u8(2);
    const int32x4_t mzero = vdupq_n_s32(0);
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = y[i].d * fp16_to_fp32(x[i].dmin);

        memcpy(utmp,x[i].scales,12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        uint8x8_t mins8 = vld1_u8((const uint8_t*)utmp + 8);
        int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(mins8));
        int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums),vld1q_s16(y[i].bsums + 8));
        int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16(q8sums),vget_low_s16(mins)),vmull_s16(vget_high_s16(q8sums),vget_high_s16(mins)));
        int sumi_mins = hsum_i32x4(prod);

        const uint8_t* scales = (const uint8_t*)utmp;
        const uint8_t* q5 = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const int8_t* q8 = y[i].qs;
        uint8x16x2_t qhbits = vld1q_u8_x2(qh);
        int sumi = 0;

        for (int j = 0; j < QK_K/64; j++) {
            uint8x16x2_t q5bits = vld1q_u8_x2(q5); q5 += 32;
            int8x16x4_t q8bytes = vld1q_s8_x4(q8); q8 += 64;

            uint8x16_t q5h0 = vshlq_n_u8(vandq_u8(mone,qhbits.val[0]),4);
            uint8x16_t q5h1 = vshlq_n_u8(vandq_u8(mone,qhbits.val[1]),4);
            uint8x16_t q5h2 = vshlq_n_u8(vandq_u8(mtwo,qhbits.val[0]),3);
            uint8x16_t q5h3 = vshlq_n_u8(vandq_u8(mtwo,qhbits.val[1]),3);
            qhbits.val[0] = vshrq_n_u8(qhbits.val[0],2);
            qhbits.val[1] = vshrq_n_u8(qhbits.val[1],2);

            int8x16_t q50 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.val[0],m4b),q5h0));
            int8x16_t q51 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.val[1],m4b),q5h1));
            int8x16_t q52 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.val[0],4),q5h2));
            int8x16_t q53 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.val[1],4),q5h3));
            sumi += hsum_i32x4(vdotq_s32(vdotq_s32(mzero,q50,q8bytes.val[0]),q51,q8bytes.val[1])) * *scales++;
            sumi += hsum_i32x4(vdotq_s32(vdotq_s32(mzero,q52,q8bytes.val[2]),q53,q8bytes.val[3])) * *scales++;
        }

        sumf += d * sumi - dmin * sumi_mins;
    }

    return sumf;
}

float inline dot_q8_K_q6_K(const block_q8_K* y, const block_q6_K* x, int n)
{
    float sum = 0.0f;
    const int nb = n / QK_K;
    const uint8x16_t m4b = vdupq_n_u8(0xF);
    const int32x4_t vzero = vdupq_n_s32(0);
    const uint8x16_t mone = vdupq_n_u8(3);

    for (int i = 0; i < nb; i++) {
        const float d_all = fp16_to_fp32(x[i].d);
        const uint8_t* q6 = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* q8 = y[i].qs;
        const int8_t* scale = x[i].scales;
        int16x8x2_t q8sums = vld1q_s16_x2(y[i].bsums);
        int8x16_t scales = vld1q_s8(scale);
        int16x8x2_t q6scales = {{ vmovl_s8(vget_low_s8(scales)), vmovl_s8(vget_high_s8(scales)) }};
        int32x4_t prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(q8sums.val[0]),vget_low_s16(q6scales.val[0])),vmull_s16(vget_high_s16(q8sums.val[0]),vget_high_s16(q6scales.val[0]))),vaddq_s32(vmull_s16(vget_low_s16(q8sums.val[1]),vget_low_s16(q6scales.val[1])),vmull_s16(vget_high_s16(q8sums.val[1]),vget_high_s16(q6scales.val[1]))));
        int isum_mins = hsum_i32x4(prod);
        int isum = 0;

        for (int j = 0; j < QK_K/128; j++) {
            uint8x16x2_t qhbits = vld1q_u8_x2(qh); qh += 32;
            uint8x16x4_t q6bits = vld1q_u8_x4(q6); q6 += 64;
            int8x16x4_t q8bytes = vld1q_s8_x4(q8); q8 += 64;
            uint8x16x4_t q6h;
            int8x16x4_t q6bytes;

            q6h.val[0] = vshlq_n_u8(vandq_u8(mone,qhbits.val[0]),4);
            q6h.val[1] = vshlq_n_u8(vandq_u8(mone,qhbits.val[1]),4);
            uint8x16_t shifted = vshrq_n_u8(qhbits.val[0],2);
            q6h.val[2] = vshlq_n_u8(vandq_u8(mone,shifted),4);
            shifted = vshrq_n_u8(qhbits.val[1],2);
            q6h.val[3] = vshlq_n_u8(vandq_u8(mone,shifted),4);
            q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0],m4b),q6h.val[0]));
            q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1],m4b),q6h.val[1]));
            q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2],m4b),q6h.val[2]));
            q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3],m4b),q6h.val[3]));
            isum += hsum_i32x4(vdotq_s32(vzero,q6bytes.val[0],q8bytes.val[0])) * scale[0] + hsum_i32x4(vdotq_s32(vzero,q6bytes.val[1],q8bytes.val[1])) * scale[1] + hsum_i32x4(vdotq_s32(vzero,q6bytes.val[2],q8bytes.val[2])) * scale[2] + hsum_i32x4(vdotq_s32(vzero,q6bytes.val[3],q8bytes.val[3])) * scale[3];
            scale += 4;

            q8bytes = vld1q_s8_x4(q8); q8 += 64;
            shifted = vshrq_n_u8(qhbits.val[0],4);
            q6h.val[0] = vshlq_n_u8(vandq_u8(mone,shifted),4);
            shifted = vshrq_n_u8(qhbits.val[1],4);
            q6h.val[1] = vshlq_n_u8(vandq_u8(mone,shifted),4);
            shifted = vshrq_n_u8(qhbits.val[0],6);
            q6h.val[2] = vshlq_n_u8(vandq_u8(mone,shifted),4);
            shifted = vshrq_n_u8(qhbits.val[1],6);
            q6h.val[3] = vshlq_n_u8(vandq_u8(mone,shifted),4);
            q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0],4),q6h.val[0]));
            q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1],4),q6h.val[1]));
            q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2],4),q6h.val[2]));
            q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3],4),q6h.val[3]));
            isum += hsum_i32x4(vdotq_s32(vzero,q6bytes.val[0],q8bytes.val[0])) * scale[0] + hsum_i32x4(vdotq_s32(vzero,q6bytes.val[1],q8bytes.val[1])) * scale[1] + hsum_i32x4(vdotq_s32(vzero,q6bytes.val[2],q8bytes.val[2])) * scale[2] + hsum_i32x4(vdotq_s32(vzero,q6bytes.val[3],q8bytes.val[3])) * scale[3];
            scale += 4;
        }

        sum += d_all * y[i].d * (isum - 32 * isum_mins);
    }

    return sum;
}
