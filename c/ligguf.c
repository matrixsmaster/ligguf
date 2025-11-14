/* LiGGUF - a tiny, dependency-free LLaMA inference engine with direct GGUF support
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025
 *
 * Pure C version
 * */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

#define ALIGNMENT 32
#define MAXNAMELEN 1024
#define QK8_0 32

#define TOKENS_KEY "tokenizer.ggml.tokens"
#define TOKENS_SCORE_KEY "tokenizer.ggml.scores"
#define TOKENS_EMBED_KEY "token_embd.weight"
#define TOKENS_BOS_KEY "tokenizer.ggml.bos_token_id"
#define TOKENS_EOS_KEY "tokenizer.ggml.eos_token_id"
#define VOCAB_SIZE_KEY "llama.vocab_size"
#define CONTEXT_LEN_KEY "llama.context_length"
#define EMBED_LEN_KEY "llama.embedding_length"
#define HEAD_COUNT_KEY "llama.attention.head_count"
#define HEAD_KV_COUNT_KEY "llama.attention.head_count_kv"
#define BLOCK_COUNT_KEY "llama.block_count"
#define RMS_EPSILON_KEY "llama.attention.layer_norm_rms_epsilon"
#define ROPE_BASE_KEY "llama.rope.freq_base"
#define ROPE_DIMS_KEY "llama.rope.dimension_count"
#define FF_LEN_KEY "llama.feed_forward_length"
#define OUTPUT_KEY "output.weight"
#define OUTPUT_NORM_KEY "output_norm.weight"

#define MIN(A,B) (((A) < (B))? (A) : (B))
#define MAX(A,B) (((A) > (B))? (A) : (B))

#define MULTITHREAD _Pragma("omp parallel for")

#define ERR(S,...) fprintf(stderr,S "\n", __VA_ARGS__)

#ifdef DEBUG
    #define DBG(S,...) printf(S "\n",__VA_ARGS__)
    #define TIMING_START struct timespec _ts,_te; clock_gettime(CLOCK_MONOTONIC,&_ts)
    #define TIMING_STOP(X) clock_gettime(CLOCK_MONOTONIC,&_te); DBG("Time passed for " X ": %lu ms",(_te.tv_sec-_ts.tv_sec)*1000+((_te.tv_nsec-_ts.tv_nsec)/1000000))
#else
    #define DBG(...)
    #define TIMING_START
    #define TIMING_STOP(...)
#endif

typedef enum { F32, F16, Q8_0 = 8 } gguf_type;

typedef enum {
    GUINT8, GINT8, GUINT16, GINT16, GUINT32, GINT32,
    GFLOAT32, GBOOL, GSTRING, GARRAY, GUINT64, GINT64, GFLOAT64,
    GGUF_VAL_TYPE_COUNT
} gguf_val_type;

typedef struct {
    uint16_t d;
    int8_t qs[QK8_0];
} block_q8_0;

typedef block_q8_0* qtensor;
typedef float* ftensor;

typedef struct {
    ftensor att_norm, ffn_norm; // RMS norm weights for attention and FFN
    qtensor att_q, att_k, att_v, att_out; // attention QKV + output weights
    qtensor ffn_up, ffn_down, ffn_gate; // FFN weights
} trans_block;

typedef struct {
    int file; // model file handle
    uint8_t* base; // base mmap address
    uint64_t fsize; // model file size
    uint8_t* tensors_off; // aligned offset of the start of tensors block

    int vocab_size; // size of the vocabulary
    int n_layers; // number of layers
    int n_heads; // number of Query heads
    int n_kv_heads; // number of Key/Value heads
    int n_embed; // input embedding size
    int n_context; // size of the context window
    int tok_bos, tok_eos; // BOS/EOS token IDs
    int head_dim; // head size (computed)
    float rms_epsilon; // epsilon for RMS norm
    float rope_base; // RoPE base freq
    int rope_dim; // RoPE dimension (normally should be equal to head_dim, but could be smaller)
    int n_ff; // Feed-Forward length
    int kv_dim; // Key-Value dimension (per head)
    uint64_t tok_key_off, tok_score_off; // offsets for tokenizer (dict and scores)

    char** tokens; // vector of known tokens (pos == index)
    ftensor tokscores; // token scores (const)
    qtensor t_embed; // embedding tensor (const)
    qtensor t_out; // output classifier weights (const)
    ftensor t_outnorm; // output classifier RMS norm weights (const)
    trans_block* tr; // transformer blocks/layers (const)

    ftensor x; // activation at current time stamp
    qtensor xq; // quantized x
    ftensor xb; // activation inside a residual branch
    ftensor xb2; // raw attention result
    ftensor hb; // ffn up result
    ftensor hb2; // ffn gate result
    ftensor q,k,v; // current QKV
    ftensor kc,vc; // KV cache
    ftensor att; // attention scores
} model_state;

model_state g_m;
float fp1632_lut[65536];

// unified reading from memory
#define RDMEM(T,N) static inline T N (uint8_t** pos)\
{\
    T r = 0;\
    memcpy(&r,*pos,sizeof(T));\
    *pos += sizeof(T);\
    return r;\
}

//RDMEM(uint8_t,rd8)
//RDMEM(uint16_t,rd16)
RDMEM(uint32_t,rd32)
RDMEM(uint64_t,rd64)
RDMEM(float,rdf32)

// unified type conversion
#define TYPECONV(F,T,N) static inline T N (F x)\
{\
    T r;\
    memcpy(&r,&x,sizeof(F));\
    return r;\
}

TYPECONV(float,uint32_t,fp32_to_bits)
TYPECONV(uint32_t,float,fp32_from_bits)

// string reader with automatic allocation
char* rdstra(uint8_t** pos)
{
    uint64_t l = rd64(pos);
    char* res = (char*)malloc(l+1);
    memcpy(res,*pos,l);
    res[l] = 0;
    *pos += l;
    return res;
}

uint64_t skipper(gguf_val_type t, uint8_t* pos)
{
    switch (t) {
        case GUINT8:
        case GINT8:
        case GBOOL:
            return 1;
        case GUINT16:
        case GINT16:
            return 2;
        case GUINT32:
        case GINT32:
        case GFLOAT32:
            return 4;
        case GUINT64:
        case GINT64:
        case GFLOAT64:
            return 8;
        case GSTRING: {
            uint64_t l = rd64(&pos);
            return l + 8;
        }
        case GARRAY: {
            uint8_t* org = pos;
            gguf_val_type nt = (gguf_val_type)rd32(&pos);
            uint64_t ne = rd64(&pos);
            for (uint64_t i = 0; i < ne; i++) pos += skipper(nt,pos);
            return pos - org;
        }
        default:
            assert(0);
    }
}

void open_mmap(const char* fn)
{
    g_m.fsize = 0;
    g_m.base = NULL;
    g_m.file = open(fn,O_RDONLY);
    assert(g_m.file != -1);

    struct stat st;
    assert(!fstat(g_m.file,&st));

    g_m.fsize = st.st_size;
    g_m.base = (uint8_t*)mmap(NULL,g_m.fsize,PROT_READ,MAP_SHARED,g_m.file,0);
    assert(g_m.base != MAP_FAILED);
}

void read_gguf()
{
    uint8_t* p = g_m.base;
    uint32_t magic = rd32(&p);
    assert(!memcmp(&magic,"GGUF",4));
    uint32_t ver = rd32(&p);
    assert(ver == 3);
    uint64_t nten = rd64(&p);
    uint64_t meta = rd64(&p);

    char* key = NULL;
    for (uint64_t i = 0; i < meta; i++) {
        key = rdstra(&p);
        gguf_val_type tag = (gguf_val_type)rd32(&p);
        uint64_t off = p - g_m.base;

        p += skipper(tag,p);
        uint8_t* dp = g_m.base + off;
#ifdef DEBUG
        uint8_t* ddp = dp;
#endif

        if (!strcmp(key,VOCAB_SIZE_KEY)) g_m.vocab_size = rd32(&dp);
        else if (!strcmp(key,BLOCK_COUNT_KEY)) g_m.n_layers = rd32(&dp);
        else if (!strcmp(key,EMBED_LEN_KEY)) g_m.n_embed = rd32(&dp);
        else if (!strcmp(key,HEAD_COUNT_KEY)) g_m.n_heads = rd32(&dp);
        else if (!strcmp(key,HEAD_KV_COUNT_KEY)) g_m.n_kv_heads = rd32(&dp);
        else if (!strcmp(key,CONTEXT_LEN_KEY)) g_m.n_context = rd32(&dp);
        else if (!strcmp(key,TOKENS_BOS_KEY)) g_m.tok_bos = rd32(&dp);
        else if (!strcmp(key,TOKENS_EOS_KEY)) g_m.tok_eos = rd32(&dp);
        else if (!strcmp(key,RMS_EPSILON_KEY)) g_m.rms_epsilon = rdf32(&dp);
        else if (!strcmp(key,ROPE_BASE_KEY)) g_m.rope_base = rdf32(&dp);
        else if (!strcmp(key,ROPE_DIMS_KEY)) g_m.rope_dim = rd32(&dp);
        else if (!strcmp(key,FF_LEN_KEY)) g_m.n_ff = rd32(&dp);
        else if (!strcmp(key,TOKENS_KEY)) g_m.tok_key_off = off;
        else if (!strcmp(key,TOKENS_SCORE_KEY)) g_m.tok_score_off = off;

        DBG("Key '%s' read, %s",key,((ddp == dp)? "ignored":"processed"));
        free(key);
    }

    // skip tensor table to get to the beginning of the tensor data frame
    uint8_t* prevpos = p;
    for (uint64_t i = 0; i < nten; i++) {
        key = rdstra(&p);
        free(key);
        uint32_t ndim = rd32(&p);
        p += 8 * ndim + 4 + 8;
    }

    uint64_t off = p - g_m.base;
    g_m.tensors_off = g_m.base + (off + (ALIGNMENT - (off % ALIGNMENT)) % ALIGNMENT);

    g_m.head_dim = g_m.n_embed / g_m.n_heads;
    assert(g_m.head_dim * g_m.n_heads == g_m.n_embed);
    g_m.kv_dim = (g_m.n_embed * g_m.n_kv_heads) / g_m.n_heads;

    // init internal tensor memory
    g_m.x = (ftensor)malloc(g_m.n_embed * sizeof(float));
    g_m.xb = (ftensor)malloc(g_m.n_embed * sizeof(float));
    g_m.xb2 = (ftensor)malloc(g_m.n_embed * sizeof(float));
    g_m.xq = (qtensor)malloc(g_m.n_ff * sizeof(block_q8_0)); // provide it with the biggest possible size for quantized hb
    g_m.hb = (ftensor)malloc(g_m.n_ff * sizeof(float));
    g_m.hb2 = (ftensor)malloc(g_m.n_ff * sizeof(float));
    g_m.q = (ftensor)malloc(g_m.n_embed * sizeof(float));
    g_m.k = (ftensor)malloc(g_m.kv_dim * sizeof(float));
    g_m.v = (ftensor)malloc(g_m.kv_dim * sizeof(float));
    g_m.kc = (ftensor)malloc(g_m.n_layers * g_m.n_context * g_m.kv_dim * sizeof(float));
    g_m.vc = (ftensor)malloc(g_m.n_layers * g_m.n_context * g_m.kv_dim * sizeof(float));
    g_m.att = (ftensor)malloc(g_m.n_heads * g_m.n_context * sizeof(float));

    // init storage for external tensor pointers
    g_m.tr = (trans_block*)malloc(g_m.n_layers * sizeof(trans_block));
    memset(g_m.tr,0,g_m.n_layers * sizeof(trans_block));

    // read and initialize tensor pointers
    p = prevpos;
    for (uint64_t i = 0; i < nten; i++) {
        key = rdstra(&p);
        uint32_t ndim = rd32(&p);
#ifdef DEBUG
        p += 8 * ndim;
        uint32_t type = rd32(&p);
#else
        p += 8 * ndim + 4;
#endif
        uint64_t off = rd64(&p);
        uint8_t* ptr = g_m.tensors_off + off;
        DBG("Tensor '%s', type %u, offset %lu",key,type,off);

        if (!strncmp(key,"blk.",4) && strstr(key,".weight")) {
            int id = atoi(key+4);
            assert(id >= 0 && id < g_m.n_layers);

            if (strstr(key,"attn_norm")) g_m.tr[id].att_norm = (ftensor)ptr;
            else if (strstr(key,"attn_q")) g_m.tr[id].att_q = (qtensor)ptr;
            else if (strstr(key,"attn_k")) g_m.tr[id].att_k = (qtensor)ptr;
            else if (strstr(key,"attn_v")) g_m.tr[id].att_v = (qtensor)ptr;
            else if (strstr(key,"attn_out")) g_m.tr[id].att_out = (qtensor)ptr;
            else if (strstr(key,"ffn_norm")) g_m.tr[id].ffn_norm = (ftensor)ptr;
            else if (strstr(key,"ffn_up")) g_m.tr[id].ffn_up = (qtensor)ptr;
            else if (strstr(key,"ffn_down")) g_m.tr[id].ffn_down = (qtensor)ptr;
            else if (strstr(key,"ffn_gate")) g_m.tr[id].ffn_gate = (qtensor)ptr;
        }
        else if (!strcmp(key,TOKENS_EMBED_KEY)) g_m.t_embed = (qtensor)ptr;
        else if (!strcmp(key,OUTPUT_KEY)) g_m.t_out = (qtensor)ptr;
        else if (!strcmp(key,OUTPUT_NORM_KEY)) g_m.t_outnorm = (ftensor)ptr;
        free(key);
    }
}

void read_tokenizer()
{
    uint8_t* pos = g_m.base + g_m.tok_key_off;

    gguf_val_type nt = (gguf_val_type)rd32(&pos);
    assert(nt == GSTRING);

    uint64_t ne = rd64(&pos);
    assert((int)ne == g_m.vocab_size);
    g_m.tokens = (char**)malloc(ne * sizeof(char*));
    memset(g_m.tokens,0,ne * sizeof(char*));
    for (uint64_t i = 0; i < ne; i++) g_m.tokens[i] = rdstra(&pos);

    pos = g_m.base + g_m.tok_score_off;
    nt = (gguf_val_type)rd32(&pos);
    assert(nt == GFLOAT32);

    ne = rd64(&pos);
    g_m.tokscores = (ftensor)malloc(ne * sizeof(float));
    memcpy(g_m.tokscores,pos,ne * sizeof(float));
}

int* tokenize(const char* str, int bos, int eos)
{
    int olen = (strlen(str) + 2) * sizeof(int);
    int* out = (int*)malloc(olen);
    memset(out,0,olen);
    int* pout = out;

    if (bos) *pout++ = g_m.tok_bos;

    char s[8] = {0};
    while (*str || *s) {
        if (!*s) *s = *str++;

        for (int i = 0; i < g_m.vocab_size; i++) {
            if (!strcmp(g_m.tokens[i],s)) {
                *pout++ = i;
                DBG("Dumb token added: '%s'",s);
                memset(s,0,sizeof(s));
                break;
            }
        }

        if (*s) {
            assert(strlen(s) == 1);
            snprintf(s,sizeof(s),"<0x%02X>",*s);
        }
    }
    if (pout > out) pout--;

    while (1) {
        unsigned plen = pout - out + 1;
        float best_score = -1e10;
        unsigned best_id = -1;
        unsigned best_idx = plen + 1;

        char acc[MAXNAMELEN] = {0};
        for (unsigned i = 0; i < plen-1; i++) {
            sprintf(acc,"%s%s",g_m.tokens[out[i]],g_m.tokens[out[i+1]]);
            int fnd = -1;
            for (int j = 0; j < g_m.vocab_size; j++) {
                if (!strcmp(g_m.tokens[j],acc)) {
                    fnd = j;
                    break;
                }
            }
            if (fnd < 0) continue;

            float sc = g_m.tokscores[fnd];
            if (sc > best_score) {
                best_score = sc;
                best_id = fnd;
                best_idx = i;
                DBG("Tokens merged: '%s'",acc);
            }
        }

        if (best_idx > plen) break;

        out[best_idx] = best_id;
        for (unsigned j = best_idx+1; j < plen-1; j++) out[j] = out[j+1];
        *pout-- = 0;
        for (int* pi = out; *pi; pi++) DBG("$$: %d",*pi);
    }

    if (eos) *pout++ = g_m.tok_eos;

    for (int* pi = out; *pi; pi++) DBG("Final token: %d",*pi);

    return out;
}

static inline float fp16_to_fp32(uint16_t h)
{
    uint32_t w = ((uint32_t)h) << 16;
    uint32_t d = w << 1;
    uint32_t result = (w & (1U << 30)) | (d < (1U << 27) ? fp32_to_bits(fp32_from_bits((d >> 17) | (126U << 23)) - .5f) : fp32_to_bits(fp32_from_bits((d >> 4) + (224U << 23)) * fp32_from_bits(120U << 20)));
    return fp32_from_bits(result);
}

static inline uint16_t fp32_to_fp16(float f)
{
    float b = (fabs(f) * fp32_from_bits(0x778U << 20)) * fp32_from_bits(136U << 20);
    uint32_t w = fp32_to_bits(f);
    uint32_t d = w << 1;
    uint32_t s = d & (255U << 24);
    if (s < (113U << 24)) s = (113U << 24);
    b += fp32_from_bits((s >> 1) + (120U << 20));
    uint32_t bits = fp32_to_bits(b);
    return ((w & (1U << 31)) >> 16) | ((d > (255U << 24)) ? (uint16_t)(126U << 8) : ((bits >> 13) & (124U << 8)) + (bits & 4095U));
}

void dequant_q80(ftensor y, block_q8_0* ptr, uint64_t nrow, int len)
{
    block_q8_0* x = ptr + nrow * (len / QK8_0);
    const int nb = len / QK8_0;

    MULTITHREAD
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);

        for (int j = 0; j < QK8_0; j++)
            y[i*QK8_0 + j] = x[i].qs[j]*d;
    }
}

void quantize_q80(qtensor y, ftensor x, int xsize)
{
    const int nb = xsize / QK8_0;

    MULTITHREAD
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax,fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = fp32_to_fp16(d);

        for (int j = 0; j < QK8_0; j++)
            y[i].qs[j] = roundf(x[i * QK8_0 + j] * id);
    }
}

void rmsnorm(ftensor out, ftensor x, ftensor w, int size)
{
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];

    ss = ss / (float)size + g_m.rms_epsilon;
    ss = 1.0f / sqrtf(ss);

    MULTITHREAD
    for (int i = 0; i < size; i++)
        out[i] = x[i] * ss * w[i];
}

void matmul(ftensor out, qtensor qx, qtensor qw, int n, int d)
{
    const int nb = n / QK8_0;

    MULTITHREAD
    for (int r = 0; r < d; r++) { // each row
        float acc = 0.0;

        for (int b = 0; b < nb; b++) { // each block
            int iw = r * nb + b;

            // integer dot
            int32_t s = 0;
            for (int i = 0; i < QK8_0; i++)
                s += (int32_t)(qx[b].qs[i]) * (int32_t)(qw[iw].qs[i]);

            // scale and accumulate result as float
            acc += fp1632_lut[qx[b].d] * fp1632_lut[qw[iw].d] * (float)s;
        }

        out[r] = acc;
    }
}

void rope(ftensor x, int n_heads, int pos)
{
    int rd = g_m.rope_dim;
    if (rd > g_m.head_dim) rd = g_m.head_dim;
    if (rd & 1) rd--; // ensure even

    MULTITHREAD
    for (int h = 0; h < n_heads; h++) { // for each head
        float* v = x + h * g_m.head_dim;

        for (int i = 0; i < rd; i += 2) {
            // pair index m in [0 .. rd/2)
            const int m = i >> 1;

            // standard RoPE frequency schedule:
            // inv_freq[m] = base^(-2m/rd), angle = pos * inv_freq[m]
            const float ang = pos * powf(g_m.rope_base, -2.0f * (float)m / (float)rd);
            const float c = cosf(ang), s = sinf(ang);

            const float x0 = v[i + 0];
            const float x1 = v[i + 1];

            v[i + 0] = x0 * c - x1 * s;
            v[i + 1] = x0 * s + x1 * c;
        }
    }
}

void softmax(float* x, int size)
{
    // find max value
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // normalize
    MULTITHREAD
    for (int i = 0; i < size; i++) x[i] /= sum;
}

ftensor inference(int tok, int pos)
{
    // input is the embedding vector for the current token
    dequant_q80(g_m.x,g_m.t_embed,tok,g_m.n_embed);

    // for all layers (blocks) of the model
    for (int l = 0; l < g_m.n_layers; l++) {
        // 1. Attention RMS norm
        rmsnorm(g_m.xb,g_m.x,g_m.tr[l].att_norm,g_m.n_embed);

        // 2. QKV over quantized x
        quantize_q80(g_m.xq,g_m.xb,g_m.n_embed);
        matmul(g_m.q,g_m.xq,g_m.tr[l].att_q,g_m.n_embed,g_m.n_embed);
        matmul(g_m.k,g_m.xq,g_m.tr[l].att_k,g_m.n_embed,g_m.kv_dim);
        matmul(g_m.v,g_m.xq,g_m.tr[l].att_v,g_m.n_embed,g_m.kv_dim);

        // 3. RoPE Q & K (float vectors)
        rope(g_m.q,g_m.n_heads,pos);
        rope(g_m.k,g_m.n_kv_heads,pos);

        // 4. Simple KV cache
        uint64_t loff = l * g_m.n_context * g_m.kv_dim; // kv cache layer offset
        float* kc_row = g_m.kc + loff + pos * g_m.kv_dim;
        float* vc_row = g_m.vc + loff + pos * g_m.kv_dim;
        memcpy(kc_row,g_m.k,g_m.kv_dim*sizeof(float));
        memcpy(vc_row,g_m.v,g_m.kv_dim*sizeof(float));

        // 5. Multi-Head Attention
        const int nrep = g_m.n_heads / g_m.n_kv_heads; // GQA, heads per KV head
        for (int h = 0; h < g_m.n_heads; h++) {
            float* att = g_m.att + (h * g_m.n_context); // attention scores for this head
            float* q = g_m.q + (h * g_m.head_dim); // start of the Q vector

            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                kc_row = g_m.kc + loff + t * g_m.kv_dim + (h / nrep) * g_m.head_dim;

                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < g_m.head_dim; i++)
                    score += q[i] * kc_row[i];
                score /= sqrtf(g_m.head_dim);

                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att,pos+1);

            // weighted sum of the values
            float* xb = g_m.xb + (h * g_m.head_dim);
            memset(xb,0,g_m.head_dim * sizeof(float));

            for (int t = 0; t <= pos; t++) {
                vc_row = g_m.vc + loff + t * g_m.kv_dim + (h / nrep) * g_m.head_dim; // value vector for this head and at this timestep
                float a = att[t]; // attention weight for this timestep
                // accumulate the weighted value
                for (int i = 0; i < g_m.head_dim; i++)
                    xb[i] += a * vc_row[i];
            }

        }

        // quantize and overwrite the attention result after multiplying with Wattention_output
        quantize_q80(g_m.xq,g_m.xb,g_m.n_embed);
        matmul(g_m.xb2,g_m.xq,g_m.tr[l].att_out,g_m.n_embed,g_m.n_embed);

        // residual connection back to x
        for (int j = 0; j < g_m.n_embed; j++) g_m.x[j] += g_m.xb2[j];

        // 6. Feed-Forward Network
        rmsnorm(g_m.xb,g_m.x,g_m.tr[l].ffn_norm,g_m.n_embed); // FFN RMS norm

        // Up / Gate (quantized matmuls)
        quantize_q80(g_m.xq,g_m.xb,g_m.n_embed);
        matmul(g_m.hb,g_m.xq,g_m.tr[l].ffn_up,g_m.n_embed,g_m.n_ff); // up
        matmul(g_m.hb2,g_m.xq,g_m.tr[l].ffn_gate,g_m.n_embed,g_m.n_ff); // gate

        // Apply SwiGLU: silu(gate) * up
        for (int i = 0; i < g_m.n_ff; i++) {
            const float g = g_m.hb2[i];
            const float silu_g = g / (1.0f + expf(-g));
            g_m.hb[i] = silu_g * g_m.hb[i];
        }

        // Down projection (quantize hidden, matmul)
        quantize_q80(g_m.xq,g_m.hb,g_m.n_ff);
        matmul(g_m.xb,g_m.xq,g_m.tr[l].ffn_down,g_m.n_ff,g_m.n_embed); // down

        // Residual back to x
        for (int j = 0; j < g_m.n_embed; j++) g_m.x[j] += g_m.xb[j];
    }

    // final RMS norm
    rmsnorm(g_m.x,g_m.x,g_m.t_outnorm,g_m.n_embed);

    // output into logits
    ftensor logs = (ftensor)malloc(g_m.vocab_size * sizeof(float));
    quantize_q80(g_m.xq,g_m.x,g_m.n_embed);
    matmul(logs,g_m.xq,g_m.t_out,g_m.n_embed,g_m.vocab_size);
    return logs;
}

int sampler(ftensor logits)
{
    if (!logits) return g_m.tok_eos;

    int best_id = 0;
    float best_v = logits[0];
    for (int i = 1; logits[i] && i < g_m.n_embed; i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best_id = i;
        }
    }

    return best_id;
}

void puttok(int tok)
{
    if (tok < 0 || tok >= g_m.vocab_size) printf(" <Error token %d> ",tok);
    else printf("%s",g_m.tokens[tok]);
    fflush(stdout);
}

void generate(int* prompt, int ntokens)
{
    ftensor logits = 0;
    int tok = 0;

    for (int i = 0; i < ntokens; i++) {
        tok = (*prompt)? *prompt++ : sampler(logits);
        if (logits) free(logits);
        if (!tok || tok == g_m.tok_eos) break;

        puttok(tok);
        if (i == ntokens-1) break; // no need to run inference anymore
        TIMING_START;
        logits = inference(tok,i);
        TIMING_STOP("inference");
    }
}

int main(int argc, char* argv[])
{
    puts("Welcome to LiGGUF C edition!");
    if (argc < 4) {
        printf("Usage: %s <model.gguf> <number_of_tokens_to_generate> <prompt>\n",argv[0]);
        return 0;
    }

    open_mmap(argv[1]);
    read_gguf();
    read_tokenizer();

    for (int i = 0; i < 65536; i++) fp1632_lut[i] = fp16_to_fp32((uint16_t)i);

    int ngen = atoi(argv[2]);
    int* toks = tokenize(argv[3],1,0);
    int ntok = 0;
    for (int i = 0; toks[i]; i++,ntok++) ;
    generate(toks,ntok+ngen);
    if (toks) free(toks);

    puts("\nDone.");
    return 0;
}
