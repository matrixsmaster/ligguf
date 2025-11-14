/* LiGGUF - a tiny, dependency-free LLaMA inference engine with direct GGUF support
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025
 *
 * C++, Debug-enabled version
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
#include <vector>
#include <string>
#include <map>

#define ALIGNMENT 32
#define MAXTENSORS 4000
#define MAXMETAKV 500
#define MAXNAMELEN 1024
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
#define OUTPUT_KEY "output.weight"
#define OUTPUT_NORM_KEY "output_norm.weight"
#define LAYER_ATT_NORM_KEY "blk.%d.attn_norm.weight"
#define LAYER_ATT_Q_KEY "blk.%d.attn_q.weight"
#define LAYER_ATT_K_KEY "blk.%d.attn_k.weight"
#define LAYER_ATT_V_KEY "blk.%d.attn_v.weight"
#define LAYER_ATT_OUT_KEY "blk.%d.attn_output.weight"
#define FF_LEN_KEY "llama.feed_forward_length"
#define LAYER_FFN_NORM_KEY "blk.%d.ffn_norm.weight"
#define LAYER_FFN_UP_KEY   "blk.%d.ffn_up.weight"    // w1
#define LAYER_FFN_DOWN_KEY "blk.%d.ffn_down.weight"  // w2
#define LAYER_FFN_GATE_KEY "blk.%d.ffn_gate.weight"  // w3
#define QK8_0 32 // From llama.cpp

#define ERR(S,...) fprintf(stderr,S "\n", __VA_ARGS__)
#if 1
    #define DBG(S,...) printf(S "\n",__VA_ARGS__)
#else
    #define DBG(...)
#endif

using namespace std;

enum gguf_type {
    F32, F16,
    Q4_0, Q4_1, Q4_2, Q4_3,
    Q5_0, Q5_1,
    Q8_0, Q8_1,
    Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K,
    IQ2_XXS, IQ2_XS, IQ3_XXS,
    IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS,
    I8, I16, I32, I64,
    F64,
    IQ1_M,
    GGUF_TYPE_COUNT
};

enum gguf_val_type {
    GUINT8, GINT8, GUINT16, GINT16, GUINT32, GINT32,
    GFLOAT32, GBOOL, GSTRING, GARRAY, GUINT64, GINT64, GFLOAT64,
    GGUF_VAL_TYPE_COUNT
};

#if 1
static const char* val_type_names[] = {
    "uint8", "int8", "uint16", "int16", "uint32", "int32",
    "float", "bool", "string", "array", "uint64", "int64", "double",
    "unknown"
};
#endif

struct gguf_kv {
    uint64_t off;
    gguf_val_type tag;
};

struct gguf_tensor {
    uint64_t off;
    vector<uint64_t> dims;
    gguf_type type;
};

struct block_q8_0 { // From llama.cpp
    uint16_t d;
    int8_t qs[QK8_0];
};

typedef vector<block_q8_0> qtensor;
typedef vector<float> ftensor;

struct model_state {
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

    vector<string> tokens; // vector of known tokens (pos == index)
    map<string,int> tokens_rev; // reverse map of tokens (by token string) for convenience
    ftensor tokscores; // token scores (const)
    map<string,gguf_kv> meta_kv; // key-value pairs with model's metadata
    map<string,gguf_tensor> tensors; // map of all tensors in model file

    ftensor x; // activation at current time stamp
    qtensor xq; // quantized x
    ftensor xb; // same, but inside a residual branch
    ftensor xb2; // buffer for raw attention result
    ftensor hb; // buffer for hidden dimension in the ffn
    ftensor hb2; // buffer for hidden dimension in the ffn
    ftensor q,k,v; // current QKV
    ftensor kc,vc; // KV cache
    ftensor att; // attention scores
};

model_state g_m;

void close_mmap()
{
    if (g_m.base && g_m.base != MAP_FAILED && g_m.fsize) {
        munmap(g_m.base, g_m.fsize);
    }
    if (g_m.file != -1) {
        close(g_m.file);
    }
    g_m.base = NULL;
    g_m.fsize = 0;
    g_m.file = -1;
}

bool open_mmap(const char* fn)
{
    g_m.file = open(fn,O_RDONLY);
    if (g_m.file == -1) {
        ERR("Can't open file %s",fn);
        return false;
    }

    struct stat st;
    if (fstat(g_m.file,&st)) {
        ERR("Can't stat file %s",fn);
        close_mmap();
        return false;
    }

    g_m.fsize = st.st_size;
    g_m.base = (uint8_t*)mmap(NULL,g_m.fsize,PROT_READ,MAP_SHARED,g_m.file,0);
    if (g_m.base == MAP_FAILED) {
        ERR("Can't mmap() file %s",fn);
        close_mmap();
        return false;
    }

    return true;
}

uint8_t inline rd8(uint8_t** pos)
{
    uint8_t r = 0;
    memcpy(&r,*pos,1);
    *pos += 1;
    return r;
}

uint16_t inline rd16(uint8_t** pos)
{
    uint16_t r = 0;
    memcpy(&r,*pos,2);
    *pos += 2;
    return r;
}

uint32_t inline rd32(uint8_t** pos)
{
    uint32_t r = 0;
    memcpy(&r,*pos,4);
    *pos += 4;
    return r;
}

uint64_t inline rd64(uint8_t** pos)
{
    uint64_t r = 0;
    memcpy(&r,*pos,8);
    *pos += 8;
    return r;
}

float inline rdf32(uint8_t** pos)
{
    float r = 0;
    memcpy(&r,*pos,4);
    *pos += 4;
    return r;
}

string rdstr(uint8_t** pos)
{
    uint64_t l = rd64(pos);
    string res((const char*)*pos,l);
    *pos += l;
    return res;
}

uint32_t inline kvrd32(string key)
{
    if (!g_m.meta_kv.count(key)) {
        ERR("Key '%s' not found!",key.c_str());
        return 0;
    }
    uint8_t* p = g_m.base + g_m.meta_kv[key].off;
    return rd32(&p);
}

float inline kvrdf32(string key)
{
    if (!g_m.meta_kv.count(key)) {
        ERR("Key '%s' not found!",key.c_str());
        return 0;
    }
    uint8_t* p = g_m.base + g_m.meta_kv[key].off;
    return rdf32(&p);
}

uint64_t skipper(gguf_val_type t, uint8_t* pos)
{
    switch(t){
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
            ERR("Unknown/unsupported tag %d",t);
            assert(false);
    }
}

bool read_gguf()
{
    uint8_t* p = g_m.base;

    if (memcmp(p,"GGUF",4)) {
        ERR("Wrong magic %4.4s",p);
        return false;
    }
    p += 4;

    uint32_t ver = rd32(&p);
    if (ver != 3) {
        ERR("Wrong file version (%u)",ver);
        return false;
    }

    uint64_t nten = rd64(&p);
    if (!nten || nten > MAXTENSORS) {
        ERR("Suspicious number of tensors: %lu",nten);
        return false;
    }

    uint64_t meta = rd64(&p);
    if (!meta || meta > MAXMETAKV) {
        ERR("Suspicious number of KV pairs: %lu",meta);
        return false;
    }

    gguf_kv kv;
    for (uint64_t i = 0; i < meta; i++) {
        string key = rdstr(&p);

        kv.tag = (gguf_val_type)rd32(&p);
        kv.off = p - g_m.base;

        string val;
        if (kv.tag == GSTRING) val = rdstr(&p);
        else p += skipper(kv.tag,p);

        g_m.meta_kv[key] = kv;

        // DEBUG block
        DBG("Key '%s' read, type is %s",key.c_str(),((kv.tag < GGUF_VAL_TYPE_COUNT)? val_type_names[kv.tag]:val_type_names[GGUF_VAL_TYPE_COUNT]));
        switch (kv.tag) {
            case GSTRING: DBG("Value = '%s'",val.c_str()); break;
            case GUINT32: DBG("Value = %u",kvrd32(key)); break;
            case GINT32: DBG("Value = %i",(int32_t)kvrd32(key)); break;
            case GFLOAT32: DBG("Value = %f",kvrdf32(key)); break;
            default: break;
        }
    }

    for (uint64_t i=0; i < nten; i++) {
        string key = rdstr(&p);
        uint32_t ndim = rd32(&p);

        gguf_tensor tz;
        tz.dims.resize(ndim);
        for (uint32_t j = 0; j < ndim; j++)
            tz.dims[j] = rd64(&p);

        tz.type = (gguf_type)rd32(&p);
        tz.off = rd64(&p);
        g_m.tensors[key] = tz;

        DBG("Tensor '%s' read, type %d, offset %lu",key.c_str(),tz.type,tz.off);
    }

    uint64_t off = p - g_m.base;
    g_m.tensors_off = g_m.base + (off + (ALIGNMENT - (off % ALIGNMENT)) % ALIGNMENT);
    DBG("Tensor base is %zu",g_m.tensors_off-g_m.base);

    g_m.vocab_size = kvrd32(VOCAB_SIZE_KEY);
    g_m.n_layers = kvrd32(BLOCK_COUNT_KEY);
    g_m.n_embed = kvrd32(EMBED_LEN_KEY);
    g_m.n_heads = kvrd32(HEAD_COUNT_KEY);
    g_m.n_kv_heads = kvrd32(HEAD_KV_COUNT_KEY);
    g_m.n_context = kvrd32(CONTEXT_LEN_KEY);
    g_m.tok_bos = kvrd32(TOKENS_BOS_KEY);
    g_m.tok_eos = kvrd32(TOKENS_EOS_KEY);
    g_m.rms_epsilon = kvrdf32(RMS_EPSILON_KEY);
    g_m.rope_base = kvrdf32(ROPE_BASE_KEY);
    g_m.rope_dim = kvrd32(ROPE_DIMS_KEY);
    g_m.n_ff = kvrd32(FF_LEN_KEY);

    g_m.head_dim = g_m.n_embed / g_m.n_heads;
    assert(g_m.head_dim * g_m.n_heads == g_m.n_embed);

    return true;
}

bool read_tokenizer()
{
    if (!g_m.meta_kv.count(TOKENS_KEY)) {
        ERR("Tokens array %s not found!",TOKENS_KEY);
        return false;
    }

    uint8_t* pos = g_m.base + g_m.meta_kv[TOKENS_KEY].off;

    gguf_val_type nt = (gguf_val_type)rd32(&pos);
    assert(nt == GSTRING);

    uint64_t ne = rd64(&pos);
    assert((int)ne == g_m.vocab_size);
    g_m.tokens.resize(ne);

    for (uint64_t i = 0; i < ne; i++) {
        g_m.tokens[i] = rdstr(&pos);
        g_m.tokens_rev[g_m.tokens[i]] = i;
    }

    if (!g_m.meta_kv.count(TOKENS_SCORE_KEY)) {
        ERR("Tokens scores array %s not found!",TOKENS_SCORE_KEY);
        return false;
    }

    pos = g_m.base + g_m.meta_kv[TOKENS_SCORE_KEY].off;
    nt = (gguf_val_type)rd32(&pos);
    assert(nt == GFLOAT32);

    ne = rd64(&pos);
    g_m.tokscores.resize(ne);
    for (uint64_t i = 0; i < ne; i++)
        g_m.tokscores[i] = rdf32(&pos);

    return true;
}

vector<int> tokenize(const char* str, bool bos, bool eos)
{
    vector<int> out;
    out.reserve(strlen(str)+2);
    if (bos) out.push_back(g_m.tok_bos);

    while (*str) {
        string s;
        s += *str;
        if (g_m.tokens_rev.count(s))
            out.push_back(g_m.tokens_rev.at(s));
        else {
            s = "<0x00>";
            snprintf(s.data(),s.length()+1,"<0x%02X>",*str);
            out.push_back(g_m.tokens_rev.at(s));
        }
        str++;
        //DBG("Dumb token added: %d '%s'",out.back(),g_m.tokens.at(out.back()).c_str());
    }

    while (1) {
        float best_score = -1e10;
        unsigned best_id = -1;
        unsigned best_idx = out.size()+1;

        string acc;
        for (unsigned i = 0; i < out.size()-1; i++) {
            acc = g_m.tokens.at(out.at(i)) + g_m.tokens.at(out.at(i+1));
            if (!g_m.tokens_rev.count(acc)) continue;

            float sc = g_m.tokscores[g_m.tokens_rev[acc]];
            if (sc > best_score) {
                best_score = sc;
                best_id = g_m.tokens_rev[acc];
                best_idx = i;
            }
        }

        if (best_idx > out.size()) break;

        //DBG("Pair merged at %u: %s",best_idx,acc.c_str());
        out[best_idx] = best_id;
        out.erase(out.begin()+best_idx+1);
    }

    if (eos) out.push_back(g_m.tok_eos);

    for (auto i = out.begin(); i < out.end(); i++)
        DBG("Final token: %d '%s'",*i,g_m.tokens.at(*i).c_str());

    return out;
}

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float fp16_to_fp32(uint16_t h) {
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

static inline uint16_t fp32_to_fp16(float f) {
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

void dequant_q80(ftensor &y, block_q8_0* ptr, uint64_t nrow, int len)
{
    block_q8_0* x = ptr + nrow * (len / QK8_0);
    DBG("dequant row %lu (len %i) @ offset %lu",nrow,len,((uint8_t*)x-g_m.base));
    const int nb = len / QK8_0;
    y.resize(len);

    float vmin = 1e30, vmax = 1e-30; // FIXME: debug only
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);

        for (int j = 0; j < QK8_0; ++j) {
            const float val = x[i].qs[j]*d;
            y[i*QK8_0 + j] = val;
            if (val > vmax) vmax = val;
            if (val < vmin) vmin = val;
            if (i < 2) DBG("%d = %f",j,val);
        }
    }

    DBG("dequant_q80: vmin = %f, vmax = %f",vmin,vmax);
}

void quantize_q80(qtensor &y, const ftensor &x)
{
    const int nb = x.size() / QK8_0;
    y.resize((size_t)nb * QK8_0);

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = max(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = fp32_to_fp16(d);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}

void rmsnorm(ftensor &out, const ftensor &x, const char* weights, int layer)
{
    char str[MAXNAMELEN] = {0};
    snprintf(str,sizeof(str),weights,layer);
    //DBG("RMS norm with weights in %s",str);
    assert(g_m.tensors.count(str));

    const gguf_tensor &tz = g_m.tensors.at(str);
    assert(tz.dims[0] == x.size());

    float ss = 0.0f;
    for (int i = 0; i < (int)x.size(); i++)
        ss += x[i] * x[i];

    //DBG("RMS norm sum = %f",ss);
    ss = ss / (float)x.size() + g_m.rms_epsilon;
    ss = 1.0f / sqrtf(ss);

    out.resize(x.size());
    float* p = (float*)(g_m.tensors_off + tz.off);
    for (int i = 0; i < (int)x.size(); i++,p++) {
        out[i] = x[i] * ss * (*p);
        //DBG(" rms %d w = %.5f",i,*p);
    }

    DBG("RMS norm scale = %f",ss);
}

void inline matmul(ftensor &out, block_q8_0* qx, block_q8_0* qw, int n, int d)
{
    const int nb = n / QK8_0;
    float maxacc = 0; // debug
    out.resize(d);

    for (int r = 0; r < d; r++) { // each row
        float acc = 0.0;

        for (int b = 0; b < nb; b++) { // each block
            int iw = r * nb + b;
            const float sx = fp16_to_fp32(qx[b].d);
            const float sw = fp16_to_fp32(qw[iw].d);

            // integer dot
            int32_t s = 0;
            for (int i = 0; i < QK8_0; i++)
                s += (int32_t)(qx[b].qs[i]) * (int32_t)(qw[iw].qs[i]);

            // scale and accumulate result as float
            acc += sw * sx * (float)s;
        }

        out[r] = (float)acc;
        if (fabs(acc) > maxacc) maxacc = fabs(acc); // debug
    }
    DBG(" matmul max acc = %f",maxacc);
}

void matmul_wrap(ftensor &out, block_q8_0* qx, int n, int d, const char* weights, int layer)
{
    char str[MAXNAMELEN] = {0};
    snprintf(str,sizeof(str),weights,layer);
    DBG("matmul with weights in %s with shape %d : %d",str,n,d);
    assert(g_m.tensors.count(str));

    const gguf_tensor &tz = g_m.tensors.at(str);
    assert(tz.type == Q8_0);
    assert(tz.dims.size() == 2);
    DBG("Tensor shape is %lu : %lu",tz.dims[0],tz.dims[1]);
    assert((int)tz.dims[0] == n);
    assert((int)tz.dims[1] == d);

    matmul(out,qx,(block_q8_0*)(g_m.tensors_off+tz.off),n,d);
}

void rope(ftensor& x, int n_heads, int pos)
{
    assert((int)x.size() == n_heads * g_m.head_dim);
    int rope_dim = g_m.rope_dim;
    if (rope_dim > g_m.head_dim) rope_dim = g_m.head_dim;
    if (rope_dim & 1) rope_dim--; // ensure even

    for (int h = 0; h < n_heads; h++) {
        float* v = x.data() + h * g_m.head_dim;

        for (int i = 0; i < rope_dim; i += 2) {
            // pair index m in [0 .. rope_dim/2)
            const int m = i >> 1;

            // standard RoPE frequency schedule:
            // inv_freq[m] = base^(-2m/rope_dim), angle = pos * inv_freq[m]
            const float ang = pos * powf(g_m.rope_base, -2.0f * (float)m / (float)rope_dim);
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
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

ftensor inference(int tok, int pos)
{
    ftensor logs;
    DBG("inference() for token %d at pos %d",tok,pos);

    assert(g_m.tensors.count(TOKENS_EMBED_KEY));
    gguf_tensor const &emb = g_m.tensors.at(TOKENS_EMBED_KEY);
    assert(emb.dims.size() == 2);
    assert(emb.type == Q8_0);
    assert((int)emb.dims[0] == g_m.n_embed);
    assert((int)emb.dims[1] == g_m.vocab_size);

    const int kv_dim = (g_m.n_embed * g_m.n_kv_heads) / g_m.n_heads;

    // input is the embedding vector for the current token
    dequant_q80(g_m.x,(block_q8_0*)(g_m.tensors_off+emb.off),tok,g_m.n_embed);

    // for all layers (blocks) of the model
    for (int l = 0; l < g_m.n_layers; l++) {
        // 1. Attention RMS norm
        rmsnorm(g_m.xb,g_m.x,LAYER_ATT_NORM_KEY,l);

        // 2. QKV over quantized x
        quantize_q80(g_m.xq,g_m.xb);
        matmul_wrap(g_m.q,g_m.xq.data(),g_m.n_embed,g_m.n_embed,LAYER_ATT_Q_KEY,l);
        matmul_wrap(g_m.k,g_m.xq.data(),g_m.n_embed,kv_dim,LAYER_ATT_K_KEY,l);
        matmul_wrap(g_m.v,g_m.xq.data(),g_m.n_embed,kv_dim,LAYER_ATT_V_KEY,l);

        // 3. RoPE Q & K (float vectors)
        rope(g_m.q,g_m.n_heads,pos);
        rope(g_m.k,g_m.n_kv_heads,pos);

#if 0
        for (int t = 0; t < 4; ++t) DBG("Q_rot[%d]=%g  K_rot[%d]=%g", t, g_m.q[t], t, g_m.k[t]);
#endif

        // 4. Simple KV cache
        size_t loff = l * g_m.n_context * kv_dim; // kv cache layer offset
        g_m.kc.resize(g_m.n_layers * g_m.n_context * kv_dim);
        g_m.vc.resize(g_m.kc.size());
        float* kc_row = g_m.kc.data() + loff + pos * kv_dim;
        float* vc_row = g_m.vc.data() + loff + pos * kv_dim;
        memcpy(kc_row,g_m.k.data(),kv_dim*sizeof(float));
        memcpy(vc_row,g_m.v.data(),kv_dim*sizeof(float));

        // 5. Multi-Head Attention
        g_m.xb.resize(g_m.n_embed);
        g_m.att.resize(g_m.n_heads * g_m.n_context);
        const int nrep = g_m.n_heads / g_m.n_kv_heads; // GQA, heads per KV head
        for (int h = 0; h < g_m.n_heads; h++) {
            float* att = g_m.att.data() + (h * g_m.n_context); // attention scores for this head
            float* q = g_m.q.data() + (h * g_m.head_dim); // start of the Q vector

            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                kc_row = g_m.kc.data() + loff + t * kv_dim + (h / nrep) * g_m.head_dim;

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
            float* xb = g_m.xb.data() + (h * g_m.head_dim);
            memset(xb,0,g_m.head_dim * sizeof(float));

            for (int t = 0; t <= pos; t++) {
                vc_row = g_m.vc.data() + loff + t * kv_dim + (h / nrep) * g_m.head_dim; // value vector for this head and at this timestep
                float a = att[t]; // attention weight for this timestep
                // accumulate the weighted value
                for (int i = 0; i < g_m.head_dim; i++)
                    xb[i] += a * vc_row[i];
            }

        }

        // quantize and overwrite the attention result after multiplying with Wattention_output
        quantize_q80(g_m.xq,g_m.xb);
        matmul_wrap(g_m.xb2,g_m.xq.data(),g_m.n_embed,g_m.n_embed,LAYER_ATT_OUT_KEY,l);

        // residual connection back to x
        float maxabs = 0.0f; // debug
        for (int j = 0; j < g_m.n_embed; ++j) {
            g_m.x[j] += g_m.xb2[j];
            if (fabsf(g_m.x[j]) > maxabs) maxabs = fabsf(g_m.x[j]);
        }
        DBG("attn resid max |x| = %f", maxabs);

        // 6. Feed-Forward Network
        rmsnorm(g_m.xb,g_m.x,LAYER_FFN_NORM_KEY,l); // FFN RMS norm

        // Up / Gate (quantized matmuls)
        quantize_q80(g_m.xq,g_m.xb);
        matmul_wrap(g_m.hb,g_m.xq.data(),g_m.n_embed,g_m.n_ff,LAYER_FFN_UP_KEY,l); // up
        matmul_wrap(g_m.hb2,g_m.xq.data(),g_m.n_embed,g_m.n_ff,LAYER_FFN_GATE_KEY,l); // gate

        // apply SwiGLU correctly: silu(gate) * up
        float ffn_max = 0.0f; // debug
        for (int i = 0; i < g_m.n_ff; ++i) {
            float g = g_m.hb2[i];
            float silu_g = g / (1.0f + expf(-g));
            g_m.hb[i] = silu_g * g_m.hb[i];
            if (fabsf(g_m.hb[i]) > ffn_max) ffn_max = fabsf(g_m.hb[i]);
        }
        DBG("L%d ffn hidden max |h| = %f",l,ffn_max);

        // Down projection (quantize hidden, matmul)
        quantize_q80(g_m.xq,g_m.hb);
        matmul_wrap(g_m.xb,g_m.xq.data(),g_m.n_ff,g_m.n_embed,LAYER_FFN_DOWN_KEY,l); // w2

        // Residual back to x
        float maxabs_ff = 0.0f; // debug
        for (int j = 0; j < g_m.n_embed; ++j) {
            g_m.x[j] += g_m.xb[j];
            if (fabsf(g_m.x[j]) > maxabs_ff) maxabs_ff = fabsf(g_m.x[j]);
        }
        DBG("L%d ffn resid max |x| = %f",l,maxabs_ff);
    }

    // final RMS norm
    rmsnorm(g_m.x,g_m.x,OUTPUT_NORM_KEY,0);

    // output into logits
    quantize_q80(g_m.xq,g_m.x);
    matmul_wrap(logs,g_m.xq.data(),g_m.n_embed,g_m.vocab_size,OUTPUT_KEY,0);

    return logs;
}

int sampler(const ftensor &logits)
{
    if (logits.empty()) {
        DBG("sampler: empty logits, returning EOS (%d)",g_m.tok_eos);
        return g_m.tok_eos;
    }

    int best_id = 0;
    float best_v = logits[0];
    for (int i = 1; i < (int)logits.size(); ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best_id = i;
        }
    }

    DBG("greedy: picked %d '%s' (logit=%f)", best_id,
        (best_id >= 0 && best_id < (int)g_m.tokens.size()) ? g_m.tokens[best_id].c_str() : "<out-of-range>",
        best_v);

    return best_id;
}

void puttok(int tok)
{
    if (tok < 0 || tok >= g_m.vocab_size) printf(" <Error token %d> ",tok);
    else printf("%s",g_m.tokens[tok].c_str());
    fflush(stdout);
}

void generate(vector<int> prompt, int ntokens)
{
    ftensor logits;

    int tok = g_m.tok_eos;
    for (int i = 0; i < ntokens; i++) {
        if (i < (int)prompt.size())
            tok = prompt.at(i);
        else
            tok = sampler(logits);

        if (!tok || tok == g_m.tok_eos) break;

        logits = inference(tok,i);
        puttok(tok);
    }
    puttok(tok);
}

vector<int> read_tokens(int argc, char* argv[], int first)
{
    vector<int> r;
    for (int i = first; i < argc; i++) r.push_back(atoi(argv[i]));
    return r;
}

int main(int argc, char* argv[])
{
    assert(argc > 1);
    assert(open_mmap(argv[1]));
    assert(read_gguf());
    assert(read_tokenizer());

    if (argc > 2) {
        vector<int> toks;
        if (strcmp(argv[2],"tokens"))
            toks = tokenize(argv[2],true,false);
        else
            toks = read_tokens(argc,argv,3);

        generate(toks,toks.size()+1);
    }

#if 1
    puts("");
    DBG("Size of x = %zu",g_m.x.size());
    DBG("Size of xb = %zu",g_m.xb.size());
    DBG("Size of xb2 = %zu",g_m.xb2.size());
    DBG("Size of xq = %zu",g_m.xq.size());
    DBG("Size of hb = %zu",g_m.hb.size());
    DBG("Size of hb2 = %zu",g_m.hb2.size());
    DBG("Size of q = %zu",g_m.q.size());
    DBG("Size of k = %zu",g_m.k.size());
    DBG("Size of v = %zu",g_m.v.size());
    DBG("Size of kc = %zu",g_m.kc.size());
    DBG("Size of vc = %zu",g_m.vc.size());
    DBG("Size of att = %zu",g_m.att.size());
#endif

    puts("\nDone.");
    close_mmap();
    return 0;
}
