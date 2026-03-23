/* LiGGUF - a tiny, dependency-free LLaMA inference engine with direct GGUF support
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025-2026
 *
 * Distributed execution version in a single C++ file
 * */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <time.h>
#include <vector>
#include <string>
#include <map>

#define ALIGNMENT 32
#define MAXTENSORS 4000
#define MAXMETAKV 500
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
#define OUTPUT_KEY "output.weight"
#define OUTPUT_NORM_KEY "output_norm.weight"
#define LAYER_ATT_NORM_KEY "blk.%d.attn_norm.weight"
#define LAYER_ATT_Q_KEY "blk.%d.attn_q.weight"
#define LAYER_ATT_K_KEY "blk.%d.attn_k.weight"
#define LAYER_ATT_V_KEY "blk.%d.attn_v.weight"
#define LAYER_ATT_OUT_KEY "blk.%d.attn_output.weight"
#define FF_LEN_KEY "llama.feed_forward_length"
#define LAYER_FFN_NORM_KEY "blk.%d.ffn_norm.weight"
#define LAYER_FFN_UP_KEY   "blk.%d.ffn_up.weight"
#define LAYER_FFN_DOWN_KEY "blk.%d.ffn_down.weight"
#define LAYER_FFN_GATE_KEY "blk.%d.ffn_gate.weight"

#define MULTITHREAD _Pragma("omp parallel for")

#ifdef DEBUG
#define DBG(S,...) fprintf(stderr,"DBG: " S "\n", ##__VA_ARGS__)
#else
#define DBG(S,...)
#endif
#define ERR(S,...) fprintf(stderr,"ERR: " S "\n", ##__VA_ARGS__)

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

struct gguf_kv {
    uint64_t off;
    gguf_val_type tag;
};

struct gguf_tensor {
    uint64_t off;
    vector<uint64_t> dims;
    gguf_type type;
};

struct block_q8_0 {
    uint16_t d;
    int8_t qs[QK8_0];
};

struct trans_block {
    float* att_norm;
    block_q8_0* att_q;
    block_q8_0* att_k;
    block_q8_0* att_v;
    block_q8_0* att_out;
    float* ffn_norm;
    block_q8_0* ffn_up;
    block_q8_0* ffn_down;
    block_q8_0* ffn_gate;
};

struct shard_desc {
    int kv_head0, kv_headn;
    int q_head0, q_headn;
    int ff0, ffn;
};

enum net_msg_kind {
    NET_NONE,
    NET_HELLO,
    NET_ATT,
    NET_FFN,
    NET_STOP
};

struct net_hdr {
    uint32_t kind;
    int32_t a,b,c;
};

struct net_peer {
    string host;
    int port,fd,rank;
};

struct run_stats {
    double wall,gen_wall,tok_s,per_token;
    int ngen;
    vector<int> out;
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
    int rope_dim; // RoPE dimension
    int n_ff; // Feed-Forward length
    int kv_dim; // Key-Value dimension
    int nrep; // Query-to-KV head replication factor

    vector<string> tokens; // vector of known tokens (pos == index)
    map<string,int> tokens_rev; // reverse map of tokens
    ftensor tokscores; // token scores
    map<string,gguf_kv> meta_kv; // model metadata
    map<string,gguf_tensor> tensors; // model tensor table
    block_q8_0* t_embed; // embedding tensor
    block_q8_0* t_out; // output classifier weights
    float* t_outnorm; // output RMS norm weights
    vector<trans_block> tr; // transformer blocks

    ftensor x; // activation at current time stamp
    ftensor xb; // activation inside a residual branch
    qtensor xq; // quantized x
    ftensor logits; // output logits

    ftensor q; // local Q projection result
    ftensor k; // local K projection result
    ftensor v; // local V projection result
    ftensor xb_local; // local attention branch output before att_out
    ftensor att; // local attention scores
    ftensor hb; // local ffn_up result
    ftensor hb2; // local ffn_gate result
    ftensor partial; // local full-width partial to reduce on master
    ftensor recv_partial; // dequantized worker partial received on master
    qtensor partial_q; // quantized local partial for transport
    qtensor recv_partial_q; // quantized worker partial receive buffer
    qtensor net_xq; // quantized activation transport buffer
    ftensor kc,vc; // local KV cache for the shard

    int rank; // local rank, 0 for master
    int nranks; // total rank count
    bool master_mode; // true when workers are attached
    bool worker_mode; // true for worker-only process
    int listen_port; // worker listen port
    int worker_fd; // accepted master connection
    vector<net_peer> peers; // worker peers owned by the master
    vector<shard_desc> shards; // precomputed shard assignment per rank
    shard_desc self; // local shard assignment
};

model_state g_m;
float fp1632_lut[65536];

double now_sec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#define RDMEM(T,N) T inline N (uint8_t** pos)\
{\
    T r = 0;\
    memcpy(&r,*pos,sizeof(T));\
    *pos += sizeof(T);\
    return r;\
}

RDMEM(uint8_t,rd8)
RDMEM(uint16_t,rd16)
RDMEM(uint32_t,rd32)
RDMEM(uint64_t,rd64)
RDMEM(float,rdf32)

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

static inline float fp16_to_fp32(uint16_t h)
{
    const uint32_t w = (uint32_t)h << 16;
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

void close_mmap()
{
    if (g_m.base && g_m.base != MAP_FAILED && g_m.fsize) munmap(g_m.base,g_m.fsize);
    if (g_m.file != -1) close(g_m.file);
    g_m.base = NULL;
    g_m.fsize = 0;
    g_m.file = -1;
}

bool open_mmap(const char* fn)
{
    g_m.fsize = 0;
    g_m.base = NULL;
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
        g_m.meta_kv[key] = kv;
        p += skipper(kv.tag,p);
    }

    for (uint64_t i = 0; i < nten; i++) {
        string key = rdstr(&p);
        uint32_t ndim = rd32(&p);
        gguf_tensor tz;
        tz.dims.resize(ndim);
        for (uint32_t j = 0; j < ndim; j++) tz.dims[j] = rd64(&p);
        tz.type = (gguf_type)rd32(&p);
        tz.off = rd64(&p);
        g_m.tensors[key] = tz;
    }

    uint64_t off = p - g_m.base;
    g_m.tensors_off = g_m.base + (off + (ALIGNMENT - (off % ALIGNMENT)) % ALIGNMENT);

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
    g_m.kv_dim = (g_m.n_embed * g_m.n_kv_heads) / g_m.n_heads;
    g_m.nrep = g_m.n_heads / g_m.n_kv_heads;

    g_m.x.resize(g_m.n_embed);
    g_m.xb.resize(g_m.n_embed);
    g_m.logits.resize(g_m.vocab_size);

    g_m.tr.resize(g_m.n_layers);
    memset(g_m.tr.data(),0,g_m.n_layers * sizeof(trans_block));

    for (map<string,gguf_tensor>::const_iterator it = g_m.tensors.begin(); it != g_m.tensors.end(); ++it) {
        const string& key = it->first;
        uint8_t* ptr = g_m.tensors_off + it->second.off;

        if (!key.compare(0,4,"blk.") && key.find(".weight") != string::npos) {
            int id = atoi(key.c_str()+4);
            assert(id >= 0 && id < g_m.n_layers);

            if (key.find("attn_norm") != string::npos) g_m.tr[id].att_norm = (float*)ptr;
            else if (key.find("attn_q") != string::npos) g_m.tr[id].att_q = (block_q8_0*)ptr;
            else if (key.find("attn_k") != string::npos) g_m.tr[id].att_k = (block_q8_0*)ptr;
            else if (key.find("attn_v") != string::npos) g_m.tr[id].att_v = (block_q8_0*)ptr;
            else if (key.find("attn_output") != string::npos) g_m.tr[id].att_out = (block_q8_0*)ptr;
            else if (key.find("ffn_norm") != string::npos) g_m.tr[id].ffn_norm = (float*)ptr;
            else if (key.find("ffn_up") != string::npos) g_m.tr[id].ffn_up = (block_q8_0*)ptr;
            else if (key.find("ffn_down") != string::npos) g_m.tr[id].ffn_down = (block_q8_0*)ptr;
            else if (key.find("ffn_gate") != string::npos) g_m.tr[id].ffn_gate = (block_q8_0*)ptr;
        }
        else if (key == TOKENS_EMBED_KEY) g_m.t_embed = (block_q8_0*)ptr;
        else if (key == OUTPUT_KEY) g_m.t_out = (block_q8_0*)ptr;
        else if (key == OUTPUT_NORM_KEY) g_m.t_outnorm = (float*)ptr;
    }

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
    for (uint64_t i = 0; i < ne; i++) g_m.tokscores[i] = rdf32(&pos);

    return true;
}

void dequant_q80(ftensor &y, block_q8_0* ptr, uint64_t nrow, int len)
{
    block_q8_0* x = ptr + nrow * (len / QK8_0);
    const int nb = len / QK8_0;
    if ((int)y.size() < len) y.resize(len);

    MULTITHREAD
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_0; j++)
            y[i*QK8_0 + j] = x[i].qs[j] * d;
    }
}

void quantize_q80(qtensor &y, const ftensor &x)
{
    const int nb = x.size() / QK8_0;
    if ((int)y.size() < nb) y.resize(nb);

    MULTITHREAD
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = max(amax,fabsf(v));
        }
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = fp32_to_fp16(d);
        for (int j = 0; j < QK8_0; j++)
            y[i].qs[j] = roundf(x[i*QK8_0 + j] * id);
    }
}

void rmsnorm(ftensor &out, const ftensor &x, float* w, int size)
{
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = ss / (float)size + g_m.rms_epsilon;
    ss = 1.0f / sqrtf(ss);
    if ((int)out.size() < size) out.resize(size);

    for (int i = 0; i < size; i++) out[i] = x[i] * ss * w[i];
}

void inline matmul(ftensor &out, block_q8_0* qx, block_q8_0* qw, int n, int d)
{
    const int nb = n / QK8_0;
    if ((int)out.size() < d) out.resize(d);

    MULTITHREAD
    for (int r = 0; r < d; r++) {
        float acc = 0.0f;
        for (int b = 0; b < nb; b++) {
            int iw = r * nb + b;
            int32_t s = 0;
            for (int i = 0; i < QK8_0; i++) s += (int32_t)qx[b].qs[i] * (int32_t)qw[iw].qs[i];
            acc += fp1632_lut[qx[b].d] * fp1632_lut[qw[iw].d] * (float)s;
        }
        out[r] = acc;
    }
}

void inline matmul_partial_cols(ftensor &out, block_q8_0* qx, block_q8_0* qw, int n, int d, int col0, int coln)
{
    const int nb = n / QK8_0;
    const int b0 = col0 / QK8_0;
    const int bn = coln / QK8_0;
    if ((int)out.size() < d) out.resize(d);

    MULTITHREAD
    for (int r = 0; r < d; r++) {
        float acc = 0.0f;
        for (int b = b0; b < bn; b++) {
            int iw = r * nb + b;
            int32_t s = 0;
            const block_q8_0 &iq = qx[b - b0];
            const block_q8_0 &w = qw[iw];
            for (int i = 0; i < QK8_0; i++) s += (int32_t)iq.qs[i] * (int32_t)w.qs[i];
            acc += fp1632_lut[iq.d] * fp1632_lut[w.d] * (float)s;
        }
        out[r] = acc;
    }
}

void rope(ftensor& x, int n_heads, int pos)
{
    int rope_dim = g_m.rope_dim;
    if (rope_dim > g_m.head_dim) rope_dim = g_m.head_dim;
    if (rope_dim & 1) rope_dim--;

    MULTITHREAD
    for (int h = 0; h < n_heads; h++) {
        float* v = x.data() + h * g_m.head_dim;
        for (int i = 0; i < rope_dim; i += 2) {
            const int m = i >> 1;
            const float ang = pos * powf(g_m.rope_base,-2.0f * (float)m / (float)rope_dim);
            const float c = cosf(ang), s = sinf(ang);
            const float x0 = v[i];
            const float x1 = v[i+1];
            v[i] = x0 * c - x1 * s;
            v[i+1] = x0 * s + x1 * c;
        }
    }
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

int sampler_greedy(const ftensor &logits)
{
    if (logits.empty()) return g_m.tok_eos;
    int best_id = 0;
    float best_v = logits[0];
    for (int i = 1; i < (int)logits.size(); i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best_id = i;
        }
    }
    return best_id;
}

void split_even(int total, int idx, int parts, int &off, int &count)
{
    int base = total / parts;
    int rem = total % parts;
    count = base + (idx < rem);
    off = idx * base + (idx < rem ? idx : rem);
}

void build_shards(int nranks)
{
    g_m.shards.resize(nranks);
    for (int r = 0; r < nranks; r++) {
        shard_desc &sd = g_m.shards[r];
        split_even(g_m.n_kv_heads,r,nranks,sd.kv_head0,sd.kv_headn);
        sd.q_head0 = sd.kv_head0 * g_m.nrep;
        sd.q_headn = sd.kv_headn * g_m.nrep;
        split_even(g_m.n_ff / QK8_0,r,nranks,sd.ff0,sd.ffn);
        sd.ff0 *= QK8_0;
        sd.ffn *= QK8_0;
        DBG("rank=%d kv=[%d,%d) q=[%d,%d) ff=[%d,%d)",r,sd.kv_head0,sd.kv_head0+sd.kv_headn,sd.q_head0,sd.q_head0+sd.q_headn,sd.ff0,sd.ff0+sd.ffn);
    }
    g_m.self = g_m.shards[g_m.rank];
}

void alloc_runtime()
{
    const int qdim = g_m.self.q_headn * g_m.head_dim;
    const int kdim = g_m.self.kv_headn * g_m.head_dim;
    const int ffblocks = g_m.self.ffn / QK8_0;
    const int eblocks = g_m.n_embed / QK8_0;

    g_m.x.assign(g_m.n_embed,0.0f);
    g_m.xb.assign(g_m.n_embed,0.0f);
    g_m.xq.resize(max(eblocks,ffblocks));
    g_m.logits.assign(g_m.vocab_size,0.0f);

    g_m.q.assign(qdim,0.0f);
    g_m.k.assign(kdim,0.0f);
    g_m.v.assign(kdim,0.0f);
    g_m.xb_local.assign(qdim,0.0f);
    g_m.att.assign((size_t)max(1,g_m.self.q_headn) * g_m.n_context,0.0f);
    g_m.hb.assign(g_m.self.ffn,0.0f);
    g_m.hb2.assign(g_m.self.ffn,0.0f);
    g_m.partial.assign(g_m.n_embed,0.0f);
    g_m.recv_partial.assign(g_m.n_embed,0.0f);
    g_m.partial_q.resize(eblocks);
    g_m.recv_partial_q.resize(eblocks);
    g_m.net_xq.resize(eblocks);
    g_m.kc.assign((size_t)g_m.n_layers * g_m.n_context * kdim,0.0f);
    g_m.vc.assign(g_m.kc.size(),0.0f);
}

void att_partial(int l, int pos, block_q8_0* xq, ftensor &out)
{
    const shard_desc &sd = g_m.self;
    const int qdim = sd.q_headn * g_m.head_dim;
    const int kdim = sd.kv_headn * g_m.head_dim;
    const int qoff = sd.q_head0 * g_m.head_dim;
    const int nb = g_m.n_embed / QK8_0;

    matmul(g_m.q,xq,g_m.tr[l].att_q + qoff * nb,g_m.n_embed,qdim);
    matmul(g_m.k,xq,g_m.tr[l].att_k + sd.kv_head0 * g_m.head_dim * nb,g_m.n_embed,kdim);
    matmul(g_m.v,xq,g_m.tr[l].att_v + sd.kv_head0 * g_m.head_dim * nb,g_m.n_embed,kdim);

    rope(g_m.q,sd.q_headn,pos);
    rope(g_m.k,sd.kv_headn,pos);

    size_t loff = (size_t)l * g_m.n_context * kdim;
    float* kc_row = g_m.kc.data() + loff + pos * kdim;
    float* vc_row = g_m.vc.data() + loff + pos * kdim;
    memcpy(kc_row,g_m.k.data(),kdim*sizeof(float));
    memcpy(vc_row,g_m.v.data(),kdim*sizeof(float));

    for (int h = 0; h < sd.q_headn; h++) {
        float* att = g_m.att.data() + (h * g_m.n_context);
        float* q = g_m.q.data() + h * g_m.head_dim;
        for (int t = 0; t <= pos; t++) {
            kc_row = g_m.kc.data() + loff + t * kdim + (h / g_m.nrep) * g_m.head_dim;
            float score = 0.0f;
            for (int i = 0; i < g_m.head_dim; i++) score += q[i] * kc_row[i];
            att[t] = score / sqrtf(g_m.head_dim);
        }
        softmax(att,pos+1);
        float* xb = g_m.xb_local.data() + h * g_m.head_dim;
        memset(xb,0,g_m.head_dim*sizeof(float));
        for (int t = 0; t <= pos; t++) {
            vc_row = g_m.vc.data() + loff + t * kdim + (h / g_m.nrep) * g_m.head_dim;
            float a = att[t];
            for (int i = 0; i < g_m.head_dim; i++) xb[i] += a * vc_row[i];
        }
    }

    quantize_q80(g_m.net_xq,g_m.xb_local);
    matmul_partial_cols(out,g_m.net_xq.data(),g_m.tr[l].att_out,g_m.n_embed,g_m.n_embed,qoff,qoff+qdim);
}

void ffn_partial(int l, block_q8_0* xq, ftensor &out)
{
    const shard_desc &sd = g_m.self;
    const int nb = g_m.n_embed / QK8_0;

    matmul(g_m.hb,xq,g_m.tr[l].ffn_up + sd.ff0 * nb,g_m.n_embed,sd.ffn);
    matmul(g_m.hb2,xq,g_m.tr[l].ffn_gate + sd.ff0 * nb,g_m.n_embed,sd.ffn);
    for (int i = 0; i < sd.ffn; i++) {
        const float g = g_m.hb2[i];
        g_m.hb[i] = (g / (1.0f + expf(-g))) * g_m.hb[i];
    }

    quantize_q80(g_m.net_xq,g_m.hb);
    matmul_partial_cols(out,g_m.net_xq.data(),g_m.tr[l].ffn_down,g_m.n_ff,g_m.n_embed,sd.ff0,sd.ff0+sd.ffn);
}

bool inline write_full(int fd, const void* buf, size_t len)
{
    const uint8_t* p = (const uint8_t*)buf;
    while (len) {
        ssize_t n = send(fd,p,len,0);
        if (n < 0) {
            if (errno == EINTR) continue;
            ERR("send(fd=%d,len=%lu) failed: %s",fd,(unsigned long)len,strerror(errno));
            return false;
        }
        if (!n) return false;
        p += n;
        len -= n;
    }
    return true;
}

bool inline read_full(int fd, void* buf, size_t len)
{
    uint8_t* p = (uint8_t*)buf;
    while (len) {
        ssize_t n = recv(fd,p,len,0);
        if (n < 0) {
            if (errno == EINTR) continue;
            ERR("recv(fd=%d,len=%lu) failed: %s",fd,(unsigned long)len,strerror(errno));
            return false;
        }
        if (!n) return false;
        p += n;
        len -= n;
    }
    return true;
}

bool set_nodelay(int fd)
{
    int one = 1;
    if (setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one))) {
        ERR("setsockopt(TCP_NODELAY) failed: %s",strerror(errno));
        return false;
    }
    return true;
}

int connect_host(const char* host, int port)
{
    struct addrinfo hints, *res = NULL, *rp = NULL;
    memset(&hints,0,sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    char portbuf[32];
    snprintf(portbuf,sizeof(portbuf),"%d",port);
    if (getaddrinfo(host,portbuf,&hints,&res)) {
        ERR("getaddrinfo failed for %s:%d",host,port);
        return -1;
    }

    int fd = -1;
    for (rp = res; rp; rp = rp->ai_next) {
        fd = socket(rp->ai_family,rp->ai_socktype,rp->ai_protocol);
        if (fd < 0) continue;
        if (!set_nodelay(fd)) {
            close(fd);
            fd = -1;
            continue;
        }
        if (!connect(fd,rp->ai_addr,rp->ai_addrlen)) break;
        close(fd);
        fd = -1;
    }

    freeaddrinfo(res);
    return fd;
}

int listen_port(int port)
{
    int fd = socket(AF_INET,SOCK_STREAM,0);
    if (fd < 0) {
        ERR("socket() failed: %s",strerror(errno));
        return -1;
    }

    int one = 1;
    setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));

    struct sockaddr_in sin;
    memset(&sin,0,sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_addr.s_addr = htonl(INADDR_ANY);
    sin.sin_port = htons(port);

    if (bind(fd,(struct sockaddr*)&sin,sizeof(sin))) {
        ERR("bind(%d) failed: %s",port,strerror(errno));
        close(fd);
        return -1;
    }

    if (listen(fd,1)) {
        ERR("listen() failed: %s",strerror(errno));
        close(fd);
        return -1;
    }

    return fd;
}

bool inline send_hdr(int fd, uint32_t kind, int a, int b, int c)
{
    net_hdr h;
    h.kind = kind;
    h.a = a;
    h.b = b;
    h.c = c;
    return write_full(fd,&h,sizeof(h));
}

bool inline recv_hdr(int fd, net_hdr &h)
{
    return read_full(fd,&h,sizeof(h));
}

bool inline send_partial(int fd, const ftensor &partial)
{
    quantize_q80(g_m.partial_q,partial);
    return write_full(fd,g_m.partial_q.data(),(g_m.n_embed / QK8_0) * sizeof(block_q8_0));
}

bool inline recv_partial(int fd, ftensor &partial)
{
    if (!read_full(fd,g_m.recv_partial_q.data(),(g_m.n_embed / QK8_0) * sizeof(block_q8_0))) return false;
    dequant_q80(partial,g_m.recv_partial_q.data(),0,g_m.n_embed);
    return true;
}

bool connect_workers()
{
    for (int i = 0; i < (int)g_m.peers.size(); i++) {
        g_m.peers[i].fd = connect_host(g_m.peers[i].host.c_str(),g_m.peers[i].port);
        if (g_m.peers[i].fd < 0) {
            ERR("Can't connect to worker %s:%d",g_m.peers[i].host.c_str(),g_m.peers[i].port);
            return false;
        }
        DBG("master connected rank=%d fd=%d",g_m.peers[i].rank,g_m.peers[i].fd);
        if (!send_hdr(g_m.peers[i].fd,NET_HELLO,g_m.peers[i].rank,g_m.nranks,0)) return false;
        net_hdr h;
        if (!recv_hdr(g_m.peers[i].fd,h)) return false;
        if (h.kind != NET_HELLO || h.a != g_m.peers[i].rank || h.b != g_m.nranks) {
            ERR("Worker handshake mismatch on rank %d",g_m.peers[i].rank);
            return false;
        }
    }
    return true;
}

void close_workers()
{
    for (int i = 0; i < (int)g_m.peers.size(); i++) {
        if (g_m.peers[i].fd < 0) continue;
        send_hdr(g_m.peers[i].fd,NET_STOP,0,0,0);
        close(g_m.peers[i].fd);
    }
}

bool wait_for_master()
{
    int lfd = listen_port(g_m.listen_port);
    if (lfd < 0) return false;
    DBG("worker rank=%d listening on %d",g_m.rank,g_m.listen_port);
    g_m.worker_fd = accept(lfd,NULL,NULL);
    close(lfd);
    if (g_m.worker_fd < 0) {
        ERR("accept() failed: %s",strerror(errno));
        return false;
    }
    DBG("worker accepted fd=%d",g_m.worker_fd);
    if (!set_nodelay(g_m.worker_fd)) return false;

    net_hdr h;
    if (!recv_hdr(g_m.worker_fd,h)) return false;
    if (h.kind != NET_HELLO || h.a != g_m.rank || h.b != g_m.nranks) {
        ERR("Bad HELLO for worker rank %d",g_m.rank);
        return false;
    }
    return send_hdr(g_m.worker_fd,NET_HELLO,g_m.rank,g_m.nranks,0);
}

bool worker_loop()
{
    while (1) {
        net_hdr h;
        if (!recv_hdr(g_m.worker_fd,h)) return false;
        if (h.kind == NET_STOP) break;
        if (h.kind != NET_ATT && h.kind != NET_FFN) {
            ERR("Unknown worker message kind %u",h.kind);
            return false;
        }

        if (!read_full(g_m.worker_fd,g_m.net_xq.data(),(g_m.n_embed / QK8_0) * sizeof(block_q8_0))) return false;
        if (h.kind == NET_ATT) att_partial(h.a,h.b,g_m.net_xq.data(),g_m.partial);
        else ffn_partial(h.a,g_m.net_xq.data(),g_m.partial);
        if (!send_partial(g_m.worker_fd,g_m.partial)) return false;
    }
    return true;
}

void puttok(int tok)
{
    if (tok < 0 || tok >= g_m.vocab_size) printf(" <Error token %d> ",tok);
    else printf("%s",g_m.tokens[tok].c_str());
    fflush(stdout);
}

vector<int> tokenize(const char* str, bool bos, bool eos)
{
    vector<int> out;
    out.reserve(strlen(str)+2);
    if (bos) out.push_back(g_m.tok_bos);

    while (*str) {
        string s;
        s += *str;
        if (g_m.tokens_rev.count(s)) out.push_back(g_m.tokens_rev.at(s));
        else {
            s = "<0x00>";
            snprintf(s.data(),s.length()+1,"<0x%02X>",*str);
            out.push_back(g_m.tokens_rev.at(s));
        }
        str++;
    }

    while (1) {
        float best_score = -1e10f;
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
        out[best_idx] = best_id;
        out.erase(out.begin()+best_idx+1);
    }

    if (eos) out.push_back(g_m.tok_eos);
    return out;
}

bool distributed_reduce(uint32_t kind, int layer, int pos, const ftensor &normed)
{
    quantize_q80(g_m.xq,normed);

    for (int i = 0; i < (int)g_m.peers.size(); i++) {
        DBG("master send kind=%u layer=%d pos=%d fd=%d",kind,layer,pos,g_m.peers[i].fd);
        if (!send_hdr(g_m.peers[i].fd,kind,layer,pos,0)) return false;
        if (!write_full(g_m.peers[i].fd,g_m.xq.data(),(g_m.n_embed / QK8_0) * sizeof(block_q8_0))) return false;
    }

    if (kind == NET_ATT)
        att_partial(layer,pos,g_m.xq.data(),g_m.partial);
    else
        ffn_partial(layer,g_m.xq.data(),g_m.partial);

    for (int i = 0; i < (int)g_m.peers.size(); i++) {
        DBG("master recv kind=%u layer=%d pos=%d fd=%d",kind,layer,pos,g_m.peers[i].fd);
        //TODO: for 2+ nodes, don't wait sequentially, allow for random returns
        if (!recv_partial(g_m.peers[i].fd,g_m.recv_partial)) return false;
        for (int i = 0; i < g_m.n_embed; i++) g_m.partial[i] += g_m.recv_partial[i];
    }
    return true;
}

bool inference_step(int tok, int pos, ftensor &logs)
{
    dequant_q80(g_m.x,g_m.t_embed,tok,g_m.n_embed);

    for (int l = 0; l < g_m.n_layers; l++) {
        rmsnorm(g_m.xb,g_m.x,g_m.tr[l].att_norm,g_m.n_embed);
        if (!distributed_reduce(NET_ATT,l,pos,g_m.xb)) return false;
        for (int j = 0; j < g_m.n_embed; j++) g_m.x[j] += g_m.partial[j];

        rmsnorm(g_m.xb,g_m.x,g_m.tr[l].ffn_norm,g_m.n_embed);
        if (!distributed_reduce(NET_FFN,l,pos,g_m.xb)) return false;
        for (int j = 0; j < g_m.n_embed; j++) g_m.x[j] += g_m.partial[j];
    }

    rmsnorm(g_m.x,g_m.x,g_m.t_outnorm,g_m.n_embed);
    quantize_q80(g_m.xq,g_m.x);
    matmul(logs,g_m.xq.data(),g_m.t_out,g_m.n_embed,g_m.vocab_size);
    return true;
}

run_stats generate_run(const vector<int> &prompt, int ngen)
{
    run_stats rs;
    rs.wall = 0.0;
    rs.gen_wall = 0.0;
    rs.tok_s = 0.0;
    rs.per_token = 0.0;
    rs.ngen = 0;

    fill(g_m.kc.begin(),g_m.kc.end(),0.0f);
    fill(g_m.vc.begin(),g_m.vc.end(),0.0f);

    ftensor logits;
    int total = prompt.size() + ngen;
    int tok = g_m.tok_eos;
    double t0 = now_sec();
    double g0 = t0;

    for (int i = 0; i < total; i++) {
        if (i < (int)prompt.size()) tok = prompt[i];
        else tok = sampler_greedy(logits);

        if (!tok || tok == g_m.tok_eos) break;

        rs.out.push_back(tok);
        puttok(tok);

        if (i >= (int)prompt.size()) rs.ngen++;
        if (i == total - 1) break;

        if (i == (int)prompt.size() - 1) g0 = now_sec();

        if (!inference_step(tok,i,logits)) {
            ERR("Inference failed at pos=%d",i);
            break;
        }
    }
    double t1 = now_sec();

    rs.wall = t1 - t0;
    rs.gen_wall = rs.ngen > 0 ? t1 - g0 : 0.0;
    if (rs.ngen > 0 && rs.gen_wall > 0.0) {
        rs.tok_s = (double)rs.ngen / rs.gen_wall;
        rs.per_token = rs.gen_wall / (double)rs.ngen;
    }
    return rs;
}

void usage(const char* progname)
{
    printf("\nUsage: %s [options] <prompt>\n",progname);
    puts("Available options:");
    puts("\t-m <model.gguf>");
    puts("\t-W <rank>/<nranks> - worker mode");
    puts("\t-M <port> - listen port in master mode");
    puts("\t-n <number_of_tokens_to_generate>");
    puts("\t-s <seed>");
    puts("\t-w <host>:<port> - add a worker address");
}

void print_run(const char* label, const run_stats &rs)
{
    printf("%s total wall: %.3fs\n",label,rs.wall);
    printf("%s generation wall: %.3fs\n",label,rs.gen_wall);
    printf("%s generated tokens: %d\n",label,rs.ngen);
    printf("%s tok/s: %.3f\n",label,rs.tok_s);
    printf("%s sec/token: %.6f\n",label,rs.per_token);
}

int parse_args(int argc, char* argv[], int &ngen)
{
    int opt = 0;
    while ((opt = getopt(argc,argv,"m:W:M:n:E:s:w:")) != -1) {
        switch (opt) {
        case 'm':
            assert(open_mmap(optarg));
            assert(read_gguf());
            assert(read_tokenizer());
            break;

        case 'W':
            if (!strchr(optarg,'/')) return -1;
            g_m.worker_mode = true;
            g_m.rank = atoi(optarg);
            g_m.nranks = atoi(strchr(optarg,'/') + 1);
            break;

        case 'M':
            g_m.master_mode = true;
            g_m.listen_port = atoi(optarg);
            break;

        case 'n':
            ngen = atoi(optarg);
            break;

        case 's':
            srand(atoi(optarg));
            break;

        case 'w':
            if (strchr(optarg,':')) {
                net_peer p;
                p.fd = -1;
                p.rank = g_m.peers.size() + 1;
                p.host = string(optarg,strchr(optarg,':') - optarg);
                p.port = atoi(strchr(optarg,':') + 1);
                if (p.host.empty() || p.port <= 0) return -1;
                g_m.peers.push_back(p);
                g_m.master_mode = true;
                break;
            }
            return -1;

        default:
            return -1;
        }
    }
    return optind;
}

int main(int argc, char* argv[])
{
    for (int i = 0; i < 65536; i++) fp1632_lut[i] = fp16_to_fp32((uint16_t)i);
    puts("Welcome to LiGGUF");

    int ngen = 64;
    int argi = parse_args(argc,argv,ngen);
    if (argi < 0 || !g_m.base) {
        usage(argv[0]);
        return 1;
    }

    if (g_m.worker_mode) {
        assert(g_m.nranks <= g_m.n_kv_heads);
        build_shards(g_m.nranks);
        alloc_runtime();
        assert(wait_for_master());
        assert(worker_loop());
        close(g_m.worker_fd);
        close_mmap();
        return 0;
    }

    vector<int> prompt = tokenize(argv[argi],true,false);

    run_stats rs;
    if (g_m.master_mode) {
        g_m.rank = 0;
        g_m.nranks = g_m.peers.size() + 1;
        assert(g_m.nranks <= g_m.n_kv_heads);
        build_shards(g_m.nranks);
        alloc_runtime();
        assert(connect_workers());
        rs = generate_run(prompt,ngen);
        puts("");
        print_run("distributed",rs);
        close_workers();

    } else {
        g_m.rank = 0;
        g_m.nranks = 1;
        build_shards(g_m.nranks);
        alloc_runtime();
        rs = generate_run(prompt,ngen);
        puts("");
        print_run("single-node",rs);
    }

    close_mmap();
    puts("Done.");
    return 0;
}
