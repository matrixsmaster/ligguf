/* LiGGUF - a small, dependency-free LLaMA inference engine with direct GGUF support
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025-2026
 * */

#include <assert.h>
#include <fcntl.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "common.h"
#include "lil_gguf.h"

// Standard GGUF keys
#define TOKENS_KEY "tokenizer.ggml.tokens"
#define TOKENS_SCORE_KEY "tokenizer.ggml.scores"
#define TOKENS_TYPE_KEY "tokenizer.ggml.token_type"
#define TOKENS_MERGES_KEY "tokenizer.ggml.merges"
#define TOKENS_MODEL_KEY "tokenizer.ggml.model"
#define TOKENS_PRE_KEY "tokenizer.ggml.pre"
#define TOKENS_EMBED_KEY "token_embd.weight"
#define TOKENS_BOS_KEY "tokenizer.ggml.bos_token_id"
#define TOKENS_EOS_KEY "tokenizer.ggml.eos_token_id"
#define TOKENS_UNK_KEY "tokenizer.ggml.unknown_token_id"
#define TOKENS_SEP_KEY "tokenizer.ggml.seperator_token_id"
#define TOKENS_PAD_KEY "tokenizer.ggml.padding_token_id"
#define TOKENS_CLS_KEY "tokenizer.ggml.cls_token_id"
#define TOKENS_MASK_KEY "tokenizer.ggml.mask_token_id"
#define TOKENS_ADD_BOS_KEY "tokenizer.ggml.add_bos_token"
#define ARCH_KEY "general.architecture"
#define OUTPUT_KEY "output.weight"
#define OUTPUT_NORM_KEY "output_norm.weight"

// LLaMA-specific keys
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

// Qwen3-specific keys
#define QWEN3_BLOCK_COUNT_KEY "qwen3.block_count"
#define QWEN3_CONTEXT_LEN_KEY "qwen3.context_length"
#define QWEN3_EMBED_LEN_KEY "qwen3.embedding_length"
#define QWEN3_HEAD_COUNT_KEY "qwen3.attention.head_count"
#define QWEN3_HEAD_KV_COUNT_KEY "qwen3.attention.head_count_kv"
#define QWEN3_RMS_EPSILON_KEY "qwen3.attention.layer_norm_rms_epsilon"
#define QWEN3_ROPE_BASE_KEY "qwen3.rope.freq_base"
#define QWEN3_ROPE_SCALING_TYPE_KEY "qwen3.rope.scaling.type"
#define QWEN3_ROPE_SCALING_FACTOR_KEY "qwen3.rope.scaling.factor"
#define QWEN3_ROPE_SCALING_ORIG_CTX_KEY "qwen3.rope.scaling.original_context_length"
#define QWEN3_KEY_LEN_KEY "qwen3.attention.key_length"
#define QWEN3_VALUE_LEN_KEY "qwen3.attention.value_length"
#define QWEN3_FF_LEN_KEY "qwen3.feed_forward_length"

//BERT-specific keys
#define BERT_BLOCK_COUNT_KEY "bert.block_count"
#define BERT_CONTEXT_LEN_KEY "bert.context_length"
#define BERT_EMBED_LEN_KEY "bert.embedding_length"
#define BERT_HEAD_COUNT_KEY "bert.attention.head_count"
#define BERT_LN_EPSILON_KEY "bert.attention.layer_norm_epsilon"
#define BERT_FF_LEN_KEY "bert.feed_forward_length"
#define BERT_POS_EMBED_KEY "position_embd.weight"
#define BERT_TYPE_EMBED_KEY "token_types.weight"
#define BERT_EMB_NORM_W_KEY "token_embd_norm.weight"
#define BERT_EMB_NORM_B_KEY "token_embd_norm.bias"

using namespace std;

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

static string rdstr(uint8_t** pos)
{
    uint64_t l = rd64(pos);
    string res((const char*)*pos,l);
    *pos += l;
    return res;
}

uint32_t gguf_model::kvrd32(const string& key) const
{
    if (!meta_kv.count(key)) {
        ERR("Key '%s' not found!",key.c_str());
        return 0;
    }
    uint8_t* p = base + meta_kv.at(key).off;
    return rd32(&p);
}

float gguf_model::kvrdf32(const string& key) const
{
    if (!meta_kv.count(key)) {
        ERR("Key '%s' not found!",key.c_str());
        return 0;
    }
    uint8_t* p = base + meta_kv.at(key).off;
    return rdf32(&p);
}

string gguf_model::kvrdstr(const string& key) const
{
    if (!meta_kv.count(key)) {
        ERR("Key '%s' not found!",key.c_str());
        return "";
    }
    uint8_t* p = base + meta_kv.at(key).off;
    return rdstr(&p);
}

bool gguf_model::kvrdbool(const string& key) const
{
    if (!meta_kv.count(key)) {
        ERR("Key '%s' not found!",key.c_str());
        return false;
    }
    uint8_t* p = base + meta_kv.at(key).off;
    return rd8(&p) != 0;
}

static uint64_t skipper(gguf_val_type t, uint8_t* pos)
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
            abort();
    }
}

static inline bool supported_qweight(gguf_type t)
{
    return t == Q8_0 || t == Q1_0 || t == Q1_G || t == Q2_K || t == Q3_K || t == Q4_K || t == Q5_K || t == Q6_K;
}

static inline bool supported_rowweight(gguf_type t)
{
    return t == F32 || t == F16 || supported_qweight(t);
}

static wtensor mkweight(uint8_t* ptr, const gguf_tensor& tz)
{
    wtensor r;
    r.ptr = ptr;
    r.type = tz.type;
    r.rsz = row_size(tz.type,tz.dims[0]);
    return r;
}

uint64_t row_size(gguf_type type, int len)
{
    switch (type) {
        case F32:
            return sizeof(float) * len;
        case F16:
            return sizeof(gguf_half) * len;
        case Q8_0:
            assert(len % QK8_0 == 0);
            return sizeof(block_q8_0) * (len / QK8_0);
        case Q1_0:
            assert(len % QK8_0 == 0);
            return sizeof(block_q1_0) * (len / QK8_0);
        case Q1_G:
            assert(len % 128 == 0);
            return sizeof(block_q1_G) * (len / 128);
        case Q2_K:
            assert(len % QK_K == 0);
            return sizeof(block_q2_K) * (len / QK_K);
        case Q3_K:
            assert(len % QK_K == 0);
            return sizeof(block_q3_K) * (len / QK_K);
        case Q4_K:
            assert(len % QK_K == 0);
            return sizeof(block_q4_K) * (len / QK_K);
        case Q5_K:
            assert(len % QK_K == 0);
            return sizeof(block_q5_K) * (len / QK_K);
        case Q6_K:
            assert(len % QK_K == 0);
            return sizeof(block_q6_K) * (len / QK_K);
        default:
            ERR("Unsupported row type %d",type);
            abort();
    }
}

bool gguf_model::open_mmap(const char* fn)
{
    fsize = 0;
    base = NULL;
    file = open(fn,O_RDONLY);
    if (file == -1) {
        ERR("Can't open file %s",fn);
        return false;
    }

    struct stat st;
    if (fstat(file,&st)) {
        ERR("Can't stat file %s",fn);
        close_mmap();
        return false;
    }

    fsize = st.st_size;
    base = (uint8_t*)mmap(NULL,fsize,PROT_READ,MAP_SHARED,file,0);
    if (base == MAP_FAILED) {
        ERR("Can't mmap() file %s",fn);
        close_mmap();
        return false;
    }

    return true;
}

void gguf_model::close_mmap()
{
    if (base && base != MAP_FAILED && fsize) munmap(base, fsize);
    if (file != -1) close(file);
    base = NULL;
    fsize = 0;
    file = -1;
}

bool gguf_model::read_tokenizer()
{
    if (!meta_kv.count(TOKENS_KEY)) {
        ERR("Tokens array %s not found!",TOKENS_KEY);
        return false;
    }

    uint8_t* pos = base + meta_kv[TOKENS_KEY].off;
    gguf_val_type nt = (gguf_val_type)rd32(&pos);
    if (nt != GSTRING) {
        ERR("Unexpected tokens array type %d",nt);
        return false;
    }

    uint64_t ne = rd64(&pos);
    if (!vocab_size) vocab_size = ne;
    if ((int)ne != vocab_size) {
        ERR("Unexpected number of tokens %lu",ne);
        return false;
    }
    tokens.resize(ne);

    for (uint64_t i = 0; i < ne; i++) {
        tokens[i] = rdstr(&pos);
        tokens_rev[tokens[i]] = i;
    }

    tok_model = meta_kv.count(TOKENS_MODEL_KEY)? kvrdstr(TOKENS_MODEL_KEY) : "";
    tok_pre = meta_kv.count(TOKENS_PRE_KEY)? kvrdstr(TOKENS_PRE_KEY) : "";
    add_bos = meta_kv.count(TOKENS_ADD_BOS_KEY)? kvrdbool(TOKENS_ADD_BOS_KEY) : true;

    if (meta_kv.count(TOKENS_SCORE_KEY)) {
        pos = base + meta_kv[TOKENS_SCORE_KEY].off;
        nt = (gguf_val_type)rd32(&pos);
        if (nt != GFLOAT32) {
            ERR("Unexpected token scores type %d",nt);
            return false;
        }

        ne = rd64(&pos);
        tokscores.resize(ne);
        for (uint64_t i = 0; i < ne; i++) tokscores[i] = rdf32(&pos);
    }

    if (meta_kv.count(TOKENS_MERGES_KEY)) {
        pos = base + meta_kv[TOKENS_MERGES_KEY].off;
        nt = (gguf_val_type)rd32(&pos);
        if (nt != GSTRING) {
            ERR("Unexpected merge array type %d",nt);
            return false;
        }

        ne = rd64(&pos);
        merges.resize(ne);
        merge_rank.clear();
        for (uint64_t i = 0; i < ne; i++) {
            merges[i] = rdstr(&pos);
            merge_rank[merges[i]] = i;
        }
    }

    return true;
}

bool gguf_model::read_gguf()
{
    uint8_t* p = base;

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
        kv.off = p - base;
        meta_kv[key] = kv;
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
        tensors[key] = tz;
    }

    uint64_t off = p - base;
    tensors_off = base + (off + (ALIGNMENT - (off % ALIGNMENT)) % ALIGNMENT);

    arch = kvrdstr(ARCH_KEY);
    tok_bos = meta_kv.count(TOKENS_BOS_KEY)? kvrd32(TOKENS_BOS_KEY) : 0;
    tok_eos = meta_kv.count(TOKENS_EOS_KEY)? kvrd32(TOKENS_EOS_KEY) : 0;
    tok_unk = meta_kv.count(TOKENS_UNK_KEY)? kvrd32(TOKENS_UNK_KEY) : 0;
    tok_sep = meta_kv.count(TOKENS_SEP_KEY)? kvrd32(TOKENS_SEP_KEY) : 0;
    tok_pad = meta_kv.count(TOKENS_PAD_KEY)? kvrd32(TOKENS_PAD_KEY) : 0;
    tok_cls = meta_kv.count(TOKENS_CLS_KEY)? kvrd32(TOKENS_CLS_KEY) : 0;
    tok_mask = meta_kv.count(TOKENS_MASK_KEY)? kvrd32(TOKENS_MASK_KEY) : 0;

    if (arch == "llama") return read_llama();
    else if (arch == "qwen3") return read_qwen3();
    else if (arch == "bert") return read_bert();

    ERR("Unsupported architecture '%s'",arch.c_str());
    return false;
}

bool gguf_model::read_llama()
{
    vocab_size = kvrd32(VOCAB_SIZE_KEY);
    n_layers = kvrd32(BLOCK_COUNT_KEY);
    n_embed = kvrd32(EMBED_LEN_KEY);
    n_heads = kvrd32(HEAD_COUNT_KEY);
    n_kv_heads = kvrd32(HEAD_KV_COUNT_KEY);
    n_context = kvrd32(CONTEXT_LEN_KEY);
    rms_epsilon = kvrdf32(RMS_EPSILON_KEY);
    rope_base = kvrdf32(ROPE_BASE_KEY);
    rope_dim = kvrd32(ROPE_DIMS_KEY);
    n_ff = kvrd32(FF_LEN_KEY);

    head_dim = n_embed / n_heads;
    assert(head_dim * n_heads == n_embed);
    kv_dim = (n_embed * n_kv_heads) / n_heads;

    tr.resize(n_layers);
    memset(tr.data(),0,n_layers * sizeof(trans_block));

    for (map<string,gguf_tensor>::const_iterator it = tensors.begin(); it != tensors.end(); ++it) {
        const string& key = it->first;
        uint8_t* ptr = tensors_off + it->second.off;
        if (!key.compare(0,4,"blk.") && key.find(".weight") != string::npos) {
            int id = atoi(key.c_str()+4);
            assert(id >= 0 && id < n_layers);

            if (key.find("attn_norm") != string::npos) tr[id].att_norm = (float*)ptr;
            else if (key.find("attn_q") != string::npos) tr[id].att_q = mkweight(ptr,it->second);
            else if (key.find("attn_k") != string::npos) tr[id].att_k = mkweight(ptr,it->second);
            else if (key.find("attn_v") != string::npos) tr[id].att_v = mkweight(ptr,it->second);
            else if (key.find("attn_output") != string::npos) tr[id].att_out = mkweight(ptr,it->second);
            else if (key.find("ffn_norm") != string::npos) tr[id].ffn_norm = (float*)ptr;
            else if (key.find("ffn_up") != string::npos) tr[id].ffn_up = mkweight(ptr,it->second);
            else if (key.find("ffn_down") != string::npos) tr[id].ffn_down = mkweight(ptr,it->second);
            else if (key.find("ffn_gate") != string::npos) tr[id].ffn_gate = mkweight(ptr,it->second);
        }
        else if (key == TOKENS_EMBED_KEY) t_embed = mkweight(ptr,it->second);
        else if (key == OUTPUT_KEY) t_out = mkweight(ptr,it->second);
        else if (key == OUTPUT_NORM_KEY) t_outnorm = (float*)ptr;
    }

    assert(supported_rowweight(t_embed.type));
    assert(supported_rowweight(t_out.type));
    return true;
}

bool gguf_model::read_qwen3()
{
    n_layers = kvrd32(QWEN3_BLOCK_COUNT_KEY);
    n_embed = kvrd32(QWEN3_EMBED_LEN_KEY);
    n_heads = kvrd32(QWEN3_HEAD_COUNT_KEY);
    n_kv_heads = kvrd32(QWEN3_HEAD_KV_COUNT_KEY);
    n_context = kvrd32(QWEN3_CONTEXT_LEN_KEY);
    rms_epsilon = kvrdf32(QWEN3_RMS_EPSILON_KEY);
    rope_base = kvrdf32(QWEN3_ROPE_BASE_KEY);
    rope_scaling = meta_kv.count(QWEN3_ROPE_SCALING_TYPE_KEY)? kvrdstr(QWEN3_ROPE_SCALING_TYPE_KEY) : "";
    rope_scale = meta_kv.count(QWEN3_ROPE_SCALING_FACTOR_KEY)? kvrdf32(QWEN3_ROPE_SCALING_FACTOR_KEY) : 1.0f;
    rope_orig_context = meta_kv.count(QWEN3_ROPE_SCALING_ORIG_CTX_KEY)? kvrd32(QWEN3_ROPE_SCALING_ORIG_CTX_KEY) : 0;
    head_dim = kvrd32(QWEN3_KEY_LEN_KEY);
    rope_dim = head_dim;
    n_ff = kvrd32(QWEN3_FF_LEN_KEY);
    int value_dim = kvrd32(QWEN3_VALUE_LEN_KEY);
    kv_dim = n_kv_heads * value_dim;
    vocab_size = 0;

    tr.resize(n_layers);
    memset(tr.data(),0,n_layers * sizeof(trans_block));

    for (map<string,gguf_tensor>::const_iterator it = tensors.begin(); it != tensors.end(); ++it) {
        const string& key = it->first;
        uint8_t* ptr = tensors_off + it->second.off;
        if (!key.compare(0,4,"blk.") && key.find(".weight") != string::npos) {
            int id = atoi(key.c_str()+4);
            assert(id >= 0 && id < n_layers);

            if (key.find("attn_norm") != string::npos) tr[id].att_norm = (float*)ptr;
            else if (key.find("attn_q_norm") != string::npos) tr[id].att_q_norm = (float*)ptr;
            else if (key.find("attn_k_norm") != string::npos) tr[id].att_k_norm = (float*)ptr;
            else if (key.find("attn_q") != string::npos) tr[id].att_q = mkweight(ptr,it->second);
            else if (key.find("attn_k") != string::npos) tr[id].att_k = mkweight(ptr,it->second);
            else if (key.find("attn_v") != string::npos) tr[id].att_v = mkweight(ptr,it->second);
            else if (key.find("attn_output") != string::npos) tr[id].att_out = mkweight(ptr,it->second);
            else if (key.find("ffn_norm") != string::npos) tr[id].ffn_norm = (float*)ptr;
            else if (key.find("ffn_up") != string::npos) tr[id].ffn_up = mkweight(ptr,it->second);
            else if (key.find("ffn_down") != string::npos) tr[id].ffn_down = mkweight(ptr,it->second);
            else if (key.find("ffn_gate") != string::npos) tr[id].ffn_gate = mkweight(ptr,it->second);
        }
        else if (key == TOKENS_EMBED_KEY) {
            t_embed = mkweight(ptr,it->second);
            if (!vocab_size && it->second.dims.size() > 1) vocab_size = it->second.dims[1];
        }
        else if (key == OUTPUT_KEY) {
            t_out = mkweight(ptr,it->second);
            if (!vocab_size && it->second.dims.size() > 1) vocab_size = it->second.dims[1];
        }
        else if (key == OUTPUT_NORM_KEY) t_outnorm = (float*)ptr;
    }

    assert(supported_rowweight(t_embed.type));
    assert(supported_rowweight(t_out.type));
    return true;
}

bool gguf_model::read_bert()
{
    n_layers = kvrd32(BERT_BLOCK_COUNT_KEY);
    n_embed = kvrd32(BERT_EMBED_LEN_KEY);
    n_heads = kvrd32(BERT_HEAD_COUNT_KEY);
    n_kv_heads = n_heads;
    n_context = kvrd32(BERT_CONTEXT_LEN_KEY);
    rms_epsilon = kvrdf32(BERT_LN_EPSILON_KEY);
    rope_base = 0.0f;
    rope_dim = 0;
    n_ff = kvrd32(BERT_FF_LEN_KEY);
    head_dim = n_embed / n_heads;
    kv_dim = n_embed;

    bert.resize(n_layers);
    memset(bert.data(),0,bert.size() * sizeof(bert_block));

    for (map<string,gguf_tensor>::const_iterator it = tensors.begin(); it != tensors.end(); ++it) {
        const string& key = it->first;
        uint8_t* ptr = tensors_off + it->second.off;

        if (!key.compare(0,4,"blk.")) {
            if (key.find(".weight") == string::npos && key.find(".bias") == string::npos) continue;
            int id = atoi(key.c_str()+4);
            assert(id >= 0 && id < n_layers);

            bert_block &b = bert[id];
            if (key.find("attn_q.w") != string::npos) b.attn_q = mkweight(ptr,it->second);
            else if (key.find("attn_q.b") != string::npos) b.attn_q_bias = (float*)ptr;
            else if (key.find("attn_k.w") != string::npos) b.attn_k = mkweight(ptr,it->second);
            else if (key.find("attn_k.b") != string::npos) b.attn_k_bias = (float*)ptr;
            else if (key.find("attn_v.w") != string::npos) b.attn_v = mkweight(ptr,it->second);
            else if (key.find("attn_v.b") != string::npos) b.attn_v_bias = (float*)ptr;
            else if (key.find("attn_output.w") != string::npos) b.attn_out = mkweight(ptr,it->second);
            else if (key.find("attn_output.b") != string::npos) b.attn_out_bias = (float*)ptr;
            else if (key.find("attn_output_norm.w") != string::npos) b.attn_norm_w = (float*)ptr;
            else if (key.find("attn_output_norm.b") != string::npos) b.attn_norm_b = (float*)ptr;
            else if (key.find("ffn_up.w") != string::npos) b.ffn_up = mkweight(ptr,it->second);
            else if (key.find("ffn_up.b") != string::npos) b.ffn_up_bias = (float*)ptr;
            else if (key.find("ffn_down.w") != string::npos) b.ffn_down = mkweight(ptr,it->second);
            else if (key.find("ffn_down.b") != string::npos) b.ffn_down_bias = (float*)ptr;
            else if (key.find("layer_output_norm.w") != string::npos) b.ffn_norm_w = (float*)ptr;
            else if (key.find("layer_output_norm.b") != string::npos) b.ffn_norm_b = (float*)ptr;
        }
        else if (key == TOKENS_EMBED_KEY) bert_tok_embd = mkweight(ptr,it->second);
        else if (key == BERT_POS_EMBED_KEY) bert_pos_embd = mkweight(ptr,it->second);
        else if (key == BERT_TYPE_EMBED_KEY) bert_tok_types = mkweight(ptr,it->second);
        else if (key == BERT_EMB_NORM_W_KEY) bert_emb_norm_w = (float*)ptr;
        else if (key == BERT_EMB_NORM_B_KEY) bert_emb_norm_b = (float*)ptr;
    }

    assert(supported_rowweight(bert_tok_embd.type));
    assert(supported_rowweight(bert_pos_embd.type));
    assert(supported_rowweight(bert_tok_types.type));
    return true;
}
