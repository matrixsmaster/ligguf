/* LiGGUF - a small, dependency-free LLaMA inference engine with direct GGUF support
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025-2026
 * */

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <map>

// Base constants
#define ALIGNMENT 32
#define MAXTENSORS 4000
#define MAXMETAKV 500
#define MAXNAMELEN 1024
#define QK8_0 32
#define QK_K 256
#define K_SCALE_SIZE 12

enum gguf_type {
    F32 = 0, F16 = 1,
    Q4_0 = 2, Q4_1 = 3, Q4_2 = 4, Q4_3 = 5,
    Q5_0 = 6, Q5_1 = 7,
    Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18,
    IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23,
    I8 = 24, I16 = 25, I32 = 26, I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34, TQ2_0 = 35,
    MXFP4 = 39,
    Q1_0 = 40,
    Q1_G = 41, // previously Q1_0_G128
    GGUF_TYPE_COUNT
};

enum gguf_val_type {
    GUINT8, GINT8, GUINT16, GINT16, GUINT32, GINT32,
    GFLOAT32, GBOOL, GSTRING, GARRAY, GUINT64, GINT64, GFLOAT64,
    GGUF_VAL_TYPE_COUNT
};

typedef uint16_t gguf_half;

struct gguf_kv {
    uint64_t off;
    gguf_val_type tag;
};

struct gguf_tensor {
    uint64_t off;
    std::vector<uint64_t> dims;
    gguf_type type;
};

struct block_q8_0 {
    uint16_t d;
    int8_t qs[QK8_0];
};

struct block_q1_0 {
    gguf_half d;
    uint8_t qs[QK8_0/8];
};

struct block_q1_G {
    gguf_half d;
    uint8_t qs[128/8];
};

struct block_q2_K {
    uint8_t scales[QK_K/16];
    uint8_t qs[QK_K/4];
    gguf_half d;
    gguf_half dmin;
};

struct block_q3_K {
    uint8_t hmask[QK_K/8];
    uint8_t qs[QK_K/4];
    uint8_t scales[12];
    gguf_half d;
};

struct block_q4_K {
    gguf_half d;
    gguf_half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
};

struct block_q5_K {
    gguf_half d;
    gguf_half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
};

struct block_q6_K {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t scales[QK_K/16];
    gguf_half d;
};

struct block_q8_K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K/16];
};

typedef std::vector<block_q8_0> qtensor;
typedef std::vector<block_q8_K> ktensor;
typedef std::vector<gguf_half> htensor;
typedef std::vector<float> ftensor;

struct wtensor {
    uint8_t* ptr;
    gguf_type type;
    uint32_t rsz;
};

struct trans_block {
    float* att_norm;
    wtensor att_q;
    wtensor att_k;
    wtensor att_v;
    wtensor att_out;
    float* att_q_norm;
    float* att_k_norm;
    float* ffn_norm;
    wtensor ffn_up;
    wtensor ffn_down;
    wtensor ffn_gate;
};

struct bert_block {
    wtensor attn_q, attn_k, attn_v, attn_out;
    float* attn_q_bias;
    float* attn_k_bias;
    float* attn_v_bias;
    float* attn_out_bias;
    float* attn_norm_w;
    float* attn_norm_b;
    wtensor ffn_up, ffn_down;
    float* ffn_up_bias;
    float* ffn_down_bias;
    float* ffn_norm_w;
    float* ffn_norm_b;
};

struct gguf_model {
    int file = -1; // model file handle
    uint8_t* base = NULL; // base mmap address
    uint64_t fsize = 0; // model file size
    uint8_t* tensors_off = NULL; // aligned offset of the start of tensors block
    std::string arch; // model architecture

    int vocab_size; // size of the vocabulary
    int n_layers; // number of layers
    int n_heads; // number of Query heads
    int n_kv_heads; // number of Key/Value heads
    int n_embed; // input embedding size
    int n_context; // size of the context window
    int tok_bos, tok_eos; // BOS/EOS token IDs
    int tok_unk, tok_sep, tok_pad, tok_cls, tok_mask; // extra tokenizer IDs
    bool add_bos = true; // tokenizer BOS policy
    int head_dim; // head size (computed)
    float rms_epsilon; // epsilon for RMS norm
    float rope_base; // RoPE base frequency
    int rope_dim; // RoPE dimension, clamped to head_dim during inference
    std::string rope_scaling;
    float rope_scale = 1.0f;
    int rope_orig_context = 0;
    int n_ff; // feed-forward hidden size
    int kv_dim; // total K/V width across KV heads

    std::vector<std::string> tokens; // vector of known tokens (pos == index)
    std::map<std::string,int> tokens_rev; // reverse token lookup by string
    ftensor tokscores; // tokenizer scores from GGUF
    std::vector<std::string> merges;
    std::map<std::string,int> merge_rank;
    std::string tok_model;
    std::string tok_pre;
    std::map<std::string,gguf_kv> meta_kv; // model metadata key/value pairs
    std::map<std::string,gguf_tensor> tensors; // all tensors described in the GGUF
    wtensor t_embed; // token embedding tensor (const)
    wtensor t_out; // output classifier tensor (const)
    float* t_outnorm; // final RMS norm weights (const)
    std::vector<trans_block> tr; // transformer blocks/layers (const)
    wtensor bert_tok_embd; // BERT token embedding tensor (const)
    wtensor bert_pos_embd; // BERT position embedding tensor (const)
    wtensor bert_tok_types; // BERT token-type embedding tensor (const)
    float* bert_emb_norm_w = NULL; // BERT embedding layernorm weights (const)
    float* bert_emb_norm_b = NULL; // BERT embedding layernorm bias (const)
    std::vector<bert_block> bert; // BERT encoder blocks/layers (const)

    uint32_t kvrd32(const std::string& key) const;
    float kvrdf32(const std::string& key) const;
    std::string kvrdstr(const std::string& key) const;
    bool kvrdbool(const std::string& key) const;

    bool open_mmap(const char* fn);
    void close_mmap();
    bool read_tokenizer();
    bool read_gguf();
    bool read_llama();
    bool read_qwen3();
    bool read_bert();
};

uint64_t row_size(gguf_type type, int len);
