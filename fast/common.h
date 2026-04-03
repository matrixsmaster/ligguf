#pragma once

#include <string>
#include "lil_gguf.h"

#define VERSION "0.5.2"
#define CAPABILITIES "AVX2 NEON Q8 Q1 QnK"
#define DEFAULT_TOPP 0.9f
#define DEFAULT_TEMP 0.5f
#define MAXLINE 4096
#define MULTITHREAD _Pragma("omp parallel for")

// Helpful debug/error macros
#define ERR(S,...) fprintf(stderr,S "\n", __VA_ARGS__)
#ifdef DEBUG
#define DBG(S,...) printf(S "\n", __VA_ARGS__)
#else
#define DBG(...)
#endif

struct sampler_entry {
    float p;
    int tok;
};

struct model_file {
    int pos;
    int n_context;
    uint64_t kv_size;
    uint64_t logits_size;
} __attribute((packed));

struct model_state {
    std::string model_fn; // model file name
    gguf_model m; // model current state

    int u_context = 0; // user-supplied context length
    int n_gen = 0; // number of tokens to generate
    int pos = 0; // current position
    int topk = 0; // TopK sampling
    float topp = DEFAULT_TOPP; // TopP sampling
    float temp = DEFAULT_TEMP; // sampling temperature
    bool greedy = false; // greedy sampling flag
    bool chat = false; // chat mode flag
    std::string chat_file, chat_log; // previous chats log filename and full log
    std::string prompt; // model prompt
    std::string ainame, usrname; // names used in chat mode
    std::string eot; // end-of-turn marker string
    std::string load_state, save_state; // files to load the model state from / save into
    std::string bert_model_fn; // BERT embedding model for VDB chat memory

    ftensor x; // current activation
    qtensor xq8; // current activation quantized to Q8_0 for Q8_0 weights
    ktensor xq; // current activation quantized to Q8_K for K-quants
    ftensor xb; // residual branch buffer / attention result
    ftensor xb2; // projected attention output
    ftensor hb; // FFN up output / SwiGLU output
    ftensor hb2; // FFN gate output
    ftensor q, k, v; // current QKV vectors
    htensor kc, vc; // KV cache
    ftensor att; // attention scores/weights
    ftensor rope_freq; // precomputed inverse RoPE frequencies
    ftensor logits; // output logits for the next token
    std::vector<sampler_entry> samp; // TopP work buffer

    int ntimed = 0; // number of times we've timed the inference
    double gen_ms = 0; // total time length of all inference steps
    int nsampled = 0; // number of times we've timed the sampler
    double samp_ms = 0; // total time length of all sampling steps

    void allocate();
};

extern model_state g_m;

double tmsec();
std::string load_text_file(const char* fn);
bool save_text_file(const char* fn, const std::string &text);
