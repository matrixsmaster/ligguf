/* LiGGUF - a small, dependency-free LLaMA inference engine with direct GGUF support
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025-2026
 * */

#include <time.h>
#include <string>
#include "common.h"

using namespace std;

void model_state::allocate()
{
    if (u_context) {
        if (u_context > m.n_context)
            ERR("This model doesn't support context larger than %d",m.n_context);
        else
            m.n_context = u_context;
        DBG("Model context length set to %d",m.n_context);
    }

    x.resize(m.n_embed);
    xb.resize(m.n_embed);
    xb2.resize(m.n_embed);
    xq8.resize(m.n_ff / QK8_0);
    xq.resize(m.n_ff / QK_K);
    hb.resize(m.n_ff);
    hb2.resize(m.n_ff);
    q.resize(m.n_embed);
    k.resize(m.kv_dim);
    v.resize(m.kv_dim);
    kc.resize((size_t)m.n_layers * m.n_context * m.kv_dim,0);
    vc.resize(kc.size(),0);
    att.resize((size_t)m.n_heads * m.n_context);
    rope_freq.resize(m.rope_dim / 2);
    logits.resize(m.vocab_size,0);
    samp.resize(m.vocab_size);

    for (int i = 0; i < (int)rope_freq.size(); i++)
        rope_freq[i] = powf(m.rope_base, -2.0f * (float)i / (float)m.rope_dim);
}

double tmsec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

string load_text_file(const char* fn)
{
    FILE* f = fopen(fn,"r");
    if (!f) {
        ERR("Can't open file %s",fn);
        return "";
    }

    char line[MAXLINE];
    string full;
    while (!feof(f)) {
        if (!fgets(line,sizeof(line),f)) break;
        full += line;
    }

    fclose(f);
    return full;
}

bool save_text_file(const char* fn, const string &text)
{
    FILE* f = fopen(fn,"w");
    if (!f) {
        ERR("Can't open file %s",fn);
        return false;
    }
    bool ok = fwrite(text.data(),text.size(),1,f);
    fclose(f);

    if (!ok) ERR("Can't write file %s",fn);
    return ok;
}
