/* LiGGUF - a small, dependency-free LLaMA inference engine with direct GGUF support
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025-2026
 * */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>
#include "common.h"
#include "lil_gguf.h"
#include "lil_math.h"
#include "tokenize.h"
#include "vdb.h"
#include "viz_sdl.h"

using namespace std;

model_state g_m;

void inference(int tok)
{
    const float att_scale = 1.0f / sqrtf(g_m.m.head_dim);
    const int pos = g_m.pos++;

    viz_begin_frame();

    // Start from the token embedding row, dequantized into the working activation buffer.
    dequantize_row(g_m.m.t_embed.type,g_m.m.t_embed.ptr + tok * g_m.m.t_embed.rsz,g_m.x.data(),g_m.m.n_embed);
    viz_trace_column(0,g_m.x.data(),g_m.m.n_embed);

    for (int l = 0; l < g_m.m.n_layers; l++) {
        // Attention branch: RMSNorm, QKV projections, RoPE on Q/K, cache the new KV, then mix values with attention weights.
        rmsnorm(g_m.xb,g_m.x,g_m.m.tr[l].att_norm,g_m.m.n_embed,g_m.m.rms_epsilon);

        quantize_q8_K(g_m.xq,g_m.xb);
        matmul3(g_m.q,g_m.k,g_m.v,&g_m.xb,&g_m.xq,NULL,g_m.m.tr[l].att_q,g_m.m.tr[l].att_k,g_m.m.tr[l].att_v,g_m.m.n_embed,g_m.m.n_embed,g_m.m.kv_dim,g_m.m.kv_dim); // Q, K & V

        if (g_m.m.arch == "qwen3") {
            // RMSnorm for current Query and Key
            float* p = g_m.q.data();
            for (int h = 0; h < g_m.m.n_heads; h++, p += g_m.m.head_dim)
                rmsnorm_inplace(p,g_m.m.tr[l].att_q_norm,g_m.m.head_dim,g_m.m.rms_epsilon);

            p = g_m.k.data();
            for (int h = 0; h < g_m.m.n_kv_heads; h++, p += g_m.m.head_dim)
                rmsnorm_inplace(p,g_m.m.tr[l].att_k_norm,g_m.m.head_dim,g_m.m.rms_epsilon);
        }

        // RoPE Q & K (float vectors)
        rope(g_m.q,g_m.m.n_heads,pos,g_m.m,g_m.rope_freq);
        rope(g_m.k,g_m.m.n_kv_heads,pos,g_m.m,g_m.rope_freq);

        // Simple KV cache
        size_t loff = (size_t)l * g_m.m.n_context * g_m.m.kv_dim; // kv cache layer offset
        gguf_half* kc_row = g_m.kc.data() + loff + pos * g_m.m.kv_dim;
        gguf_half* vc_row = g_m.vc.data() + loff + pos * g_m.m.kv_dim;
        f32_to_f16_row(g_m.k.data(),kc_row,g_m.m.kv_dim);
        f32_to_f16_row(g_m.v.data(),vc_row,g_m.m.kv_dim);

        const int nrep = g_m.m.n_heads / g_m.m.n_kv_heads; // GQA, heads per KV head

        MULTITHREAD
        for (int h = 0; h < g_m.m.n_heads; h++) {
            float* att = g_m.att.data() + (h * g_m.m.n_context); // attention scores for this head
            float* q = g_m.q.data() + (h * g_m.m.head_dim); // start of the Q vector
            float* k = g_m.k.data() + (h / nrep) * g_m.m.head_dim;
            float* v = g_m.v.data() + (h / nrep) * g_m.m.head_dim;

            // Iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // calculate the attention score as the dot product of q and k
                float score;
                if (t == pos) {
                    score = vec_dot_f32(q,k,g_m.m.head_dim);
                } else {
                    kc_row = g_m.kc.data() + loff + t * g_m.m.kv_dim + (h / nrep) * g_m.m.head_dim;
                    score = vec_dot_f16_f32(kc_row,q,g_m.m.head_dim);
                }

                // save the score to the attention buffer
                att[t] = score * att_scale;
            }

            // Turn attention scores over the visible prefix into normalized weights.
            softmax(att,pos+1);

            // Weighted sum of the values
            float* xb = g_m.xb.data() + (h * g_m.m.head_dim);
            memset(xb,0,g_m.m.head_dim * sizeof(float));

            // Accumulate the weighted sum of cached values for this head.
            for (int t = 0; t <= pos; t++) {
                float a = att[t];
                if (t == pos) {
                    for (int i = 0; i < g_m.m.head_dim; i++) xb[i] += a * v[i];
                } else {
                    vc_row = g_m.vc.data() + loff + t * g_m.m.kv_dim + (h / nrep) * g_m.m.head_dim;
                    for (int i = 0; i < g_m.m.head_dim; i++) xb[i] += a * fp16_to_fp32(vc_row[i]);
                }
            }
        }

        // Project the mixed attention result back to model width and add the residual.
        quantize_q8_K(g_m.xq,g_m.xb);
        matmul(g_m.xb2,&g_m.xb,&g_m.xq,NULL,g_m.m.tr[l].att_out,g_m.m.n_embed,g_m.m.n_embed);
        viz_trace_column(1 + l * 2,g_m.xb2.data(),g_m.m.n_embed);
        for (int j = 0; j < g_m.m.n_embed; j++) g_m.x[j] += g_m.xb2[j];

        // FFN branch: run up/gate together, apply SwiGLU, project down, then add the residual.
        rmsnorm(g_m.xb,g_m.x,g_m.m.tr[l].ffn_norm,g_m.m.n_embed,g_m.m.rms_epsilon);

        quantize_q8_K(g_m.xq,g_m.xb);
        matmul2(g_m.hb,g_m.hb2,&g_m.xb,&g_m.xq,NULL,g_m.m.tr[l].ffn_up,g_m.m.tr[l].ffn_gate,g_m.m.n_embed,g_m.m.n_ff); // up & gate

        // Apply SwiGLU: silu(gate) * up
        for (int i = 0; i < g_m.m.n_ff; i++) {
            const float g = g_m.hb2[i];
            g_m.hb[i] = (g / (1.0f + expf(-g))) * g_m.hb[i];
        }

        // Down projection (quantize hidden, matmul)
        quantize_q8_K(g_m.xq,g_m.hb);
        matmul(g_m.xb,&g_m.hb,&g_m.xq,NULL,g_m.m.tr[l].ffn_down,g_m.m.n_ff,g_m.m.n_embed);
        viz_trace_column(2 + l * 2,g_m.xb.data(),g_m.m.n_embed);

        // Residual back to x
        for (int j = 0; j < g_m.m.n_embed; j++) g_m.x[j] += g_m.xb[j];
    }

    // Final RMSNorm plus the output projection gives logits for the next token.
    rmsnorm(g_m.x,g_m.x,g_m.m.t_outnorm,g_m.m.n_embed,g_m.m.rms_epsilon);
    if (g_m.m.t_out.type == Q8_0 || g_m.m.t_out.type == Q1_0 || g_m.m.t_out.type == Q1_G) {
        quantize_q8_0(g_m.xq8,g_m.x);
        matmul(g_m.logits,&g_m.x,NULL,&g_m.xq8,g_m.m.t_out,g_m.m.n_embed,g_m.m.vocab_size);
    } else {
        quantize_q8_K(g_m.xq,g_m.x);
        matmul(g_m.logits,&g_m.x,&g_m.xq,NULL,g_m.m.t_out,g_m.m.n_embed,g_m.m.vocab_size);
    }
    viz_trace_column(1 + g_m.m.n_layers * 2,g_m.logits.data(),g_m.m.vocab_size);
}

int sampler_greedy()
{
    if (g_m.logits.empty()) return g_m.m.tok_eos;

    int best_id = 0;
    float best_v = g_m.logits[0];
    for (int i = 1; i < (int)g_m.logits.size(); i++) {
        if (g_m.logits[i] > best_v) {
            best_v = g_m.logits[i];
            best_id = i;
        }
    }

    return best_id;
}

static int cmp_sampler_ent_desc(const void* pa, const void* pb)
{
    const sampler_entry* a = (const sampler_entry*)pa;
    const sampler_entry* b = (const sampler_entry*)pb;

    if (a->p < b->p) return 1;
    if (a->p > b->p) return -1;
    return a->tok - b->tok;
}

int sampler_topp()
{
    const int nvoc = g_m.m.vocab_size;
    int n_maxlog = sampler_greedy();
    float maxlog = g_m.logits[n_maxlog];
    if (g_m.temp <= 0.0f) return n_maxlog;
    float inv_temp = 1.0f / g_m.temp;

    float sum = 0.0f;
    for (int i = 0; i < nvoc; i++) {
        float p = expf((g_m.logits[i] - maxlog) * inv_temp);
        g_m.samp[i].p = p;
        g_m.samp[i].tok = i;
        sum += p;
    }
    if (sum <= 0.0f) return n_maxlog;

    for (int i = 0; i < nvoc; i++) g_m.samp[i].p /= sum;
    qsort(g_m.samp.data(),nvoc,sizeof(g_m.samp[0]),cmp_sampler_ent_desc);

    float top = g_m.topp;
    if (top <= 0.0f) top = g_m.samp[0].p;
    if (top > 1.0f) top = 1.0f;
    int topk = g_m.topk;
    if (topk <= 0 || topk > nvoc) topk = nvoc;

    float cum = 0.0f;
    int cutoff = 0;
    for (; cutoff < topk; cutoff++) {
        cum += g_m.samp[cutoff].p;
        if (cum >= top) break;
    }
    if (cutoff >= topk) cutoff = topk - 1;

    float r = ((float)rand() / (float)RAND_MAX) * cum;
    float pcum = 0.0f;
    for (int i = 0; i <= cutoff; i++) {
        pcum += g_m.samp[i].p;
        if (pcum >= r) return g_m.samp[i].tok;
    }

    return g_m.samp[cutoff].tok;
}

bool load_model_state(const char* fn)
{
    FILE* f = fopen(fn,"rb");
    if (!f) {
        ERR("Can't read file %s",fn);
        return false;
    }

    model_file hdr;
    model_state ns = g_m;
    uint64_t sz;
    uint64_t lsz = 0;

    if (!fread(&hdr,sizeof(hdr),1,f)) goto err_read;
    if (!hdr.kv_size || !hdr.logits_size || !hdr.pos) goto err_read;

    ns.u_context = hdr.n_context;
    lsz = (uint64_t)hdr.pos * ns.m.kv_dim;
    if (hdr.kv_size != (uint64_t)ns.m.n_layers * lsz) goto err_read;

    ns.allocate();
    for (int q = 0; q < 2; q++) {
        uint8_t* dst = (uint8_t*)(q? ns.vc.data() : ns.kc.data());
        for (int l = 0; l < ns.m.n_layers; l++) {
            uint64_t off = (uint64_t)l * ns.m.n_context * ns.m.kv_dim * sizeof(gguf_half);
            sz = lsz * sizeof(gguf_half);
            if (!fread(dst + off,sz,1,f)) {
                ERR("Unable to read %s cache for layer %d from %s",(q? "Value":"Key"),l,fn);
                goto err_read;
            }
        }
    }

    if (hdr.logits_size != ns.logits.size()) goto err_read;
    sz = hdr.logits_size * sizeof(float);
    if (!fread(ns.logits.data(),sz,1,f)) {
        ERR("Unable to read logits from %s",fn);
        goto err_read;
    }
    fclose(f);

    ns.pos = hdr.pos;
    g_m = ns;
    return true;

err_read:
    fclose(f);
    return false;
}

bool save_model_state(const char* fn)
{
    if (!g_m.pos || !g_m.logits.size()) {
        ERR("Model state is empty - nothing to save to %s",fn);
        return false;
    }

    FILE* f = fopen(fn,"wb");
    if (!f) {
        ERR("Can't create file %s",fn);
        return false;
    }

    model_file hdr;
    memset(&hdr,0,sizeof(hdr));
    hdr.kv_size = (uint64_t)g_m.m.n_layers * g_m.pos * g_m.m.kv_dim;
    hdr.logits_size = g_m.logits.size();
    hdr.n_context = g_m.m.n_context;
    hdr.pos = g_m.pos;

    fwrite(&hdr,sizeof(hdr),1,f);

    uint64_t sz = (uint64_t)g_m.pos * g_m.m.kv_dim * sizeof(gguf_half); // per layer
    for (int q = 0; q < 2; q++) {
        uint8_t* src = (uint8_t*)(q? g_m.vc.data() : g_m.kc.data());
        for (int l = 0; l < g_m.m.n_layers; l++) {
            uint64_t off = (uint64_t)l * g_m.m.n_context * g_m.m.kv_dim * sizeof(gguf_half);
            fwrite(src + off,sz,1,f);
        }
    }

    sz = hdr.logits_size * sizeof(float);
    fwrite(g_m.logits.data(),sz,1,f);
    fclose(f);

    return true;
}

static void puttok(int tok)
{
    string text = tok_to_str(tok);
    if (!text.empty()) fwrite(text.data(),text.size(),1,stdout);
    fflush(stdout);
}

static void trim_line(string &s)
{
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
}

static void inline timed_inference(int tok)
{
    double t0 = tmsec();
    inference(tok);
    g_m.gen_ms += tmsec() - t0;
    g_m.ntimed++;
}

static int inline timed_sampling()
{
    double t0 = tmsec();
    int tok = g_m.greedy? sampler_greedy() : sampler_topp();
    g_m.samp_ms += tmsec() - t0;
    g_m.nsampled++;
    return tok;
}

void kv_backtrack(int n_tokens)
{
    if (n_tokens <= 0) return;
    if (n_tokens > g_m.pos) n_tokens = g_m.pos;
    g_m.pos -= n_tokens;
}

void feed_text(string s, bool echo, bool bos, bool space = false)
{
    if (space) s = " " + s;
    vector<int> toks = tokenize(s.c_str(),bos,false);
    if (g_m.pos + (int)toks.size() >= g_m.m.n_context) {
        ERR("Input does not fit into context window (%d + %zu >= %d)",g_m.pos,toks.size(),g_m.m.n_context);
        return;
    }

    for (int i = 0; i < (int)toks.size(); i++) {
        int tok = toks.at(i);
        if (!tok || tok == g_m.m.tok_eos) break;
        if (echo) puttok(tok);
        if (g_m.pos) viz_present(tok);
        timed_inference(tok);
    }
}

static void stats()
{
    if (g_m.ntimed && g_m.gen_ms > 0) {
        printf("\n\nInference-only tok/s: %.2f\n", (double)g_m.ntimed * 1000.0 / g_m.gen_ms);
        printf("Inference per-token average time: %.2f ms\n", g_m.gen_ms / (double)g_m.ntimed);
    }
    if (g_m.nsampled && g_m.samp_ms > 0)
        printf("Sampling per-token average time: %.2f ms", g_m.samp_ms / (double)g_m.nsampled);
}

void generate()
{
    feed_text(g_m.prompt,true,true);

    int ng = g_m.n_gen? g_m.n_gen : g_m.m.n_context;
    if (ng > g_m.m.n_context) ng = g_m.m.n_context;
    for (int tok,i = 0; i < ng; i++) {
        tok = timed_sampling();
        if (!tok || tok == g_m.m.tok_eos) break;

        puttok(tok);
        if (i) viz_present(tok);
        if (i < g_m.n_gen-1) timed_inference(tok);
    }
}

void pre_chat()
{
    feed_text(g_m.prompt,true,true);

    if (!g_m.save_state.empty()) {
        printf("\nSaving model state... ");
        fflush(stdout);
        if (save_model_state(g_m.save_state.c_str())) puts("done!");
    }
}

void chat()
{
    char pline[MAXLINE];
    bool pref_found = false;
    const bool pref_in_eot = (g_m.eot.find(g_m.usrname) != string::npos);

    while (g_m.pos < g_m.m.n_context) {
        if (!pref_found) {
            if (g_m.usrname.empty()) printf("\n> ");
            else printf("\n%s: ",g_m.usrname.c_str());
            fflush(stdout);
            g_m.chat_log += "\n";
        }

        string line;
        if (!fgets(pline,sizeof(pline),stdin)) break;
        line = pline;
        trim_line(line);
        if (line.empty()) break;

        string turn = pref_found? " " : g_m.usrname + ": ";
        turn += line + "\n";
        feed_text(turn,false,false,true);

        vecdb mem = vdb_query(line);
        if (!mem.empty()) {
            DBG("Injecting %zu VDB records before AI reply",mem.size());
            feed_text(vdb_to_prompt(mem),false,false);
        }

        if (!g_m.ainame.empty()) {
            feed_text(g_m.ainame+":",false,false,true);
            printf("%s:",g_m.ainame.c_str());
        } else
            printf("AI:");

        fflush(stdout);
        pref_found = false;

        string reply;
        int ntoks = 0;
        int ngen = 0;
        while (g_m.pos < g_m.m.n_context) {
            int tok = timed_sampling();
            if (!tok || tok == g_m.m.tok_eos) break;
            if (g_m.pos) viz_present(tok);

            timed_inference(tok);
            reply += tok_to_str(tok);
            puttok(tok);
            ntoks++;
            if (++ngen >= g_m.n_gen && g_m.n_gen) break;

            if (!g_m.eot.empty() && reply.find(g_m.eot) != string::npos) {
                pref_found = pref_in_eot;
                //DBG("EOT found, pref = %d",pref_found);
                break;
            }
        }

        vdb_log_turn(g_m.chat_log,line,reply);
    }
}

static void banner()
{
    printf("\nLiGGUF ver. %s by MatrixS_Master\n",VERSION);
    printf("Features enabled: %s\n\n",CAPABILITIES);
}

static void usage(const char* progname)
{
    printf("\nUsage: %s [options]\n",progname);
    puts("Available options:");
    puts("\t-m <model.gguf>");
    puts("\t-n <number_of_tokens_to_generate>");
    puts("\t-s <seed>");
    puts("\t-c <n_context>");
    puts("\t-p <prompt>");
    puts("\t-f <prompt_file_to_load>");
    puts("\t-l <chat_log_file>");
    puts("\t-S <model_state_file>: saves model state after prompt is processed");
    puts("\t-L <model_state_file>: loads previous model state");
    puts("\t-E <end_of_turn_marker>");
    puts("\t-A <AI_name>");
    puts("\t-U <User_name>");
    puts("\t-K <TopK>");
    puts("\t-T <TopP>");
    puts("\t-M <temperature>");
    puts("\t-B <BERT_model>");
    puts("\t-G: enable greedy sampling");
    puts("\t-C: enable chat mode");
    puts("\t-V: enable visualizer");
}

int parse_args(int argc, char* argv[])
{
    int opt = 0;
    while ((opt = getopt(argc,argv,"m:n:s:c:p:f:l:S:L:E:A:U:K:T:M:B:GCV")) != -1) {
        switch (opt) {
        case 'm':
            g_m.model_fn = optarg;
            break;
        case 'n':
            g_m.n_gen = atoi(optarg);
            break;
        case 's':
            srand(atoi(optarg));
            break;
        case 'c':
            g_m.u_context = atoi(optarg);
            break;
        case 'p':
            g_m.prompt = optarg;
            break;
        case 'f':
            g_m.prompt = load_text_file(optarg);
            break;
        case 'l':
            g_m.chat_file = optarg;
            g_m.chat_log = load_text_file(optarg);
            break;
        case 'S':
            g_m.save_state = optarg;
            break;
        case 'L':
            g_m.load_state = optarg;
            break;
        case 'E':
            g_m.eot = optarg;
            break;
        case 'A':
            g_m.ainame = optarg;
            break;
        case 'U':
            g_m.usrname = optarg;
            break;
        case 'K':
            g_m.topk = atoi(optarg);
            break;
        case 'T':
            g_m.topp = atof(optarg);
            break;
        case 'M':
            g_m.temp = atof(optarg);
            break;
        case 'B':
            g_m.bert_model_fn = optarg;
            break;
        case 'G':
            g_m.greedy = true;
            break;
        case 'C':
            g_m.chat = true;
            break;
        case 'V':
            viz_enabled = true;
            break;
        default:
            return -1;
        }
    }

    return optind;
}

int main(int argc, char* argv[])
{
    srand(time(NULL));
    fp1632_init();

    banner();
    if (parse_args(argc,argv) < 0) {
        usage(argv[0]);
        return 1;
    }

    assert(g_m.m.open_mmap(g_m.model_fn.c_str()));
    assert(g_m.m.read_gguf());
    g_m.allocate();
    assert(g_m.m.read_tokenizer());

    if (!g_m.load_state.empty() && !load_model_state(g_m.load_state.c_str())) return 2;

    viz_init(g_m.m.n_layers*2+2,VIZ_NNUMROWS);

    if (g_m.chat) {
        if (!g_m.bert_model_fn.empty()) {
            vdb_set_model(g_m.bert_model_fn);
            if (!g_m.ainame.empty() && !g_m.usrname.empty()) vdb_set_names(g_m.ainame,g_m.usrname);
            vdb_create(g_m.chat_log);
        }
        if (g_m.load_state.empty()) pre_chat();
        chat();
        if (!g_m.chat_file.empty() && !g_m.chat_log.empty())
            save_text_file(g_m.chat_file.c_str(),g_m.chat_log);
    } else
        generate();

    stats();

    viz_quit();
    g_m.m.close_mmap();
    puts("\nDone.");
    return 0;
}
