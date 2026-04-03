/* LiGGUF - a small, dependency-free LLaMA inference engine with direct GGUF support
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025-2026
 * */

#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "common.h"
#include "lil_math.h"
#include "vdb.h"

using namespace std;

struct bert_runtime {
    gguf_model m;
    ftensor embd;
    ftensor embd2;
    ftensor q;
    ftensor k;
    ftensor v;
    ftensor ctx;
    ftensor att;
    ftensor row0;
    ftensor row1;
    ftensor row2;
    ftensor row3;
    ftensor rowff;
    qtensor xq8;
    string model_fn;
    bool ready;
};

static bert_runtime m_brt;
static vecdb m_vdb;
static string m_ai_name = "AI";
static string m_user_name = "User";
static string m_last_ai;

#define VDB_LOG_HDR "@@"
#define VDB_LOG_USER "<<<<<<<"
#define VDB_LOG_AI ">>>>>>>"

static inline bool bert_is_punct(unsigned char c)
{
    if ((c >= '!' && c <= '/') || (c >= ':' && c <= '@')) return true;
    if ((c >= '[' && c <= '`') || (c >= '{' && c <= '~')) return true;
    return false;
}

void vdb_set_model(string model_fn)
{
    if (m_brt.ready && m_brt.model_fn == model_fn) return;
    if (m_brt.ready) {
        m_brt.m.close_mmap();
        m_brt.ready = false;
    }
    m_brt.model_fn = model_fn;
}

static bool ensure_bert()
{
    if (m_brt.ready) return true;
    if (m_brt.model_fn.empty()) return false;
    assert(m_brt.m.open_mmap(m_brt.model_fn.c_str()));
    assert(m_brt.m.read_gguf());
    assert(m_brt.m.read_tokenizer());
    assert(m_brt.m.arch == "bert");
    m_brt.ready = true;
    DBG("VDB model ready: layers=%d embed=%d heads=%d context=%d",m_brt.m.n_layers,m_brt.m.n_embed,m_brt.m.n_heads,m_brt.m.n_context);
    return true;
}

static vector<int> bert_tokenize(string in)
{
    ensure_bert();
    assert(m_brt.ready);

    vector<int> out;
    for (auto &c : in) c = tolower(c);
    int i = 0;
    while (i < (int)in.size()) {
        while (i < (int)in.size() && isspace(in[i])) i++;
        if (i >= (int)in.size()) break;

        int j = i;
        while (j < (int)in.size() && !isspace(in[j]) && !bert_is_punct(in[j])) j++;
        if (j == i && bert_is_punct(in[j])) j++;

        bool bad = false;
        int pos = i;
        while (pos < j) {
            int found = -1;
            int found_end = pos;

            for (int end = j; end > pos; end--) {
                string part = in.substr(pos,end-pos);
                if (pos > i) part = "##" + part;
                if (m_brt.m.tokens_rev.count(part)) {
                    found = m_brt.m.tokens_rev[part];
                    found_end = end;
                    break;
                }
            }

            if (found < 0) {
                bad = true;
                break;
            }

            out.push_back(found);
            pos = found_end;
        }

        if (bad) out.push_back(m_brt.m.tok_unk);
        i = j;
    }

    return out;
}

static void bert_embed(const string &text, ftensor &out)
{
    ensure_bert();
    assert(m_brt.ready);

    vector<int> toks = bert_tokenize(text);
    const gguf_model &m = m_brt.m;
    const int n_embed = m.n_embed;
    const int head_dim = m.head_dim;
    const int n_heads = m.n_heads;
    const float att_scale = 1.0f / sqrtf(head_dim);

    if ((int)toks.size() > m.n_context - 2) toks.resize(m.n_context - 2);
    toks.insert(toks.begin(),m.tok_cls);
    toks.push_back(m.tok_sep);

    const int ntok = toks.size();
    assert(ntok > 0 && ntok <= m.n_context);

    m_brt.embd.resize((size_t)ntok * n_embed);
    m_brt.embd2.resize(m_brt.embd.size());
    m_brt.q.resize(m_brt.embd.size());
    m_brt.k.resize(m_brt.embd.size());
    m_brt.v.resize(m_brt.embd.size());
    m_brt.ctx.resize(m_brt.embd.size());
    m_brt.att.resize(ntok);
    m_brt.row0.resize(n_embed);
    m_brt.row1.resize(n_embed);
    m_brt.row2.resize(n_embed);
    m_brt.row3.resize(n_embed);
    m_brt.rowff.resize(m.n_ff);

    dequantize_row(m.bert_tok_types.type,m.bert_tok_types.ptr,m_brt.row2.data(),n_embed);

    for (int t = 0; t < ntok; t++) {
        float* x = m_brt.embd.data() + (size_t)t * n_embed;
        uint8_t* tok_ptr = m.bert_tok_embd.ptr + (size_t)toks[t] * m.bert_tok_embd.rsz;
        uint8_t* pos_ptr = m.bert_pos_embd.ptr + (size_t)t * m.bert_pos_embd.rsz;

        dequantize_row(m.bert_tok_embd.type,tok_ptr,m_brt.row0.data(),n_embed);
        dequantize_row(m.bert_pos_embd.type,pos_ptr,m_brt.row1.data(),n_embed);

        for (int i = 0; i < n_embed; i++) m_brt.row0[i] = m_brt.row0[i] + m_brt.row1[i] + m_brt.row2[i];
        layernorm(m_brt.row0,m_brt.row0,m.bert_emb_norm_w,m.bert_emb_norm_b,n_embed,m.rms_epsilon);
        memcpy(x,m_brt.row0.data(),n_embed * sizeof(float));
    }

    for (int l = 0; l < m.n_layers; l++) {
        const bert_block &b = m.bert[l];

        for (int t = 0; t < ntok; t++) {
            float* x = m_brt.embd.data() + (size_t)t * n_embed;
            float* q = m_brt.q.data() + (size_t)t * n_embed;
            float* k = m_brt.k.data() + (size_t)t * n_embed;
            float* v = m_brt.v.data() + (size_t)t * n_embed;

            memcpy(m_brt.row0.data(),x,n_embed * sizeof(float));
            matmul3(m_brt.row1,m_brt.row2,m_brt.row3,&m_brt.row0,NULL,&m_brt.xq8,b.attn_q,b.attn_k,b.attn_v,n_embed,n_embed,n_embed,n_embed);

            for (int i = 0; i < n_embed; i++) {
                q[i] = m_brt.row1[i] + b.attn_q_bias[i];
                k[i] = m_brt.row2[i] + b.attn_k_bias[i];
                v[i] = m_brt.row3[i] + b.attn_v_bias[i];
            }
        }

        for (int t = 0; t < ntok; t++) {
            float* ctx = m_brt.ctx.data() + (size_t)t * n_embed;
            float* q = m_brt.q.data() + (size_t)t * n_embed;

            for (int i = 0; i < n_embed; i++) ctx[i] = 0.0f;

            for (int h = 0; h < n_heads; h++) {
                float* att = m_brt.att.data();
                float* ch = ctx + h * head_dim;
                float* qh = q + h * head_dim;

                for (int tt = 0; tt < ntok; tt++) {
                    float* kh = m_brt.k.data() + (size_t)tt * n_embed + h * head_dim;
                    att[tt] = vec_dot_f32(qh,kh,head_dim) * att_scale;
                }
                softmax(att,ntok);

                for (int tt = 0; tt < ntok; tt++) {
                    float* vh = m_brt.v.data() + (size_t)tt * n_embed + h * head_dim;
                    const float a = att[tt];
                    for (int i = 0; i < head_dim; i++) ch[i] += a * vh[i];
                }
            }
        }

        for (int t = 0; t < ntok; t++) {
            float* x = m_brt.embd.data() + (size_t)t * n_embed;
            float* x2 = m_brt.embd2.data() + (size_t)t * n_embed;
            float* ctx = m_brt.ctx.data() + (size_t)t * n_embed;

            memcpy(m_brt.row0.data(),ctx,n_embed * sizeof(float));
            matmul(m_brt.row1,&m_brt.row0,NULL,&m_brt.xq8,b.attn_out,n_embed,n_embed);
            for (int i = 0; i < n_embed; i++) m_brt.row1[i] += b.attn_out_bias[i] + x[i];
            layernorm(m_brt.row0,m_brt.row1,b.attn_norm_w,b.attn_norm_b,n_embed,m.rms_epsilon);

            matmul(m_brt.rowff,&m_brt.row0,NULL,&m_brt.xq8,b.ffn_up,n_embed,m.n_ff);
            for (int i = 0; i < m.n_ff; i++) m_brt.rowff[i] += b.ffn_up_bias[i];
            gelu(m_brt.rowff);

            matmul(m_brt.row1,&m_brt.rowff,NULL,&m_brt.xq8,b.ffn_down,m.n_ff,n_embed);
            for (int i = 0; i < n_embed; i++) m_brt.row1[i] += b.ffn_down_bias[i] + m_brt.row0[i];
            layernorm(m_brt.row2,m_brt.row1,b.ffn_norm_w,b.ffn_norm_b,n_embed,m.rms_epsilon);
            memcpy(x2,m_brt.row2.data(),n_embed * sizeof(float));
        }

        m_brt.embd.swap(m_brt.embd2);
    }

    out.resize(n_embed);
    for (int i = 0; i < n_embed; i++) out[i] = 0.0f;
    for (int t = 0; t < ntok; t++) {
        float* x = m_brt.embd.data() + (size_t)t * n_embed;
        for (int i = 0; i < n_embed; i++) out[i] += x[i];
    }
    for (int i = 0; i < n_embed; i++) out[i] /= ntok;
    l2norm(out);
}

static vdb_entry make_entry(uint32_t ts, const string &prev_ai, const string &user, const string &ai)
{
    DBG("Making entry:\nAI0: '%s'\nUser: '%s'\nAI1: '%s'",prev_ai.c_str(),user.c_str(),ai.c_str());
    ftensor emb;
    vdb_entry e;
    e.timestamp = ts;
    e.context_before = prev_ai;
    e.user_text = user;
    e.context_after = ai;
    bert_embed(user,emb);
    e.vec.resize(emb.size());
    f32_to_f16_row(emb.data(),e.vec.data(),emb.size());
    DBG("VDB entry ts=%u prev_ai_len=%zu user_len=%zu ai_len=%zu emb=[%.4f %.4f %.4f %.4f]",e.timestamp,prev_ai.size(),user.size(),ai.size(),emb[0],emb[1],emb[2],emb[3]);
    return e;
}

void vdb_set_names(string ai_name, string user_name)
{
    m_ai_name = ai_name;
    m_user_name = user_name;
}

void vdb_create(string log_text)
{
    m_vdb.clear();
    if (!ensure_bert()) return;
    DBG("Processing chat log (%zu chars)...",log_text.length());

    int fsm = 0;
    string acc;
    vdb_entry ent;
    for (auto &i : log_text) {
        //DBG("fsm = %d; acc = '%s'",fsm,acc.c_str());
        switch (fsm) {
        case 0: // timestamp marker
            acc += i;
            if (acc == VDB_LOG_HDR) {
                acc.clear();
                ent = vdb_entry();
                fsm++;
            } else if (i == '\n')
                acc.clear();
            break;

        case 1: // timestamp
        case 2: // user line marker
            if (i == '\n') {
                if (fsm == 1) {
                    ent.timestamp = atoi(acc.c_str());
                    fsm++;
                } else {
                    if (acc == VDB_LOG_USER) fsm++; // user mark found
                }
                acc.clear();
            } else
                acc += i;
            break;

        case 3: // user line(s)
            if (i == '\n') {
                if (acc == VDB_LOG_AI) fsm++; // AI mark found
                else ent.user_text += acc + "\n";
                acc.clear();
            } else
                acc += i;
            break;

        case 4: // ai line(s)
            if (acc+i == VDB_LOG_HDR) { // Next timestamp mark found
                m_vdb.push_back(make_entry(ent.timestamp,ent.context_before,ent.user_text,ent.context_after));
                acc.clear();
                ent.context_before = ent.context_after; // shift context for the next entry
                ent.user_text.clear();
                ent.context_after.clear();
                fsm = 1;
            } else if (i == '\n') {
                ent.context_after += acc + "\n";
                acc.clear();
            } else
                acc += i;
            break;

        default: fsm = 0;
        }
    }

    if (fsm == 4) m_vdb.push_back(make_entry(ent.timestamp,ent.context_before,ent.user_text,ent.context_after+acc));
    DBG("VDB created: %zu records",m_vdb.size());
}

vecdb vdb_query(string intext)
{
    if (!ensure_bert()) return vecdb();

    ftensor emb;
    float best[VDB_MAX_RETRIEVAL];
    int best_idx[VDB_MAX_RETRIEVAL];
    for (int i = 0; i < VDB_MAX_RETRIEVAL; i++) {
        best[i] = -1e30f;
        best_idx[i] = -1;
    }

    bert_embed(intext,emb);
    DBG("VDB query len=%zu emb=[%.4f %.4f %.4f %.4f]",intext.size(),emb[0],emb[1],emb[2],emb[3]);

    for (int i = 0; i < (int)m_vdb.size(); i++) {
        if (m_vdb[i].vec.size() != emb.size()) continue;
        float sim = vec_dot_f16_f32(m_vdb[i].vec.data(),emb.data(),emb.size());
        DBG("VDB cand idx=%d sim=%.4f ts=%u",i,sim,m_vdb[i].timestamp);
        if (sim < VDB_LIKENESS_THRESHOLD) continue;

        for (int k = 0; k < VDB_MAX_RETRIEVAL; k++) {
            if (sim <= best[k]) continue;
            for (int s = VDB_MAX_RETRIEVAL - 1; s > k; s--) {
                best[s] = best[s-1];
                best_idx[s] = best_idx[s-1];
            }
            best[k] = sim;
            best_idx[k] = i;
            break;
        }
    }

    vecdb out;
    for (int i = 0; i < VDB_MAX_RETRIEVAL; i++) {
        if (best_idx[i] < 0) continue;
        DBG("VDB hit %d idx=%d sim=%.4f ts=%u",i,best_idx[i],best[i],m_vdb[best_idx[i]].timestamp);
        out.push_back(m_vdb[best_idx[i]]);
    }
    return out;
}

static string rec_to_prompt(const vdb_entry &e)
{
    char tsbuf[128];
    time_t tt = e.timestamp;
    struct tm tmv;
    localtime_r(&tt,&tmv);
    strftime(tsbuf,sizeof(tsbuf),"%Y-%m-%d %H:%M:%S %Z",&tmv);

    string out = "At ";
    out += tsbuf;
    if (!e.context_before.empty()) {
        out += " I (";
        out += m_ai_name;
        out += ") said:\n";
        out += e.context_before;
        out += "\nThen ";
    } else
        out += " ";
    out += m_user_name;
    out += " said:\n";
    out += e.user_text;
    if (!e.context_after.empty()) {
        out += "\nThen I replied:\n";
        out += e.context_after;
    }
    out += "\n";
    return out;
}

string vdb_to_prompt(const vecdb &mem)
{
    char hdr[128];
    snprintf(hdr,sizeof(hdr),"Attached are %zu records found in my memory database:\n\n",mem.size());

    string out = hdr;
    for (int i = 0; i < (int)mem.size(); i++) {
        char tag[32];
        snprintf(tag,sizeof(tag),"*** Record %d\n",i + 1);
        out += tag;
        out += rec_to_prompt(mem[i]);
        out += "\n";
    }
    out += "*** End of records";
    DBG("VDB prompt = '%s'",out.c_str());
    return out;
}

void vdb_log_turn(string &log, string user, string ai)
{
    char hdr[64];
    uint32_t ts = time(NULL);
    snprintf(hdr,sizeof(hdr),"%s%u\n%s\n",VDB_LOG_HDR,ts,VDB_LOG_USER);

    log += hdr + user + "\n" + VDB_LOG_AI + "\n" + ai + "\n";

    if (ensure_bert()) m_vdb.push_back(make_entry(ts,m_last_ai,user,ai));
    m_last_ai = ai;
}
