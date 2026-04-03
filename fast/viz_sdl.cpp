/* LiGGUF - a small, dependency-free LLaMA inference engine with direct GGUF support
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025-2026
 * */

#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <SDL.h>
#include "common.h"
#include "tokenize.h"
#include "viz_sdl.h"

struct viz_state {
    SDL_Window* win;
    SDL_Renderer* ren;
    SDL_Texture* tex;
    viz_trace trace;
    int win_w;
    int win_h;
    bool paused;
    bool ready;
};

bool viz_enabled = false;
static viz_state g_viz;

static void blit_frame()
{
    SDL_SetRenderTarget(g_viz.ren,NULL);
    SDL_SetRenderDrawBlendMode(g_viz.ren,SDL_BLENDMODE_NONE);
    SDL_SetRenderDrawColor(g_viz.ren,0,0,0,255);
    SDL_RenderClear(g_viz.ren);
    SDL_RenderCopy(g_viz.ren,g_viz.tex,NULL,NULL);
    SDL_RenderPresent(g_viz.ren);
}

static void viz_pump()
{
    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
        switch (ev.type) {
            case SDL_QUIT:
                g_viz.ready = false;
                break;
            case SDL_KEYDOWN:
                if (ev.key.keysym.sym == SDLK_SPACE) g_viz.paused = !g_viz.paused;
                break;
            case SDL_WINDOWEVENT:
                if (ev.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                    g_viz.win_w = ev.window.data1;
                    g_viz.win_h = ev.window.data2;
                }
                break;
        }
    }
}

static void fiery(uint8_t v, uint8_t* r, uint8_t* g, uint8_t* b)
{
    int sv = (int)v - 8;
    if (!sv) {
        *r = 0; *g = 0; *b = 0;
        return;
    }

    if (sv < 0) {
        sv = -sv;
        *r = 0;
        *g = 24 + sv * 10;
        *b = 70 + sv * 20;
        return;
    }

    if (sv < 4) {
        *r = 120 + sv * 20;
        *g = 16 + sv * 8;
        *b = 0;
        return;
    }

    if (sv < 7) {
        *r = 200 + (sv - 4) * 18;
        *g = 64 + (sv - 4) * 38;
        *b = 0;
        return;
    }

    *r = 255;
    *g = 210;
    *b = 170;
}

static void draw_neon_segment(int x0, int y0, int x1, int y1)
{
    SDL_SetRenderDrawBlendMode(g_viz.ren,SDL_BLENDMODE_ADD);
    SDL_SetRenderDrawColor(g_viz.ren,255,40,0,14);
    for (int d = -12; d <= 12; d++) SDL_RenderDrawLine(g_viz.ren,x0,y0 + d,x1,y1 + d);
    SDL_SetRenderDrawColor(g_viz.ren,255,90,0,22);
    for (int d = -9; d <= 9; d++) SDL_RenderDrawLine(g_viz.ren,x0,y0 + d,x1,y1 + d);
    SDL_SetRenderDrawColor(g_viz.ren,255,150,16,34);
    for (int d = -6; d <= 6; d++) SDL_RenderDrawLine(g_viz.ren,x0,y0 + d,x1,y1 + d);
    SDL_SetRenderDrawColor(g_viz.ren,255,210,70,52);
    for (int d = -3; d <= 3; d++) SDL_RenderDrawLine(g_viz.ren,x0,y0 + d,x1,y1 + d);

    SDL_SetRenderDrawBlendMode(g_viz.ren,SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(g_viz.ren,255,250,220,255);
    for (int d = -1; d <= 1; d++) SDL_RenderDrawLine(g_viz.ren,x0,y0 + d,x1,y1 + d);
}

static int collect_1111(int col, int* rows, int max_rows)
{
    int n = 0;
    int last = -2;
    for (int row = 0; row < g_viz.trace.rows; row++) {
        if (g_viz.trace.get(col,row) != 15) continue;
        if (row == last + 1) {
            rows[n - 1] = row;
            last = row;
            continue;
        }
        if (n < max_rows) rows[n++] = row;
        last = row;
    }

    if (n) return n;

    rows[0] = g_viz.trace.hot[col];
    return 1;
}

bool viz_init(int cols, int rows)
{
    if (!viz_enabled) return true;

    g_viz = {};
    if (SDL_Init(SDL_INIT_VIDEO)) {
        ERR("SDL_Init failed: %s",SDL_GetError());
        return false;
    }

    g_viz.win_w = 1280;
    g_viz.win_h = 720;
    g_viz.win = SDL_CreateWindow("LiGGUF hotpath",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,g_viz.win_w,g_viz.win_h,SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    if (!g_viz.win) {
        ERR("SDL_CreateWindow failed: %s",SDL_GetError());
        SDL_Quit();
        return false;
    }

    g_viz.ren = SDL_CreateRenderer(g_viz.win,-1,SDL_RENDERER_ACCELERATED);
    if (!g_viz.ren) g_viz.ren = SDL_CreateRenderer(g_viz.win,-1,SDL_RENDERER_SOFTWARE);
    if (!g_viz.ren) {
        ERR("SDL_CreateRenderer failed: %s",SDL_GetError());
        SDL_DestroyWindow(g_viz.win);
        SDL_Quit();
        return false;
    }

    g_viz.tex = SDL_CreateTexture(g_viz.ren,SDL_PIXELFORMAT_RGBA8888,SDL_TEXTUREACCESS_TARGET,g_viz.win_w,g_viz.win_h);
    if (!g_viz.tex) {
        ERR("SDL_CreateTexture failed: %s",SDL_GetError());
        SDL_DestroyRenderer(g_viz.ren);
        SDL_DestroyWindow(g_viz.win);
        SDL_Quit();
        return false;
    }

    g_viz.trace.resize(cols,rows);
    g_viz.ready = true;
    return true;
}

void viz_begin_frame()
{
    if (!viz_enabled || !g_viz.ready) return;
    viz_pump();
    if (!g_viz.ready) return;
    g_viz.trace.clear();
}

void viz_trace_column(int col, const float* x, int n)
{
    if (!viz_enabled || !g_viz.ready || !x || col < 0 || col >= g_viz.trace.cols || n <= 0) return;

    const int rows = g_viz.trace.rows;
    ftensor bins(rows,0);

    float vmax = 0.0f;
    float vmin = 0.0f;
    int hot = 0;
    for (int i = 0; i < n; i++) {
        int row = (int)((int64_t)i * rows / n);
        if (row >= rows) row = rows - 1;
        float v = x[i];
        if (v > 0.0f) {
            if (v > bins[row]) bins[row] = v;
            if (v > vmax) vmax = v;
        } else {
            if (v < bins[row]) bins[row] = v;
            if (v < vmin) vmin = v;
        }
    }

    for (int row = 0; row < rows; row++) {
        if (bins[row] > 0.0f && bins[row] > vmax * 0.5f) {
            hot = row;
        }
    }

    g_viz.trace.hot[col] = hot;
    for (int row = 0; row < rows; row++) {
        float v = bins[row];
        int q = 8;
        if (v > 0.0f && vmax > 0.0f) q = 8 + (int)(7.99f * v / vmax);
        else if (v < 0.0f && vmin < 0.0f) q = 8 - (int)(7.99f * v / vmin);
        if (q < 0) q = 0;
        if (q > 15) q = 15;
        g_viz.trace.set(col,row,q);
    }
}

void viz_present(int tok)
{
    if (!viz_enabled || !g_viz.ready) return;

    viz_pump();
    if (!g_viz.ready) return;
    while (g_viz.paused && g_viz.ready) {
        viz_pump();
        blit_frame();
        usleep(60000);
    }
    if (!g_viz.ready) return;

    char title[128];
    snprintf(title,sizeof(title),"LiGGUF hotpath | tok %d [%s]%s",tok,tok_to_str(tok).c_str(),g_viz.paused ? " | paused" : "");
    SDL_SetWindowTitle(g_viz.win,title);
    SDL_SetRenderTarget(g_viz.ren,g_viz.tex);
    SDL_SetRenderDrawBlendMode(g_viz.ren,SDL_BLENDMODE_NONE);
    SDL_SetRenderDrawColor(g_viz.ren,0,0,0,255);
    SDL_RenderClear(g_viz.ren);

    const int cols = g_viz.trace.cols;
    const int rows = g_viz.trace.rows;
    const float cw = (float)g_viz.win_w / (float)cols;
    const float ch = (float)g_viz.win_h / (float)rows;

    for (int col = 0; col < cols; col++) {
        int x0 = (int)(col * cw);
        int x1 = (int)((col + 1) * cw + 1.0f);
        for (int row = 0; row < rows; row++) {
            uint8_t q = g_viz.trace.get(col,row);
            if (!q) continue;
            uint8_t r,g,b;
            fiery(q,&r,&g,&b);
            SDL_SetRenderDrawColor(g_viz.ren,r / 5,g / 5,b / 5,255);
            SDL_Rect rc = { x0, g_viz.win_h - (int)((row + 1) * ch), x1 - x0, (int)(ch + 1.0f) };
            SDL_RenderFillRect(g_viz.ren,&rc);
        }
    }

    //TODO: de-hardcode magics
    for (int col = 0; col + 1 < cols; col++) {
        int lhs[64], rhs[64];
        int nl = collect_1111(col,lhs,64);
        int nr = collect_1111(col + 1,rhs,64);
        int x0 = (int)((col + 0.5f) * cw);
        int x1 = (int)((col + 1.5f) * cw);
        int n = nl > nr ? nl : nr;
        for (int i = 0; i < n; i++) {
            int yl = lhs[i < nl ? i : nl - 1];
            int yr = rhs[i < nr ? i : nr - 1];
            int y0 = g_viz.win_h - (int)((yl + 0.5f) * ch);
            int y1 = g_viz.win_h - (int)((yr + 0.5f) * ch);
            draw_neon_segment(x0,y0,x1,y1);
        }
    }

    blit_frame();
}

void viz_quit()
{
    if (!viz_enabled) return;
    if (g_viz.tex) SDL_DestroyTexture(g_viz.tex);
    if (g_viz.ren) SDL_DestroyRenderer(g_viz.ren);
    if (g_viz.win) SDL_DestroyWindow(g_viz.win);
    g_viz = {};
    SDL_Quit();
}
