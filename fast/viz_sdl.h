#pragma once

#include <stdint.h>
#include <vector>

#define VIZ_NNUMROWS 256

struct viz_trace {
    int cols;
    int rows;
    std::vector<uint8_t> packed;
    std::vector<uint16_t> hot;

    void resize(int ncols, int nrows)
    {
        cols = ncols;
        rows = nrows;
        packed.resize((size_t)(cols * rows + 1) / 2);
        hot.resize(cols);
    }

    void clear()
    {
        memset(packed.data(),0,packed.size());
        memset(hot.data(),0,hot.size() * sizeof(hot[0]));
    }

    void set(int col, int row, uint8_t v)
    {
        size_t idx = (size_t)col * rows + row;
        size_t off = idx >> 1;
        int shift = (idx & 1) << 2;
        packed[off] = (packed[off] & ~(15 << shift)) | ((v & 15) << shift);
    }

    uint8_t get(int col, int row) const
    {
        size_t idx = (size_t)col * rows + row;
        return (packed[idx >> 1] >> ((idx & 1) << 2)) & 15;
    }
};

#ifdef USE_SDL

bool viz_init(int cols, int rows);
void viz_begin_frame();
void viz_trace_column(int col, const float* x, int n);
void viz_present(int tok);
void viz_quit();

extern bool viz_enabled;

#else

static inline bool viz_init(int cols, int rows) { return true; }
static inline void viz_begin_frame() {}
static inline void viz_trace_column(int col, const float* x, int n) {}
static inline void viz_present(int tok) {}
static inline void viz_quit() {}

bool viz_enabled;

#endif /*USE_SDL*/
