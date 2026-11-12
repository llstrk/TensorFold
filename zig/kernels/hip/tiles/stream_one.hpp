#pragma once

// The stream tile for products of one round (K <= 32 * lanes a column: hc up, an expert's down): a lane holds the same
// chunk of every column, so a wave keeps its rows' x chunk and sum in registers and walks many column groups, the next
// group's words in flight while it folds this one. A lane's value, the lanes' xor tree and the epilogue are
// stream_body's, so the bits are the stream tile's; it only stops re-reading x for every few columns.

#include "tiles/stream.hpp"
#include "tiles/matrix_fold_mix.hpp"

namespace tf {
namespace rocm {

constexpr int kOneWaves = 4;   // waves a block
constexpr int kOneGroups = 4;  // column groups a wave walks

// Columns a lane by row tile: 4 at four rows, 2 below (stream_pick's narrowest at one and two rows).
constexpr int stream_one_cb(int rows) { return rows == 4 ? 4 : 2; }

// Where a lane's CB columns lie and what it does with their sums: the plain layout and Epi, or hc up's four streams.
// `Pre` is what a lane loads for the epilogue with a group's words; `finish` takes every row's sums after the tree.
template <class Epi, int CB, int LOG>
struct OnePlain {
    struct Extra {};
    struct Pre {};
    static constexpr int kPer = (32 >> LOG) * CB;  // outputs a group
    __device__ static int groups(const Extra&, int n) { return (n + kPer - 1) / kPer; }
    __device__ static int col(const Extra&, int g, int slot, int c) { return g * kPer + slot * CB + c; }
    template <int R, class Args>
    __device__ static Pre pre(const Args&, const Extra&, int, int, int, int) { return {}; }
    template <class Dec, class T, int R, class Args>
    __device__ static void finish(const Args& a, const Extra& e, int g, int slot, int j, int row0, float (&v)[R][CB], const Pre&) {
        if (j != 0) return;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row >= a.m) continue;
#pragma unroll
            for (int c = 0; c < CB; ++c) {
                const int cc = col(e, g, slot, c);
                if (cc >= a.n) continue;
                if (!__builtin_isfinite(v[r][c])) v[r][c] = Dec::template repair_stream<T>(a, IdentityAct<T>::row(a, row), cc, v[r][c]);
                Epi::put(a, out_row(a, row), cc, v[r], c, 0, 0.f);
            }
        }
    }
};

// Hyper-connection up projection (n = 4 dims) with tf_fn_hc_mix as the epilogue: a lane's four columns are the four
// streams of one dim, mixed[r, d] = mean over s of round(round(sigmoid(round(g))) * xs), the same bits as the launch.
// After the tree every lane of a column has the sums: lane j takes row j / 4's stream j % 4 (its xs read with the
// words), and the stream-0 lane sums the four in stream order.
template <int LOG>
struct OneHcMix {
    struct Extra {
        const uint16_t* xs;  // (m, 4 dims): the normed streams
        uint16_t* mixed;     // (m, dims)
        int dims;
    };
    using Pre = uint16_t;
    static constexpr int kPer = 32 >> LOG;  // dims a group
    __device__ static int groups(const Extra& e, int) { return (e.dims + kPer - 1) / kPer; }
    __device__ static int col(const Extra& e, int g, int slot, int c) {
        const int d = g * kPer + slot;
        return c * e.dims + (d < e.dims ? d : e.dims - 1);
    }
    template <int R, class Args>
    __device__ static Pre pre(const Args& a, const Extra& e, int g, int slot, int j, int row0) {
        const int d = g * kPer + slot;
        const int r = j >> 2;
        if (r >= R || row0 + r >= a.m || d >= e.dims) return 0;
        return e.xs[(static_cast<long long>(row0 + r) * 4 + (j & 3)) * e.dims + d];
    }
    template <class Dec, class T, int R, class Args>
    __device__ static void finish(const Args& a, const Extra& e, int g, int slot, int j, int row0, float (&v)[R][4], const Pre& xv) {
        static_assert(4 * R <= (1 << LOG), "a column's lanes take every row's streams");
        const int d = g * kPer + slot;
        const int r = j >> 2;
        const int s = j & 3;
        float gv = 0.f;
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
#pragma unroll
            for (int ss = 0; ss < 4; ++ss) gv = rr == r && ss == s ? v[rr][ss] : gv;
        }
        const bool on = r < R && row0 + r < a.m && d < e.dims;
        float t = 0.f;
        if (on) {
            if (!__builtin_isfinite(gv)) gv = Dec::template repair_stream<T>(a, IdentityAct<T>::row(a, row0 + r), s * e.dims + d, gv);
            const float gq = mix_f(mix_bf16(gv));
            const float gate = mix_f(mix_bf16(1.0f / (1.0f + expf(-gq))));
            t = mix_f(mix_bf16(gate * mix_f(xv)));
        }
        const float t1 = __shfl_down(t, 1, 32);
        const float t2 = __shfl_down(t, 2, 32);
        const float t3 = __shfl_down(t, 3, 32);
        if (on && s == 0) {
            float total = 0.f;
            total += t;
            total += t1;
            total += t2;
            total += t3;
            e.mixed[static_cast<long long>(row0 + r) * e.dims + d] = mix_bf16(total / 4.0f);
        }
    }
};

template <class Dec, class T, class Cols, int R, int CB, int LOG, bool WIDE>
__device__ inline void stream_one_body(typename Dec::Args& a, const typename Cols::Extra& e, int gshift, int by,
                                       int wave_id, int nwaves) {
    using pair = typename T::pair;
    constexpr uint32_t kOne = T::ones;
    constexpr int KW = Dec::kWords;
    constexpr int L = 1 << LOG;
    const int lane = threadIdx.x & 31;
    const int j = lane & (L - 1);
    const int slot = lane >> LOG;
    const int nch = a.k >> 5;
    const bool live = j < nch;
    const int q = live ? j : nch - 1;  // a lane past the last chunk reads it again, with zero scale and bias
    const int row0 = by * R;
    const long long words_row = Dec::template row_words<long long>(a);
    const long long groups = a.k / a.group;
    const typename Dec::StreamTerms terms(a);

    pair xp[R][16];
    float sx[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const u32x4* src = reinterpret_cast<const u32x4*>(IdentityAct<T>::row(a, row0 + r < a.m ? row0 + r : 0) +
                                                          static_cast<long long>(q) * 32);
        uint32_t xn[16];
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const u32x4 t = src[v];
            xn[4 * v] = t.x;
            xn[4 * v + 1] = t.y;
            xn[4 * v + 2] = t.z;
            xn[4 * v + 3] = t.w;
        }
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) s = T::dot(__builtin_bit_cast(pair, xn[i]), __builtin_bit_cast(pair, kOne), s);
        sx[r] = s;
        stream_x_pairs<Dec, T>(xn, xp[r], std::make_integer_sequence<int, 16>{});
    }

    const int ngroups = Cols::groups(e, a.n);
    using Pre = typename Cols::Pre;
    auto fetch = [&](int g, uint32_t (&w)[CB][KW], unsigned short (&sr)[CB], unsigned short (&br)[CB], Pre& p) {
        p = Cols::template pre<R>(a, e, g, slot, j, row0);
#pragma unroll
        for (int c = 0; c < CB; ++c) {
            const int cc = Cols::col(e, g, slot, c);
            const long long col = cc < a.n ? cc : a.n - 1;
            Dec::template load<WIDE>(Dec::words(a) + col * words_row + static_cast<long long>(q) * KW, w[c]);
            const long long at = col * groups + (q >> gshift);
            sr[c] = terms.scale_raw(at);
            br[c] = terms.bias_raw(at);
        }
    };
    auto work = [&](int g, const uint32_t (&w)[CB][KW], const unsigned short (&sr)[CB], const unsigned short (&br)[CB],
                    const Pre& p) {
        float v[R][CB] = {};
#pragma unroll
        for (int c = 0; c < CB; ++c) {
            const float sc = live ? terms.widen(sr[c]) : 0.f;
            const float bi = live ? terms.widen(br[c]) : 0.f;
            pair qp[16];
            stream_codes<Dec, T>(w[c], qp, std::make_integer_sequence<int, 16>{});
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (row0 + r >= a.m) break;  // an item's rows: a routed expert at four rows has one or two
                float d = 0.f;
#pragma unroll
                for (int i = 0; i < 16; ++i) d = T::dot(xp[r][i], qp[i], d);
                v[r][c] = Dec::fold_bias(Dec::fold_scale(0.f, d, sc), sx[r], bi);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r >= a.m) break;
#pragma unroll
            for (int c = 0; c < CB; ++c) {
#pragma unroll
                for (int d = L >> 1; d > 0; d >>= 1) v[r][c] += __shfl_xor(v[r][c], d, 32);
            }
        }
        Cols::template finish<Dec, T, R>(a, e, g, slot, j, row0, v, p);
    };
    // one work site in a rolled loop: the unrolled walk outgrows the instruction cache
    uint32_t wa[CB][KW], wb[CB][KW];
    unsigned short sa[CB], ba[CB], s2[CB], b2[CB];
    Pre pa{}, pb{};
    int g = wave_id;
    if (g < ngroups) fetch(g, wa, sa, ba, pa);
#pragma clang loop unroll(disable)
    for (; g < ngroups; g += nwaves) {
        if (g + nwaves < ngroups) fetch(g + nwaves, wb, s2, b2, pb);
        work(g, wa, sa, ba, pa);
        pa = pb;
#pragma unroll
        for (int c = 0; c < CB; ++c) {
#pragma unroll
            for (int i = 0; i < KW; ++i) wa[c][i] = wb[c][i];
            sa[c] = s2[c];
            ba[c] = b2[c];
        }
    }
}

// Item z of the plan (or the plain product); grid x = wave blocks times row blocks, the row block fastest.
template <class Dec, class T, class Cols, int R, int CB, int LOG>
__device__ inline void stream_one_tile(typename Dec::Args& a, const typename Cols::Extra& e, int gshift) {
    const int nby = (a.m + R - 1) / R;  // the launch's row blocks, before an item narrows m to its count
    if (!Dec::take_item(a, blockIdx.z)) return;
    const int by = static_cast<int>(blockIdx.x) % nby;
    if (by * R >= a.m) return;
    const int nwaves = static_cast<int>(gridDim.x) / nby * kOneWaves;
    const int wave_id = static_cast<int>(blockIdx.x) / nby * kOneWaves + static_cast<int>(threadIdx.x >> 5);
    if constexpr (Dec::kWideBytes == 4) {
        stream_one_body<Dec, T, Cols, R, CB, LOG, false>(a, e, gshift, by, wave_id, nwaves);
    } else {
        if (Dec::wide_aligned(a)) {
            stream_one_body<Dec, T, Cols, R, CB, LOG, true>(a, e, gshift, by, wave_id, nwaves);
        } else {
            stream_one_body<Dec, T, Cols, R, CB, LOG, false>(a, e, gshift, by, wave_id, nwaves);
        }
    }
}

}  // namespace rocm
}  // namespace tf
