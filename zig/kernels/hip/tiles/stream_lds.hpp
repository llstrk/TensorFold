#pragma once

// The stream tile for several-round products of many columns at three or four rows (qkvz, the routed experts'
// gate|up): the block's rows' chunks are staged in LDS once, as stream_x_pairs' pairs and their sums with ones, and
// its waves walk several column groups with the next (group, round)'s words in flight instead of re-reading x from
// memory for every few columns. A column's chunk order (stream_body's rotation by col0 >> 3), the per-lane folds and
// the xor tree are stream_body's, so the bits are the stream tile's.

#include "tiles/stream.hpp"

namespace tf {
namespace rocm {

constexpr int kLdsWaves = 4;

template <class Dec, class T, class Epi, int R, int CB, int LOG, bool WIDE>
__device__ inline void stream_lds_body(typename Dec::Args& a, int gshift, int by, int wave_id, int nwaves, int pair_cols,
                                       float limit, u32x4* xs, float* sxs) {
    using pair = typename T::pair;
    constexpr bool PAIR = Epi::kPair;
    constexpr uint32_t kOne = T::ones;
    constexpr int KW = Dec::kWords;
    constexpr int L = 1 << LOG;
    constexpr int CH = PAIR ? CB / 2 : CB;
    constexpr int per_group = (32 >> LOG) * CH;
    const int lane = threadIdx.x & 31;
    const int j = lane & (L - 1);
    const int slot = lane >> LOG;
    const int nch = a.k >> 5;
    const int row0 = by * R;

    // stage the rows' chunks: [r][v][q] as u32x4 so a wave's lanes read 512 contiguous bytes; only the item's rows
    // (a routed expert at four rows has one or two), the products stop at them
    const int staged = a.m - row0 < R ? a.m - row0 : R;
    for (int t = threadIdx.x; t < staged * nch; t += blockDim.x) {
        const int r = t / nch;
        const int q = t - r * nch;
        const u32x4* src = reinterpret_cast<const u32x4*>(IdentityAct<T>::row(a, row0 + r < a.m ? row0 + r : 0) +
                                                          static_cast<long long>(q) * 32);
        uint32_t xn[16];
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const u32x4 u = src[v];
            xn[4 * v] = u.x;
            xn[4 * v + 1] = u.y;
            xn[4 * v + 2] = u.z;
            xn[4 * v + 3] = u.w;
        }
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) s = T::dot(__builtin_bit_cast(pair, xn[i]), __builtin_bit_cast(pair, kOne), s);
        sxs[r * nch + q] = s;
        pair xp[16];
        stream_x_pairs<Dec, T>(xn, xp, std::make_integer_sequence<int, 16>{});
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            uint32_t* dst = reinterpret_cast<uint32_t*>(xs) + ((r * 4 + v) * nch + q) * 4;
#pragma unroll
            for (int e = 0; e < 4; ++e) dst[e] = __builtin_bit_cast(uint32_t, xp[4 * v + e]);
        }
    }
    __syncthreads();

    const int cols = PAIR ? pair_cols : a.n;
    const int ngroups = (cols + per_group - 1) / per_group;
    const int rounds = (nch + L - 1) >> LOG;
    const long long words_row = Dec::template row_words<long long>(a);
    const long long groups = a.k / a.group;
    const typename Dec::StreamTerms terms(a);

    // a group's first column for this lane and its first round (stream_body's col0 and rot)
    auto col0_of = [&](int g) { return g * per_group + slot * CH; };
    auto rot_of = [&](int c0) { return rounds > 1 ? (c0 >> 3) % rounds : 0; };
    auto base_of = [&](int rot, int r) {
        const int ri = r + rot >= rounds ? r + rot - rounds : r + rot;
        return ri << LOG;
    };
    auto fetch = [&](int c0, int base, uint32_t (&w)[CB][KW], unsigned short (&sr)[CB], unsigned short (&br)[CB]) {
        const int q = base + j < nch ? base + j : nch - 1;
#pragma unroll
        for (int c = 0; c < CB; ++c) {
            const int cj = PAIR ? c % CH : c;
            const long long col = (c0 + cj < cols ? c0 + cj : cols - 1) + (PAIR && c >= CH ? pair_cols : 0);
            Dec::template load<WIDE>(Dec::words(a) + col * words_row + static_cast<long long>(q) * KW, w[c]);
            const long long at = col * groups + (q >> gshift);
            sr[c] = terms.scale_raw(at);
            br[c] = terms.bias_raw(at);
        }
    };
    float acc[R][CB];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int c = 0; c < CB; ++c) acc[r][c] = 0.f;
    }
    auto work = [&](int base, const uint32_t (&w)[CB][KW], const unsigned short (&sr)[CB], const unsigned short (&br)[CB]) {
        const bool live = base + j < nch;
        const int q = live ? base + j : nch - 1;
        float sc[CB], bi[CB];
#pragma unroll
        for (int c = 0; c < CB; ++c) {
            sc[c] = live ? terms.widen(sr[c]) : 0.f;
            bi[c] = live ? terms.widen(br[c]) : 0.f;
        }
        pair qp[CB][16];
#pragma unroll
        for (int c = 0; c < CB; ++c) stream_codes<Dec, T>(w[c], qp[c], std::make_integer_sequence<int, 16>{});
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r >= a.m) break;
            pair xp[16];
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const uint32_t* srcl = reinterpret_cast<const uint32_t*>(xs) + ((r * 4 + v) * nch + q) * 4;
#pragma unroll
                for (int e = 0; e < 4; ++e) xp[4 * v + e] = __builtin_bit_cast(pair, srcl[e]);
            }
            const float sx = sxs[r * nch + q];  // (dword reads: vector-component bit casts to pairs miscompiled)
#pragma unroll
            for (int c = 0; c < CB; ++c) {
                float d = 0.f;
#pragma unroll
                for (int i = 0; i < 16; ++i) d = T::dot(xp[i], qp[c][i], d);
                acc[r][c] = Dec::fold_scale(acc[r][c], d, sc[c]);
                acc[r][c] = Dec::fold_bias(acc[r][c], sx, bi[c]);
            }
        }
    };
    auto finish = [&](int c0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float v[CB];
#pragma unroll
            for (int c = 0; c < CB; ++c) {
                v[c] = acc[r][c];
                acc[r][c] = 0.f;
#pragma unroll
                for (int d = L >> 1; d > 0; d >>= 1) v[c] += __shfl_xor(v[c], d, 32);
            }
            if (j != 0 || row0 + r >= a.m) continue;
#pragma unroll
            for (int c = 0; c < CB; ++c) {
                const int col = c0 + c % CH;
                if (col < cols && !__builtin_isfinite(v[c])) {
                    const int wc = col + (PAIR && c >= CH ? pair_cols : 0);
                    v[c] = Dec::template repair_stream<T>(a, IdentityAct<T>::row(a, row0 + r), wc, v[c]);
                }
            }
#pragma unroll
            for (int c = 0; c < CH; ++c) {
                const int col = c0 + c;
                if (col >= cols) continue;
                Epi::put(a, out_row(a, row0 + r), col, v, c, pair_cols, limit);
            }
        }
    };

    int g = wave_id;
    if (g >= ngroups) return;
    int c0 = col0_of(g), rot = rot_of(c0), rr = 0;
    uint32_t wa[CB][KW], wb[CB][KW];
    unsigned short sa[CB], ba[CB], s2[CB], b2[CB];
    fetch(c0, base_of(rot, 0), wa, sa, ba);
#pragma clang loop unroll(disable)
    while (true) {
        int g2 = g, rr2 = rr + 1;
        if (rr2 == rounds) {
            rr2 = 0;
            g2 = g + nwaves;
        }
        const int c02 = col0_of(g2);
        const int rot2 = rr2 == 0 ? rot_of(c02) : rot;
        const bool more = g2 < ngroups;
        if (more) fetch(c02, base_of(rot2, rr2), wb, s2, b2);
        work(base_of(rot, rr), wa, sa, ba);
        if (rr == rounds - 1) finish(c0);
        if (!more) break;
#pragma unroll
        for (int c = 0; c < CB; ++c) {
#pragma unroll
            for (int i = 0; i < KW; ++i) wa[c][i] = wb[c][i];
            sa[c] = s2[c];
            ba[c] = b2[c];
        }
        g = g2;
        rr = rr2;
        c0 = c02;
        rot = rot2;
    }
}

template <class Dec, class T, class Epi, int R, int CB, int LOG, class Sides>
__device__ inline void stream_lds_tile(typename Dec::Args& a, const Sides& sides, int gshift) {
    extern __shared__ u32x4 lds_x[];
    const int nby = (a.m + R - 1) / R;
    if (!Dec::take_item(a, blockIdx.z)) return;
    const int by = static_cast<int>(blockIdx.x) % nby;
    if (by * R >= a.m) return;
    // a group's side s owns wave blocks [first[s], first[s + 1]) and walks its own columns
    int bx = static_cast<int>(blockIdx.x) / nby;
    int blocks = static_cast<int>(gridDim.x) / nby;
    if (sides.count > 0) {
        int s = 0;
        while (s + 1 < sides.count && bx >= sides.first[s + 1]) ++s;
        blocks = sides.first[s + 1] - sides.first[s];
        bx -= sides.first[s];
        Dec::bind(a, sides, s);
        a.n = sides.n[s];
        if (sides.out_half) {
            a.out16 = static_cast<__half*>(sides.out[s]);
        } else {
            a.out = static_cast<float*>(sides.out[s]);
        }
    }
    const int nwaves = blocks * kLdsWaves;
    const int wave_id = bx * kLdsWaves + static_cast<int>(threadIdx.x >> 5);
    const int nch = a.k >> 5;
    float* sxs = reinterpret_cast<float*>(lds_x + R * 4 * nch);
    if constexpr (Dec::kWideBytes == 4) {
        stream_lds_body<Dec, T, Epi, R, CB, LOG, false>(a, gshift, by, wave_id, nwaves, sides.pair_cols, sides.limit, lds_x, sxs);
    } else {
        if (Dec::wide_aligned(a)) {
            stream_lds_body<Dec, T, Epi, R, CB, LOG, true>(a, gshift, by, wave_id, nwaves, sides.pair_cols, sides.limit, lds_x, sxs);
        } else {
            stream_lds_body<Dec, T, Epi, R, CB, LOG, false>(a, gshift, by, wave_id, nwaves, sides.pair_cols, sides.limit, lds_x, sxs);
        }
    }
}

}  // namespace rocm
}  // namespace tf

