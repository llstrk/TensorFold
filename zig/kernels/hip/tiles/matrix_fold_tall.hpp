#pragma once

// The routed fold tiles of the tall plan (tf_moe_route's tile 256): an expert's pairs past its first 128 in items of
// up to 256 rows, its last 128 or fewer as before. An item of more than 128 rows runs on a 256 x 128 block (16 waves
// as 8 x 2), so a stage's weights are fetched and decoded once for 256 rows; the 128-row tiles take the others. Each
// wave makes the same 32 x 64 sums in wmma_fold_cols_tile's order from the same staged values: the same bits.

#include "tiles/matrix_fold_mix.hpp"

namespace tf {
namespace rocm {

constexpr int kTallRows = 256;

// An item of more than 128 rows (routed launches only): the tall tile's, not the 128-row tiles'.
__device__ __forceinline__ bool tall_item(const Affine& a, int z) {
    return a.route.items != nullptr && a.route.items[3 * z + 2] > kGemmM;
}

template <class Dec, class Act, class T, class Cols>
__device__ __forceinline__ void wmma_fold_tall_tile(typename Dec::Args& a, const typename Cols::Extra& e) {
#if TF_DEVICE_WMMA_GFX11
    if (!tall_item(a, blockIdx.z)) return;
    if (!Dec::take_item(a, blockIdx.z)) return;
    const int bx = blockIdx.x;
    const int m0 = 0;
    __shared__ __attribute__((aligned(16))) uint32_t xs[kTallRows * kFoldLdw];
    __shared__ __attribute__((aligned(16))) uint32_t ws[kGemmN * kFoldLdw];

    const int tid = threadIdx.x;
    const int words_row = Dec::template row_words<int>(a);
    const int groups = a.k / a.group;
    const int stages = a.k / kFoldK;
    const int rows_left = a.m;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int lane = tid & 31;

    // weights: threads 0..255 a column's 32-code chunk as in the 128-row tile; x: every thread half a row's 64 codes
    const int line = tid & (kGemmN - 1);
    const int chunk = (tid >> 7) & 1;
    const bool wtask = tid < 2 * kGemmN;
    const int col_s = Cols::col(e, bx, line);
    const uint32_t* wsrc = Dec::words(a) + static_cast<long long>(col_s) * words_row;
    const long long tbase = static_cast<long long>(col_s) * groups;
    const int xrow = tid >> 1, xhalf = tid & 1;
    const bool xlive = xrow < rows_left;
    const int row_s = xlive ? m0 + xrow : m0;
    const u32x4* xsrc = reinterpret_cast<const u32x4*>(Act::row(a, row_s));
    uint32_t wreg[Dec::kWords];
    u32x4 xreg[4];
    typename Dec::Term term;
    auto fetch = [&](int st) {
        const int c = st * 2 + chunk;  // the column's 32-code chunk
        if (wtask) {
            Dec::template load<true>(wsrc + c * Dec::kWords, wreg);
            term = Dec::term(a, tbase + (c * 32) / a.group);
        }
        if (xlive) {
#pragma unroll
            for (int v = 0; v < 4; ++v) xreg[v] = xsrc[st * 8 + xhalf * 4 + v];
        } else {
#pragma unroll
            for (int v = 0; v < 4; ++v) xreg[v] = u32x4{0u, 0u, 0u, 0u};
        }
    };
    auto stash = [&]() {
        auto pack = [](float w0, float w1) {  // BF16 rounded half up: + 0x8000, then one permute takes both halves
            return __builtin_amdgcn_perm(__float_as_uint(w1) + 0x8000u, __float_as_uint(w0) + 0x8000u, 0x07060302u);
        };
        if (wtask) {
            const float scale = Dec::scale_value(a, term), bias = Dec::bias_value(a, term);
            u32x4* dst = reinterpret_cast<u32x4*>(ws + line * kFoldLdw + chunk * 16);
            if constexpr (Dec::kBits == 4) {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const uint32_t ev = wreg[v] & 0x0f0f0f0fu, od = (wreg[v] >> 4) & 0x0f0f0f0fu;
                    float c[8];
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        c[2 * b] = fmaf(static_cast<float>((ev >> (8 * b)) & 0xffu), scale, bias);
                        c[2 * b + 1] = fmaf(static_cast<float>((od >> (8 * b)) & 0xffu), scale, bias);
                    }
                    dst[v] = u32x4{pack(c[0], c[1]), pack(c[2], c[3]), pack(c[4], c[5]), pack(c[6], c[7])};
                }
            } else {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    uint32_t d[4];
#pragma unroll
                    for (int h = 0; h < 4; ++h) {
                        const int t = v * 8 + h * 2;
                        const float w0 = fmaf(static_cast<float>(piece_bits<Dec::kBits>(wreg, t)), scale, bias);
                        const float w1 = fmaf(static_cast<float>(piece_bits<Dec::kBits>(wreg, t + 1)), scale, bias);
                        d[h] = pack(w0, w1);
                    }
                    dst[v] = u32x4{d[0], d[1], d[2], d[3]};
                }
            }
        }
        u32x4* xdst = reinterpret_cast<u32x4*>(xs + xrow * kFoldLdw + xhalf * 16);
#pragma unroll
        for (int v = 0; v < 4; ++v) xdst[v] = xreg[v];
    };

    const int wm = wave & 7, wn = wave >> 3;
    const int rl = lane & 15, rh = lane >> 4;
    const bool live0 = wm * 32 < rows_left;
    const bool live1 = wm * 32 + 16 < rows_left;
    f32x8 acc[2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) acc[mt][nt] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
    fetch(0);
    for (int s = 0; s < stages; ++s) {
        stash();
        __syncthreads();
        if (s + 1 < stages) fetch(s + 1);  // in flight while this stage multiplies
        if (live0) {
#pragma unroll FOLD_UNROLL
            for (int t = 0; t < kFoldK / 16; ++t) {
                bf16x16 bf[4];
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) bf[nt] = frag16(ws + (wn * 64 + nt * 16 + rl) * kFoldLdw + t * 8);
                const bf16x16 a0 = frag16(xs + (wm * 32 + rl) * kFoldLdw + t * 8);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) acc[0][nt] = T::mma(a0, bf[nt], acc[0][nt]);
                if (live1) {
                    const bf16x16 a1 = frag16(xs + (wm * 32 + 16 + rl) * kFoldLdw + t * 8);
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) acc[1][nt] = T::mma(a1, bf[nt], acc[1][nt]);
                }
            }
        }
        __syncthreads();
    }
    Cols::finish(a, e, bx, m0, wm, wn, rl, rh, acc);
#else
    (void)a; (void)e;
    __builtin_trap();
#endif
}

// wmma_fold_tile's columns (the last column repeated past n) and epilogue (Epi::store a lane's sums) as Cols.
template <class Epi>
struct FoldCols {
    struct Extra {
        int n;
    };
    __device__ static int col(const Extra& e, int bx, int line) {
        const int c = bx * kGemmN + line;
        return c < e.n ? c : e.n - 1;
    }
    template <class Args>
    __device__ static void finish(const Args& a, const Extra&, int bx, int m0, int wm, int wn, int rl, int rh,
                                  const f32x8 (&acc)[2][4]) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = m0 + wm * 32 + mt * 16 + 2 * i + rh;
                if (row >= a.m) continue;
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int col = bx * kGemmN + wn * 64 + nt * 16 + rl;
                    if (col < a.n) Epi::store(a, row, col, acc[mt][nt][i]);
                }
            }
        }
    }
};

}  // namespace rocm
}  // namespace tf
