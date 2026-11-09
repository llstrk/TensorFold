#pragma once

// gfx11 WMMA prefill GEMM with the group terms folded into the weights as they are staged (opt-in, policy gemm=fold):
// each weight enters the products as the BF16 of code * scale + bias, so a tile keeps one accumulator, no per-group
// fold and no row sums; 64 codes a stage, the next stage's words and x held in registers while this one multiplies.
// Not wmma_gemm_tile's bits: the weights round to BF16 once (the exact-code tile rounds nothing but the products' sums).

#include "common/wmma.hpp"
#include "tiles/gemm.hpp"
#include "tiles/plan.hpp"

namespace tf {
namespace rocm {

#ifndef FOLD_UNROLL
#define FOLD_UNROLL 1  // a stage's k steps one at a time: unrolled, the fragments spill
#endif
constexpr int kFoldK = 64;            // codes a stage: two 32-code chunks a column
constexpr int kFoldLd = kFoldK + 8;   // BF16 a staged row: 16-byte aligned, rows spread over the banks
constexpr int kFoldLdw = kFoldLd / 2; // words a staged row

// Block 128 x 128 (the matrix tile's grid), 8 waves as 4 x 2: rows 32 wm.., columns 64 wn.. a wave.
template <class Dec, class Act, class T, class Epi>
__device__ __forceinline__ void wmma_fold_tile(typename Dec::Args& a) {
#if TF_DEVICE_WMMA_GFX11
    if (!Dec::take_item(a, blockIdx.z)) return;
    int bx, by;
    band_tile(bx, by);
    const int m0 = by * kGemmM;
    if (m0 >= a.m) return;
    __shared__ __attribute__((aligned(16))) uint32_t xs[kGemmM * kFoldLdw];
    __shared__ __attribute__((aligned(16))) uint32_t ws[kGemmN * kFoldLdw];

    const int tid = threadIdx.x;
    const int n0 = bx * kGemmN;
    const int words_row = Dec::template row_words<int>(a);
    const int groups = a.k / a.group;
    const int stages = a.k / kFoldK;
    const int rows_left = a.m - m0;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int lane = tid & 31;

    // weights: a thread a column's 32-code chunk (chunk 0 or 1 of the stage); x: a thread half a row's 64 codes
    const int line = tid & (kGemmN - 1);
    const int chunk = tid >> 7;
    const int col_s = n0 + line < a.n ? n0 + line : a.n - 1;
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
        Dec::template load<true>(wsrc + c * Dec::kWords, wreg);
        term = Dec::term(a, tbase + (c * 32) / a.group);
        if (xlive) {
#pragma unroll
            for (int v = 0; v < 4; ++v) xreg[v] = xsrc[st * 8 + xhalf * 4 + v];
        } else {
#pragma unroll
            for (int v = 0; v < 4; ++v) xreg[v] = u32x4{0u, 0u, 0u, 0u};
        }
    };
    auto stash = [&]() {
        const float scale = Dec::scale_value(a, term), bias = Dec::bias_value(a, term);
        u32x4* dst = reinterpret_cast<u32x4*>(ws + line * kFoldLdw + chunk * 16);
        auto pack = [](float w0, float w1) {  // BF16 rounded half up: + 0x8000, then one permute takes both halves
            return __builtin_amdgcn_perm(__float_as_uint(w1) + 0x8000u, __float_as_uint(w0) + 0x8000u, 0x07060302u);
        };
        if constexpr (Dec::kBits == 4) {
            // a word's 8 codes as two words of 4 bytes: each byte converts to float in one instruction
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
#ifdef FOLD_NO_DECODE
                d[h] = wreg[(v * 4 + h) % Dec::kWords] ^ __float_as_uint(scale);
#else
                const float w0 = fmaf(static_cast<float>(piece_bits<Dec::kBits>(wreg, t)), scale, bias);
                const float w1 = fmaf(static_cast<float>(piece_bits<Dec::kBits>(wreg, t + 1)), scale, bias);
                d[h] = pack(w0, w1);
#endif
            }
            dst[v] = u32x4{d[0], d[1], d[2], d[3]};
        }
        }
        u32x4* xdst = reinterpret_cast<u32x4*>(xs + xrow * kFoldLdw + xhalf * 16);
#pragma unroll
        for (int v = 0; v < 4; ++v) xdst[v] = xreg[v];
    };

    const int wm = wave & 3, wn = wave >> 2;
    const int rl = lane & 15, rh = lane >> 4;
    const bool live0 = wm * 32 < rows_left && n0 + wn * 64 < a.n;  // a wave past the columns (n = 320, 48) multiplies nothing
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
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = m0 + wm * 32 + mt * 16 + 2 * i + rh;
            if (row >= a.m) continue;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int col = n0 + wn * 64 + nt * 16 + rl;
                if (col < a.n) Epi::store(a, row, col, acc[mt][nt][i]);
            }
        }
    }
#else
    (void)a;
    __builtin_trap();
#endif
}

// x as it is read (wmma_fold16_tile's Norm): staged as loaded. A Norm's `load(e, row, k)` (issued with the stage's x,
// k its first of the thread's 32 values) gives what `map(e, pre, v, i)` takes to turn values k + 8 i.. into the product's.
struct NoNorm {
    struct Extra {};
    struct Pre {};
    __device__ static Pre load(const Extra&, int, int) { return {}; }
    __device__ static u32x4 map(const Extra&, const Pre&, u32x4 v, int) { return v; }
};

// wmma_fold_tile on a dense weight's BF16 copy (policy fold=bf16, the copy made by affine_fold_decode): the same
// stages and k order, the staged weights read as they are, so the same bits. `Norm` turns the x it stages into what the
// product takes (NoNorm: as they are), its loads in flight with the stage's x.
// Block 128 x 128 (the matrix tile's grid), 8 waves as 4 x 2: rows 32 wm.., columns 64 wn.. a wave.
template <class Dec, class Act, class T, class Epi, class Norm = NoNorm>
__device__ __forceinline__ void wmma_fold16_tile(typename Dec::Args& a, const uint16_t* w16,
                                                 const typename Norm::Extra& ne = {}) {
#if TF_DEVICE_WMMA_GFX11
    if (!Dec::take_item(a, blockIdx.z)) return;
    int bx, by;
    band_tile(bx, by);
    const int m0 = by * kGemmM;
    if (m0 >= a.m) return;
    __shared__ __attribute__((aligned(16))) uint32_t xs[kGemmM * kFoldLdw];
    __shared__ __attribute__((aligned(16))) uint32_t ws[kGemmN * kFoldLdw];

    const int tid = threadIdx.x;
    const int n0 = bx * kGemmN;
    const int words_row = Dec::template row_words<int>(a);
    const int groups = a.k / a.group;
    const int stages = a.k / kFoldK;
    const int rows_left = a.m - m0;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int lane = tid & 31;

    // weights: a thread a column's 32-code chunk (chunk 0 or 1 of the stage); x: a thread half a row's 64 codes
    const int line = tid & (kGemmN - 1);
    const int chunk = tid >> 7;
    const int col_s = n0 + line < a.n ? n0 + line : a.n - 1;
    // the weights already BF16 (n, k): a column's 32-code chunk is 64 contiguous bytes
    const u32x4* wsrc = reinterpret_cast<const u32x4*>(w16 + static_cast<long long>(col_s) * a.k);
    (void)words_row; (void)groups;
    const int xrow = tid >> 1, xhalf = tid & 1;
    const bool xlive = xrow < rows_left;
    const int row_s = xlive ? m0 + xrow : m0;
    const u32x4* xsrc = reinterpret_cast<const u32x4*>(Act::row(a, row_s));
    u32x4 wreg[4];
    u32x4 xreg[4];
    typename Norm::Pre pre;
    auto fetch = [&](int st) {
        const int c = st * 2 + chunk;  // the column's 32-code chunk
        pre = Norm::load(ne, row_s, st * kFoldK + xhalf * 32);
#pragma unroll
        for (int v = 0; v < 4; ++v) wreg[v] = wsrc[c * 4 + v];
        if (xlive) {
#pragma unroll
            for (int v = 0; v < 4; ++v) xreg[v] = xsrc[st * 8 + xhalf * 4 + v];
        } else {
#pragma unroll
            for (int v = 0; v < 4; ++v) xreg[v] = u32x4{0u, 0u, 0u, 0u};
        }
    };
    auto stash = [&]() {
        u32x4* dst = reinterpret_cast<u32x4*>(ws + line * kFoldLdw + chunk * 16);
#pragma unroll
        for (int v = 0; v < 4; ++v) dst[v] = wreg[v];
        u32x4* xdst = reinterpret_cast<u32x4*>(xs + xrow * kFoldLdw + xhalf * 16);
#pragma unroll
        for (int v = 0; v < 4; ++v) xdst[v] = Norm::map(ne, pre, xreg[v], v);
    };

    const int wm = wave & 3, wn = wave >> 2;
    const int rl = lane & 15, rh = lane >> 4;
    const bool live0 = wm * 32 < rows_left && n0 + wn * 64 < a.n;  // a wave past the columns (n = 320, 48) multiplies nothing
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
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = m0 + wm * 32 + mt * 16 + 2 * i + rh;
            if (row >= a.m) continue;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int col = n0 + wn * 64 + nt * 16 + rl;
                if (col < a.n) Epi::store(a, row, col, acc[mt][nt][i]);
            }
        }
    }
#else
    (void)a;
    __builtin_trap();
#endif
}


}  // namespace rocm
}  // namespace tf
