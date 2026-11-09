#pragma once

// The fold tile with its columns regrouped and an epilogue over a lane's columns: a hyper-connection's stream mix,
// a routed gate and up's activation (prefill, fold=on). The same bits as the fold tile and then the separate kernel.

#include <hip/hip_bfloat16.h>

#include "tiles/matrix_fold.hpp"
#include "tiles/plan.hpp"

namespace tf {
namespace rocm {

// tf_fn_hc_mix16's roundings (ops/common.hpp's float_to_bf16 and rounded): nearest even, NaN to 0x7fc0.
__device__ __forceinline__ uint16_t mix_bf16(float f) {
    if (f != f) return 0x7fc0;
    const uint32_t u = __float_as_uint(f);
    return static_cast<uint16_t>((u + (((u >> 16) & 1u) + 0x7fffu)) >> 16);
}
__device__ __forceinline__ float mix_f(uint16_t h) { return __uint_as_float(static_cast<uint32_t>(h) << 16); }

// wmma_fold_tile with its columns laid out by `Cols` and `Cols::finish` as the epilogue over a wave's sums: staged row
// `line` (local column wn 64 + nt 16 + rl) is weight row Cols::col(e, bx, line), so an epilogue that needs several
// columns of one row (the four streams of a dim, the gate and up of a unit) finds them in one lane's tiles.
template <class Dec, class Act, class T, class Cols>
__device__ __forceinline__ void wmma_fold_cols_tile(typename Dec::Args& a, const typename Cols::Extra& e) {
#if TF_DEVICE_WMMA_GFX11
    if (!Dec::take_item(a, blockIdx.z)) return;
    int bx, by;
    band_tile(bx, by);
    const int m0 = by * kGemmM;
    if (m0 >= a.m) return;
    __shared__ __attribute__((aligned(16))) uint32_t xs[kGemmM * kFoldLdw];
    __shared__ __attribute__((aligned(16))) uint32_t ws[kGemmN * kFoldLdw];

    const int tid = threadIdx.x;
    const int words_row = Dec::template row_words<int>(a);
    const int groups = a.k / a.group;
    const int stages = a.k / kFoldK;
    const int rows_left = a.m - m0;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int lane = tid & 31;

    // weights: a thread a column's 32-code chunk (chunk 0 or 1 of the stage); x: a thread half a row's 64 codes
    const int line = tid & (kGemmN - 1);
    const int chunk = tid >> 7;
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


// A hyper-connection's up projection (n = 4 dims) with tf_fn_hc_mix16 as the epilogue: block x takes dims 32 x.. of the
// four streams (nt is the stream), mixed[r, d] = mean over s of round(round(sigmoid(round(g))) * xs), the same bits.
struct HcMixCols {
    struct Extra {
        const uint16_t* xs;  // (m, 4 dims): the normed streams
        uint16_t* mixed;     // (m, dims)
        int dims;
    };
    __device__ static int col(const Extra& e, int bx, int line) {
        return ((line >> 4) & 3) * e.dims + bx * 32 + (line >> 6) * 16 + (line & 15);
    }
    template <class Args>
    __device__ static void finish(const Args& a, const Extra& e, int bx, int m0, int wm, int wn, int rl, int rh,
                                  const f32x8 (&acc)[2][4]) {
        const int dim = bx * 32 + wn * 16 + rl;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = m0 + wm * 32 + mt * 16 + 2 * i + rh;
                if (row >= a.m) continue;
                const long long base = static_cast<long long>(row) * 4 * e.dims + dim;
                float total = 0.f;
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const float g = mix_f(mix_bf16(acc[mt][s][i]));
                    const float gate = mix_f(mix_bf16(1.0f / (1.0f + expf(-g))));
                    total += mix_f(mix_bf16(gate * mix_f(e.xs[base + s * e.dims])));
                }
                e.mixed[static_cast<long long>(row) * e.dims + dim] = mix_bf16(total / 4.0f);
            }
        }
    }
};

// A hyper-connection norm's value of x at (row, k) from the streams it normed (tf_fn_hc_norm_rinv's h and rinv):
// bf16((h * rinv[row, stream]) * scale[k]), tf_fn_hc_norm_v4's x_out bytes. h (rows, streams * dims) bf16.
struct HcNormed {
    const uint16_t* h;
    const float* rinv;   // (rows, streams)
    const float* scale;  // (streams * dims)
    int streams, dims;
    // x at (row, stream s, dim d)
    __device__ float at(int row, int s, int d) const {
        const long long rs = static_cast<long long>(row) * streams + s;
        const float v = mix_f(h[rs * dims + d]);
        return mix_f(mix_bf16((v * rinv[rs]) * scale[s * dims + d]));
    }
};

// wmma_fold16_tile's x as HcNormed's values (the hc down projection reads the streams, not their normed copy). A
// thread's 32 values of a stage never cross a stream (dims % 64 == 0): one rinv, their 32 scales, loaded with the x.
struct HcNorm {
    using Extra = HcNormed;
    struct Pre {
        float r;
        float4 c[8];
    };
    __device__ static Pre load(const Extra& e, int row, int k) {
        Pre p;
        const int s = k / e.dims;
        p.r = e.rinv[row * e.streams + s];
#pragma unroll
        for (int i = 0; i < 8; ++i) p.c[i] = *reinterpret_cast<const float4*>(e.scale + k + 4 * i);
        return p;
    }
    __device__ static u32x4 map(const Extra&, const Pre& p, u32x4 v, int vi) {
        const float4 c0 = p.c[2 * vi], c1 = p.c[2 * vi + 1];
        const float sc[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float lo = mix_f(static_cast<uint16_t>(w[i] & 0xffffu)), hi = mix_f(static_cast<uint16_t>(w[i] >> 16));
            w[i] = static_cast<uint32_t>(mix_bf16((lo * p.r) * sc[2 * i])) |
                   static_cast<uint32_t>(mix_bf16((hi * p.r) * sc[2 * i + 1])) << 16;
        }
        return u32x4{w[0], w[1], w[2], w[3]};
    }
};

// HcMixCols with the normed streams read through HcNormed (no normed copy): the same bits.
struct HcMixNormCols {
    struct Extra {
        HcNormed xs;
        uint16_t* mixed;  // (m, dims)
        int dims;
    };
    __device__ static int col(const Extra& e, int bx, int line) {
        return ((line >> 4) & 3) * e.dims + bx * 32 + (line >> 6) * 16 + (line & 15);
    }
    template <class Args>
    __device__ static void finish(const Args& a, const Extra& e, int bx, int m0, int wm, int wn, int rl, int rh,
                                  const f32x8 (&acc)[2][4]) {
        const int dim = bx * 32 + wn * 16 + rl;
        const HcNormed& n = e.xs;
        float sc[4];  // the lane's dim of each stream's scale
#pragma unroll
        for (int s = 0; s < 4; ++s) sc[s] = n.scale[s * n.dims + dim];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = m0 + wm * 32 + mt * 16 + 2 * i + rh;
                if (row >= a.m) continue;
                const float4 r4 = *reinterpret_cast<const float4*>(n.rinv + static_cast<long long>(row) * 4);
                const float rv[4] = {r4.x, r4.y, r4.z, r4.w};
                const uint16_t* hr = n.h + static_cast<long long>(row) * 4 * n.dims + dim;
                float total = 0.f;
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const float g = mix_f(mix_bf16(acc[mt][s][i]));
                    const float gate = mix_f(mix_bf16(1.0f / (1.0f + expf(-g))));
                    const float xv = mix_f(mix_bf16((mix_f(hr[s * n.dims]) * rv[s]) * sc[s]));
                    total += mix_f(mix_bf16(gate * xv));
                }
                e.mixed[static_cast<long long>(row) * e.dims + dim] = mix_bf16(total / 4.0f);
            }
        }
    }
};

// A routed gate and up (n = 2 width) with moe_act_kernel as the epilogue: block x takes units 64 x.., nt 0 and 1 their
// gates, nt 2 and 3 the same units' ups; out[pair, c] = silu(g) * u (each clamped by limit above 0), the same bits.
struct PairCols {
    struct Extra {
        uint16_t* out;  // (pairs, width) bf16
        int width;
        float limit;
    };
    __device__ static int unit(int bx, int wn, int half, int rl) { return bx * 64 + wn * 32 + half * 16 + rl; }
    __device__ static int col(const Extra& e, int bx, int line) {
        const int nt = (line >> 4) & 3;
        return (nt >= 2 ? e.width : 0) + unit(bx, line >> 6, nt & 1, line & 15);
    }
    template <class Args>
    __device__ static void finish(const Args& a, const Extra& e, int bx, int m0, int wm, int wn, int rl, int rh,
                                  const f32x8 (&acc)[2][4]) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = m0 + wm * 32 + mt * 16 + 2 * i + rh;
                if (row >= a.m) continue;
                const long long orow = out_row(a, row);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    float g = acc[mt][h][i];
                    float u = acc[mt][2 + h][i];
                    if (e.limit > 0.f) {
                        g = fminf(g, e.limit);
                        u = fminf(fmaxf(u, -e.limit), e.limit);
                    }
                    const hip_bfloat16 v(g / (1.f + expf(-g)) * u);
                    e.out[orow * e.width + unit(bx, wn, h, rl)] = __builtin_bit_cast(uint16_t, v);
                }
            }
        }
    }
};

}  // namespace rocm
}  // namespace tf
