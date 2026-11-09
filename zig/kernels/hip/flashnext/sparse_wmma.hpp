#pragma once

#include "../ops/common.hpp"
#include "../common/arch.hpp"

// sparse_parts on gfx11's matrix cores (opt-in, policy sparse=wmma): a kv head's (up to 12) query heads as the rows of a
// WMMA tile, scores and values as BF16 products with fp32 sums. Two passes over a part's keys: every score first (a
// wave 16 keys of each 64, no barrier), the exact max a head, then p and the values 32 keys a tile, so no running max
// and no rescale. Under 32 KB of LDS: four blocks a WGP. Not sparse_parts' bits: q*scale and p round to BF16 before
// each product, the sums in another order. A part takes at most 320 keys (the launch's parts: a row's / 320 or more).

namespace {

template <int MASK> __device__ __forceinline__ float sw_xmask(float x) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), 0x160 | MASK, 0xf, 0xf, false));
}
__device__ __forceinline__ float sw_rowmax(float x) {
    x = fmaxf(x, sw_xmask<1>(x)); x = fmaxf(x, sw_xmask<2>(x)); x = fmaxf(x, sw_xmask<4>(x)); return fmaxf(x, sw_xmask<8>(x));
}

typedef __bf16 sw_bf16x16 __attribute__((ext_vector_type(16)));
typedef float sw_f32x8 __attribute__((ext_vector_type(8)));
constexpr int kSwWaves = 4, kSwDims = 256, kSwHeads = 12;
#ifndef SW_TILE
#define SW_TILE 32
#endif
#ifndef SW_MAXKEYS
#define SW_MAXKEYS 320
#endif
constexpr int kSwTile = SW_TILE;                   // keys a value tile
constexpr int kSwMax = SW_MAXKEYS;                 // a part's keys at most (a multiple of 64)
// S rows are 16 banks apart (kSwMax + 16 floats): conflict-free pass 1 stores and p reads
constexpr int kSwQRow = kSwDims + 8, kSwVRow = kSwTile + 8, kSwSRow = kSwMax + 16, kSwPRow = kSwMax + 8;

__device__ __forceinline__ sw_bf16x16 sw_frag(const uint16_t* p) {
    const uint4 lo = *reinterpret_cast<const uint4*>(p);
    const uint4 hi = *reinterpret_cast<const uint4*>(p + 8);
    typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
    return __builtin_bit_cast(sw_bf16x16, u32x8{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w});
}

// kMerge: the last of a (kv head, row)'s parts to finish (counted in done, which it sets back to 0) merges its heads'
// parts as tf_fn_sparse_merge does (in part order, the same bits) into out: no merge launch.
template <bool kMerge = false>
__device__ void sparse_wmma(const float* q, const void* k, const void* v, const int* ids, long long id_stride, const int* count,
                          const int* sparse, float* parts_o, float* parts_ms, int heads, int kv_heads, int dims, long long cap,
                          float scale, int parts, float* out = nullptr, int* done = nullptr) {
#if TF_DEVICE_WMMA_GFX11
    constexpr int kA = kSwHeads * kSwSRow * 4 > kSwDims * kSwVRow * 2 ? kSwHeads * kSwSRow * 4 : kSwDims * kSwVRow * 2;
    constexpr int kB = kSwHeads * kSwPRow * 2 > kSwHeads * kSwQRow * 2 ? kSwHeads * kSwPRow * 2 : kSwHeads * kSwQRow * 2;
    __shared__ __attribute__((aligned(16))) unsigned char bufa[kA];  // the scores, then vt
    __shared__ __attribute__((aligned(16))) unsigned char bufb[kB];  // q, then p
    __shared__ int kall[kSwMax];
    __shared__ float wmax[kSwWaves][16], wl[16];
    float (*S)[kSwSRow] = reinterpret_cast<float (*)[kSwSRow]>(bufa);
    uint16_t (*vt)[kSwVRow] = reinterpret_cast<uint16_t (*)[kSwVRow]>(bufa);
    uint16_t (*qs)[kSwQRow] = reinterpret_cast<uint16_t (*)[kSwQRow]>(bufb);
    uint16_t (*ps)[kSwPRow] = reinterpret_cast<uint16_t (*)[kSwPRow]>(bufb);
    const int g = blockIdx.x, r = blockIdx.y, part = blockIdx.z;
    const int group = heads / kv_heads;
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5, rl = lane & 15, rh = lane >> 4;
    const int ra = rl < kSwHeads ? rl : kSwHeads - 1;  // A rows past the heads read the last head's (and are dropped)
    const int n = count[r];
    const bool listed = ids != nullptr && sparse != nullptr && sparse[r] != 0;
    const int lo = static_cast<int>(static_cast<long long>(part) * n / parts);
    const int hi = static_cast<int>(static_cast<long long>(part + 1) * n / parts);
    const int len = hi - lo;
    const int chunks = (len + 63) / 64;            // pass 1: 64 keys a step, 16 a wave
    const int tiles = (len + kSwTile - 1) / kSwTile;
    const long long head_base = static_cast<long long>(g) * cap * kSwDims;
    const uint16_t* k16 = static_cast<const uint16_t*>(k);
    const uint16_t* v16 = static_cast<const uint16_t*>(v);
    {
        // q * scale as BF16, four dims a thread a step: every load in flight before the first store
        constexpr int kQ4 = kSwHeads * kSwDims / 4, kQIt = kQ4 / (kSwWaves * 32);
        static_assert(kQ4 % (kSwWaves * 32) == 0, "q staging steps");
        const float* qrow = q + (static_cast<long long>(r) * heads + g * group) * dims;
        float4 qv[kQIt];
#pragma unroll
        for (int it = 0; it < kQIt; ++it) {
            const int i = t + it * kSwWaves * 32, h = i / (kSwDims / 4), d = (i % (kSwDims / 4)) * 4;
            qv[it] = h < group ? *reinterpret_cast<const float4*>(qrow + static_cast<long long>(h) * dims + d) : float4{0.f, 0.f, 0.f, 0.f};
        }
#pragma unroll
        for (int it = 0; it < kQIt; ++it) {
            const int i = t + it * kSwWaves * 32, h = i / (kSwDims / 4), d = (i % (kSwDims / 4)) * 4;
            const uint32_t lo = float_to_bf16(qv[it].x * scale) | (static_cast<uint32_t>(float_to_bf16(qv[it].y * scale)) << 16);
            const uint32_t hi = float_to_bf16(qv[it].z * scale) | (static_cast<uint32_t>(float_to_bf16(qv[it].w * scale)) << 16);
            *reinterpret_cast<uint2*>(&qs[h][d]) = uint2{lo, hi};
        }
    }
    for (int j = t; j < chunks * 64; j += kSwWaves * 32) {
        const int jj = j < len ? j : 0;  // padding keys repeat the part's first; their p is 0
        kall[j] = listed ? ids[r * id_stride + lo + jj] : lo + jj;
    }
    __syncthreads();
    // pass 1: the scores, 16 keys a wave a step, into S (head rows, key columns)
    float m[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) m[i] = -INFINITY;
    for (int c = 0; c < chunks; ++c) {
        const int kj = c * 64 + wave * 16 + rl;
        const uint16_t* krow = k16 + head_base + static_cast<long long>(kall[kj]) * kSwDims;
        sw_f32x8 s = sw_f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kc = 0; kc < kSwDims / 16; ++kc)
            s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(sw_frag(&qs[ra][kc * 16]), sw_frag(krow + kc * 16), s);
        const bool live = kj < len;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int h = 2 * i + rh;
            const float x = live ? s[i] : -INFINITY;
            if (h < kSwHeads) S[h][kj] = x;
            m[i] = fmaxf(m[i], x);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        m[i] = sw_rowmax(m[i]);
        if (rl == 0) wmax[wave][2 * i + rh] = m[i];
    }
    __syncthreads();  // S and wmax complete; q is spent
    // p: 8 threads a head (96 of the 128), a thread every 8th key
    if (t < kSwHeads * 8) {
        const int h = t >> 3, p8 = t & 7;
        float top = wmax[0][h];
#pragma unroll
        for (int w = 1; w < kSwWaves; ++w) top = fmaxf(top, wmax[w][h]);
        float sum = 0.f;
        for (int j = p8; j < tiles * kSwTile; j += 8) {
            const float x = j < len ? S[h][j] : -INFINITY;
            const uint16_t eb = float_to_bf16(x == -INFINITY ? 0.f : expf(x - top));
            ps[h][j] = eb;
            sum += bf16_to_float(eb);
        }
        sum += __shfl_xor(sum, 1, 32); sum += __shfl_xor(sum, 2, 32); sum += __shfl_xor(sum, 4, 32);
        if (p8 == 0) wl[h] = sum;
    }
    __syncthreads();  // p complete; S is spent
    // pass 2: the values a tile at a time; lanes 16 h.. a wave's 32 dims 32 h.., a lane two neighbouring keys
    sw_f32x8 acc[4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) acc[nt] = sw_f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    constexpr int KP = kSwTile / 16;              // key pairs a lane... keys a lane: 2 (tile 32) or 4 (tile 64)
    uint4 va[KP][4];
    const int vd0 = wave * 64 + rh * 32;          // this lane's 32 dims
    auto vfetch = [&](int c0) {
#pragma unroll
        for (int kp = 0; kp < KP; ++kp) {
            const long long key = kall[c0 * kSwTile + rl * KP + kp];
            const uint4* p = reinterpret_cast<const uint4*>(v16 + head_base + key * kSwDims + vd0);
#pragma unroll
            for (int c = 0; c < 4; ++c) va[kp][c] = p[c];
        }
    };
    if (tiles > 0) vfetch(0);
    for (int c0 = 0; c0 < tiles; ++c0) {
        // vt[d][j]: 32-bit words of two keys
#pragma unroll
        for (int kp = 0; kp < KP; kp += 2) {
            const int j = rl * KP + kp;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint4 a4 = va[kp][c], b4 = va[kp + 1][c];
                const uint32_t ea[4] = {a4.x, a4.y, a4.z, a4.w}, eb[4] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    const int d = vd0 + c * 8 + 2 * x;
                    *reinterpret_cast<uint32_t*>(&vt[d][j]) = (ea[x] & 0xffff) | (eb[x] << 16);
                    *reinterpret_cast<uint32_t*>(&vt[d + 1][j]) = (ea[x] >> 16) | (eb[x] & 0xffff0000u);
                }
            }
        }
        __syncthreads();  // vt complete
        if (c0 + 1 < tiles) vfetch(c0 + 1);
#pragma unroll
        for (int ks = 0; ks < kSwTile / 16; ++ks) {
            const sw_bf16x16 pa = sw_frag(&ps[ra][c0 * kSwTile + ks * 16]);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
                acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pa, sw_frag(&vt[wave * 64 + nt * 16 + rl][ks * 16]), acc[nt]);
        }
        __syncthreads();  // vt is rewritten by the next tile
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int h = 2 * i + rh;
            if (h >= group) continue;
            const long long out = (static_cast<long long>(r) * heads + g * group + h) * parts + part;
            parts_o[out * dims + wave * 64 + nt * 16 + rl] = acc[nt][i];
        }
    if (t < group) {
        const long long out = (static_cast<long long>(r) * heads + g * group + t) * parts + part;
        float top = wmax[0][t];
        for (int w = 1; w < kSwWaves; ++w) top = fmaxf(top, wmax[w][t]);
        parts_ms[out * 2] = len > 0 ? top : -INFINITY;
        parts_ms[out * 2 + 1] = len > 0 ? wl[t] : 0.f;
    }
    if constexpr (kMerge) {
        __shared__ int last;
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __syncthreads();
        if (t == 0) {
            int* cell = done + static_cast<long long>(r) * kv_heads + g;
            last = __hip_atomic_fetch_add(cell, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) == parts - 1;
            if (last) __hip_atomic_store(cell, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __syncthreads();
        if (!last) return;
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        // each head's part weights and total once (tf_fn_sparse_merge's values: the same expf of the same inputs, the
        // total in part order), then (head, 4 dims) items a thread, each sum in part order
        __shared__ float mw[kSwHeads][8];
        __shared__ float mt[kSwHeads];
        if (t < group) {
            const long long at = (static_cast<long long>(r) * heads + g * group + t) * parts;
            float top = -INFINITY;
            for (int p = 0; p < parts; ++p) top = fmaxf(top, parts_ms[(at + p) * 2]);
            float total = 0.f;
            for (int p = 0; p < parts; ++p) {
                const float mp = parts_ms[(at + p) * 2];
                const float w = mp == -INFINITY ? 0.f : expf(mp - top);
                total += parts_ms[(at + p) * 2 + 1] * w;
                mw[t][p] = w;
            }
            mt[t] = total;
        }
        __syncthreads();
        const int q4 = dims / 4;
        for (int it = t; it < group * q4; it += kSwWaves * 32) {
            const int hh = it / q4, d = (it % q4) * 4;
            const long long at = (static_cast<long long>(r) * heads + g * group + hh) * parts;
            float4 acc = {0.f, 0.f, 0.f, 0.f};
            for (int p = 0; p < parts; ++p) {
                const float w = mw[hh][p];
                const float4 x = *reinterpret_cast<const float4*>(parts_o + (at + p) * dims + d);
                acc.x += x.x * w; acc.y += x.y * w; acc.z += x.z * w; acc.w += x.w * w;
            }
            const float total = mt[hh];
            const float4 o = total > 0.f ? float4{acc.x / total, acc.y / total, acc.z / total, acc.w / total}
                                         : float4{0.f, 0.f, 0.f, 0.f};
            *reinterpret_cast<float4*>(out + (static_cast<long long>(r) * heads + g * group + hh) * dims + d) = o;
        }
    }
#endif
}

extern "C" __global__ void __launch_bounds__(kSwWaves * 32)
tf_fn_sparse_wmma_bf16(const float* q, const void* k, const void* v, const int* ids, long long id_stride, const int* count,
               const int* sparse, float* parts_o, float* parts_ms, int heads, int kv_heads, int dims, long long cap, float scale,
               int parts) {
    sparse_wmma(q, k, v, ids, id_stride, count, sparse, parts_o, parts_ms, heads, kv_heads, dims, cap, scale, parts);
}

// tf_fn_sparse_wmma_bf16 and then tf_fn_sparse_merge in one launch: `done` (rows, kv heads) zero, left zero.
extern "C" __global__ void __launch_bounds__(kSwWaves * 32)
tf_fn_sparse_wmma_merge_bf16(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
               const int* count, const int* sparse, float* parts_o, float* parts_ms, int heads, int kv_heads, int dims,
               long long cap, float scale, int parts, float* out, int* done) {
    sparse_wmma<true>(q, k, v, ids, id_stride, count, sparse, parts_o, parts_ms, heads, kv_heads, dims, cap, scale,
                      parts, out, done);
}
}  // namespace
