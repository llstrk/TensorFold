#pragma once

// The router's prompt logits on gfx11's matrix cores (opt-in, policy fold=on): BF16 x against the BF16 rows the fp32
// rows widen (so the same products), summed by WMMA in fp32. Not tf_router_tile's bits: another summation order.

#include "../common/arch.hpp"

namespace {

typedef __bf16 rw_bf16x16 __attribute__((ext_vector_type(16)));
typedef float rw_f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned rw_u32x8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ rw_bf16x16 rw_frag(const uint16_t* p) {
    const uint4 lo = *reinterpret_cast<const uint4*>(p);
    const uint4 hi = *reinterpret_cast<const uint4*>(p + 8);
    return __builtin_bit_cast(rw_bf16x16, rw_u32x8{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w});
}

constexpr int kRwM = 128, kRwN = 128, kRwK = 64, kRwLd = kRwK + 8;  // a staged row: 16-byte aligned, banks spread

// Block (x, y), 8 waves as 4 x 2: rows 128 y + 32 wm .., experts 128 x + 64 wn ..; 64 of d staged at a time through
// LDS, the next stage held in registers while this one multiplies (d % 64 == 0).
extern "C" __global__ void __launch_bounds__(256)
tf_router_wmma_bf16(const void* x, const void* rows16, float* logits, int r, int d, int e) {
#if TF_DEVICE_WMMA_GFX11
    __shared__ __attribute__((aligned(16))) uint16_t xs[kRwM * kRwLd];
    __shared__ __attribute__((aligned(16))) uint16_t ws[kRwN * kRwLd];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, rl = lane & 15, rh = lane >> 4;
    const int m0 = blockIdx.y * kRwM, n0 = blockIdx.x * kRwN;
    const uint16_t* xg = static_cast<const uint16_t*>(x);
    const uint16_t* wg = static_cast<const uint16_t*>(rows16);
    // loads: 8 threads a row's 64 values (16 bytes each), 32 rows a pass, 4 passes
    const int lr = tid >> 3, lq = tid & 7;
    const uint4* xp[4];
    const uint4* wp[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int row = min(m0 + p * 32 + lr, r - 1), ex = min(n0 + p * 32 + lr, e - 1);
        xp[p] = reinterpret_cast<const uint4*>(xg + static_cast<long long>(row) * d) + lq;
        wp[p] = reinterpret_cast<const uint4*>(wg + static_cast<long long>(ex) * d) + lq;
    }
    uint4 xr[4], wr[4];
    auto fetch = [&](int st) {
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            xr[p] = xp[p][st * (kRwK / 8)];
            wr[p] = wp[p][st * (kRwK / 8)];
        }
    };
    auto stash = [&]() {
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            *reinterpret_cast<uint4*>(&xs[(p * 32 + lr) * kRwLd + lq * 8]) = xr[p];
            *reinterpret_cast<uint4*>(&ws[(p * 32 + lr) * kRwLd + lq * 8]) = wr[p];
        }
    };
    const int wm = wave & 3, wn = wave >> 2;
    rw_f32x8 acc[2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) acc[mt][nt] = rw_f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const int stages = d / kRwK;
    fetch(0);
    for (int s = 0; s < stages; ++s) {
        stash();
        __syncthreads();
        if (s + 1 < stages) fetch(s + 1);
#pragma unroll
        for (int t = 0; t < kRwK / 16; ++t) {
            rw_bf16x16 b[4];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) b[nt] = rw_frag(&ws[(wn * 64 + nt * 16 + rl) * kRwLd + t * 16]);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const rw_bf16x16 a = rw_frag(&xs[(wm * 32 + mt * 16 + rl) * kRwLd + t * 16]);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b[nt], acc[mt][nt]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = m0 + wm * 32 + mt * 16 + 2 * i + rh;
            if (row >= r) continue;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int ex = n0 + wn * 64 + nt * 16 + rl;
                if (ex < e) logits[static_cast<long long>(row) * e + ex] = acc[mt][nt][i];
            }
        }
#endif
}

}  // namespace
