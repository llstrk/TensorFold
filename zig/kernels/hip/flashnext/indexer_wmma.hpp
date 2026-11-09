#pragma once

#include "../ops/common.hpp"
#include "../common/arch.hpp"

// tf_fn_idx_scores on gfx11's matrix cores (opt-in, policy sparse=wmma): q (rounded to BF16) against BF16 pooled
// keys of 128 dims, 16 rows by 64 blocks a workgroup, the relu'd head sums in fp32. Not the scalar kernel's bits.

namespace {

typedef __bf16 iw_bf16x16 __attribute__((ext_vector_type(16)));
typedef float iw_f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned iw_u32x8 __attribute__((ext_vector_type(8)));

// Workgroup tiles of 64 blocks a workgroup takes (each wave 16 of a tile): q staged once for kIwJ tiles.
constexpr int kIwJ = 4;

// Block (x, y), 4 waves: blocks 256 x + 64 j + 16 w .. (j < kIwJ) of rows 16 y ..; writes only what tf_fn_idx_scores
// writes. A tile past every row's complete blocks (or of rows at most `top` complete) is skipped whole: it writes nothing.
extern "C" __global__ void __launch_bounds__(128)
tf_fn_idx_scores_wmma(const float* q, const void* pooled, const int* complete, float* scores, int rows, int heads,
                      int blocks, long long stride, int top) {
#if TF_DEVICE_WMMA_GFX11
    constexpr int kDims = 128;
    __shared__ __attribute__((aligned(16))) uint16_t qs[8][16][kDims + 8];
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5, rl = lane & 15, rh = lane >> 4;
    const int r0 = blockIdx.y * 16, bx0 = blockIdx.x * 64 * kIwJ;
    // the most blocks a row of the 16 writes (complete, where above top): the same in every lane
    int need = 0;
    if (r0 + rl < rows) {
        const int c = complete[r0 + rl];
        need = c > top ? c : 0;
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) need = max(need, __shfl_xor(need, o, 32));
    if (bx0 >= need) return;
    // q as BF16, four dims a thread a step, eight steps' loads in flight before their stores
    const int n4 = heads * 16 * kDims / 4;
    for (int base = 0; base < n4; base += 128 * 8) {
        float4 v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int i = base + u * 128 + t;
            const int h = i / (16 * kDims / 4), rr = (i / (kDims / 4)) % 16, d = (i % (kDims / 4)) * 4;
            const int r = r0 + rr;
            v[u] = i < n4 && r < rows ? *reinterpret_cast<const float4*>(q + (static_cast<long long>(r) * heads + h) * kDims + d)
                                      : float4{0.f, 0.f, 0.f, 0.f};
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int i = base + u * 128 + t;
            if (i >= n4) break;
            const int h = i / (16 * kDims / 4), rr = (i / (kDims / 4)) % 16, d = (i % (kDims / 4)) * 4;
            const uint32_t lo = float_to_bf16(v[u].x) | (static_cast<uint32_t>(float_to_bf16(v[u].y)) << 16);
            const uint32_t hi = float_to_bf16(v[u].z) | (static_cast<uint32_t>(float_to_bf16(v[u].w)) << 16);
            *reinterpret_cast<uint2*>(&qs[h][rr][d]) = uint2{lo, hi};
        }
    }
    __syncthreads();
    const float norm = sqrtf(static_cast<float>(kDims));
    for (int j = 0; j < kIwJ; ++j) {
        if (bx0 + j * 64 >= need) break;
        const int b = bx0 + j * 64 + wave * 16 + rl;
        const uint16_t* key = static_cast<const uint16_t*>(pooled) + static_cast<long long>(min(b, blocks - 1)) * kDims;
        iw_bf16x16 kf[kDims / 16];
#pragma unroll
        for (int c = 0; c < kDims / 16; ++c) {
            const uint4 lo = *reinterpret_cast<const uint4*>(key + c * 16);
            const uint4 hi = *reinterpret_cast<const uint4*>(key + c * 16 + 8);
            const iw_u32x8 w = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
            kf[c] = __builtin_bit_cast(iw_bf16x16, w);
        }
        iw_f32x8 total = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int h = 0; h < heads; ++h) {
            iw_f32x8 s = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int c = 0; c < kDims / 16; ++c) {
                const uint4 lo = *reinterpret_cast<const uint4*>(&qs[h][rl][c * 16]);
                const uint4 hi = *reinterpret_cast<const uint4*>(&qs[h][rl][c * 16 + 8]);
                const iw_u32x8 w = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__builtin_bit_cast(iw_bf16x16, w), kf[c], s);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) total[i] += fmaxf(s[i], 0.f);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = r0 + 2 * i + rh;
            if (r >= rows || b >= blocks) continue;
            if (complete[r] <= top || b >= complete[r]) continue;
            scores[r * stride + b] = total[i] / norm;
        }
    }
#else
    (void)q; (void)pooled; (void)complete; (void)scores; (void)rows; (void)heads; (void)blocks; (void)stride; (void)top;
#endif
}

}  // namespace
