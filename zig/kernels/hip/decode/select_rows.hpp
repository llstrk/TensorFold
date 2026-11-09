#pragma once

// MoE picks for prompt rows, a wave a row: tf_select_decode's picks and weights without its plan. Included by
// decode.hip after select_body and round_bf16_f.

namespace {

// select_body's picks and weights for prompt rows (no plan), a wave a row: k rounds of the wave's best (value, then the
// lower index on a tie: select_body's rank order), so the same top k in the same order and the same weights.
constexpr int kSelectRowWaves = 8;

template <int PER>
__device__ void select_rows_body(const float* logits, int* pick, float* wts, int experts, int top_k, int rows) {
    __shared__ float top_v[kSelectRowWaves][32];
    __shared__ int top_i[kSelectRowWaves][32];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int row = blockIdx.x * kSelectRowWaves + wave;
    if (row >= rows) return;
    const int slots = top_k + 1;
    const float* l = logits + static_cast<long long>(row) * (experts + 1);
    float v[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int e = lane + 32 * j;
        v[j] = e < experts ? l[e] : 0.f;
    }
    uint32_t gone = 0;  // bit j: lane + 32 j is padding or picked
#pragma unroll
    for (int j = 0; j < PER; ++j)
        if (lane + 32 * j >= experts) gone |= 1u << j;
    for (int k = 0; k < top_k; ++k) {
        float bv = 0.f;
        int bi = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int e = lane + 32 * j;
            if (!((gone >> j) & 1u) && (bi == 0x7fffffff || v[j] > bv || (v[j] == bv && e < bi))) {
                bv = v[j];
                bi = e;
            }
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float ov = __shfl_xor(bv, mask, 32);
            const int oi = __shfl_xor(bi, mask, 32);
            if (oi != 0x7fffffff && (bi == 0x7fffffff || ov > bv || (ov == bv && oi < bi))) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            top_v[wave][k] = bv;
            top_i[wave][k] = bi;
        }
        if ((bi & 31) == lane) gone |= 1u << (bi >> 5);
    }
    if (lane != 0) return;
    const float top = top_v[wave][0];
    float ex[32];
    float total = 0.f;
    for (int k = 0; k < top_k; ++k) {
        ex[k] = expf(top_v[wave][k] - top);
        total += ex[k];
    }
    for (int k = 0; k < top_k; ++k) {
        pick[row * slots + k] = top_i[wave][k];
        wts[row * slots + k] = round_bf16_f(ex[k] / total);
    }
    const float gate = round_bf16_f(l[experts]);
    pick[row * slots + top_k] = experts;
    wts[row * slots + top_k] = round_bf16_f(1.f / (1.f + expf(-gate)));
}

}  // namespace

extern "C" __global__ void __launch_bounds__(32 * kSelectRowWaves)
    tf_select_rows16(const float* logits, int* pick, float* wts, int experts, int top_k, int rows) {
    select_rows_body<16>(logits, pick, wts, experts, top_k, rows);
}

extern "C" __global__ void __launch_bounds__(32 * kSelectRowWaves)
    tf_select_rows32(const float* logits, int* pick, float* wts, int experts, int top_k, int rows) {
    select_rows_body<32>(logits, pick, wts, experts, top_k, rows);
}
