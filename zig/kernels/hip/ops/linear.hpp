#pragma once

#include "common.hpp"

// The linear attention's prefill: conv, gate and gated norm.

namespace {

// Conv prefill: window [state | x], taps summed in order, then silu; row block 0 writes new_state, never `state`.
constexpr int kConvRows = 16;   // rows a conv_prefill thread slides over
constexpr int kConvTaps = 8;    // most taps it holds

constexpr int kConvNormHead = 128;  // the q and k heads a normed conv's block holds two of

// With key_width, channels [0, kw), [kw, 2 kw) and the rest go to out, k_out and v_out, each its own contiguous rows.
// kNorm: q and k RMS-normed over 128-wide heads with tf_rms_rows' arithmetic (key_width and channels % 256 == 0).
template <bool kNorm>
__device__ __forceinline__ void conv_prefill_body(const void* x, int kind, const float* weight, const float* state,
                                                  float* out, float* new_state, int len, int channels, int kernel,
                                                  float* k_out, float* v_out, int key_width, const float* q_weight,
                                                  const float* k_weight, float eps) {
    // a thread one channel over kConvRows rows, the window sliding in registers (each input read once)
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float* dst = out;
    int pitch = channels, col = c, part = 2;
    if (key_width > 0) {
        part = c < key_width ? 0 : c < 2 * key_width ? 1 : 2;
        dst = part == 0 ? out : part == 1 ? k_out : v_out;
        pitch = part == 2 ? channels - 2 * key_width : key_width;
        col = c - part * key_width;
    }
    const bool staged = kNorm && part < 2;  // a whole block: q or k rows go through the tile to their norm
    const int r0 = blockIdx.y * kConvRows;
    const int kept = kernel - 1;
    auto window = [&](int t) -> float {
        if (t < kept) return state ? state[static_cast<long long>(t) * channels + c] : 0.0f;
        return load(x, kind, static_cast<long long>(t - kept) * channels + c);
    };
    if (blockIdx.y == 0)
        for (int t = 0; t < kept; ++t) new_state[static_cast<long long>(t) * channels + c] = window(len + t);
    float w[kConvTaps], win[kConvTaps];
    for (int tap = 0; tap < kernel; ++tap) w[tap] = weight[static_cast<long long>(c) * kernel + tap];
    for (int tap = 0; tap + 1 < kernel; ++tap) win[tap] = window(r0 + tap);
    const int r1 = r0 + kConvRows < len ? r0 + kConvRows : len;
    float (*tile)[256] = nullptr;  // kNorm: the block's rows of q or k before their norm
    if constexpr (kNorm) {
        __shared__ float rows[kConvRows][256];
        tile = rows;
    }
    for (int r = r0; r < r1; ++r) {
        win[kernel - 1] = window(r + kernel - 1);
        float acc = 0.0f;
        for (int tap = 0; tap < kernel; ++tap) {
            float p = win[tap] * w[tap];
            acc = acc + p;
        }
        if (staged) tile[r - r0][threadIdx.x] = silu(acc);
        else dst[static_cast<long long>(r) * pitch + col] = silu(acc);
        for (int tap = 0; tap + 1 < kernel; ++tap) win[tap] = win[tap + 1];
    }
    if constexpr (kNorm) {
        if (!staged) return;
        __syncthreads();
        // a wave a (row, head): a lane's values L + 32 j in order, the xor tree, rsqrt and the weight (tf_rms_rows)
        const float* nw = part == 0 ? q_weight : k_weight;
        const int lane = threadIdx.x & 31, col0 = col - threadIdx.x;
        for (int task = threadIdx.x >> 5; task < kConvRows * 2; task += blockDim.x >> 5) {
            const int rr = task >> 1, head = task & 1;
            if (r0 + rr >= len) break;
            float v[kConvNormHead / 32];
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < kConvNormHead / 32; ++j) {
                v[j] = tile[rr][head * kConvNormHead + lane + 32 * j];
                sum += v[j] * v[j];
            }
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
            const float inv = rsqrtf(__shfl(sum, 0, 32) / static_cast<float>(kConvNormHead) + eps);
            float* row = dst + static_cast<long long>(r0 + rr) * pitch + col0 + head * kConvNormHead;
#pragma unroll
            for (int j = 0; j < kConvNormHead / 32; ++j) {
                float value = v[j] * inv;
                if (nw != nullptr) value *= nw[lane + 32 * j];
                row[lane + 32 * j] = value;
            }
        }
    }
}

extern "C" __global__ void tf_conv_prefill(const void* x, int kind, const float* weight, const float* state, float* out,
                                    float* new_state, int len, int channels, int kernel, float* k_out, float* v_out,
                                    int key_width) {
    conv_prefill_body<false>(x, kind, weight, state, out, new_state, len, channels, kernel, k_out, v_out, key_width,
                             nullptr, nullptr, 0.f);
}

// tf_conv_prefill whose q and k leave as tf_rms_rows would norm them after it: no fp32 pass over them.
extern "C" __global__ void __launch_bounds__(256)
tf_conv_prefill_qknorm(const void* x, int kind, const float* weight, const float* state, float* out, float* new_state,
                       int len, int channels, int kernel, float* k_out, float* v_out, int key_width,
                       const float* q_weight, const float* k_weight, float eps) {
    conv_prefill_body<true>(x, kind, weight, state, out, new_state, len, channels, kernel, k_out, v_out, key_width,
                            q_weight, k_weight, eps);
}

// _gate_beta: beta = sigmoid(b), gate = exp(-exp(a_log) * softplus(a + dt_bias)) with torch's softplus.
extern "C" __global__ void tf_gdn_gate_prefill(const void* a, const void* b, int kind, const float* a_log,
                                        const float* dt_bias, float* gate, float* beta, int count, int heads) {
    long long i = gid();
    if (i >= static_cast<long long>(count) * heads) return;
    int h = static_cast<int>(i % heads);
    beta[i] = sigmoid(load(b, kind, i));
    float x = load(a, kind, i) + dt_bias[h];
    float xb = x * 1.0f;
    float sp = xb > 20.0f ? x : log1pf(expf(xb)) / 1.0f;
    float e = -expf(a_log[h]);
    gate[i] = expf(e * sp);
}

// out = y * silu(z) widened, rounded to kind: the linear-attention output before its projection.
extern "C" __global__ void tf_gnorm_silu(const float* y, const void* z, void* out, int kind, long long n,
                                         int sigmoid_gate) {
    long long i = gid();
    if (i >= n) return;
    const float zv = load(z, kind, i);
    float s = rounded(sigmoid_gate ? sigmoid(zv) : silu(zv), kind);
    store(out, kind, i, y[i] * s);
}

}  // namespace
