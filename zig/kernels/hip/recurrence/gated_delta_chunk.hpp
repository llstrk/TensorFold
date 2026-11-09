#pragma once
// The gated delta rule's prompt scan in chunks of 32 on gfx11's matrix cores (policy delta=chunk, opt-in). Per chunk,
// c the running sum of log gates, b beta:
//   A = strict_lower(b_t e^(c_t - c_j) k_t.k_j), T = (I + A)^-1, Tb = T diag(b)
//   D = Tb (V - diag(e^c) K S^T)                       (the UT form's T diag(b e^c) K folded into Tb's operand)
//   y = diag(e^c) Q S^T + (lower(e^(c_t - c_j) q_t.k_j)) D
//   S = e^(c_n) S + D^T diag(e^(c_n - c_t)) K
// The state S (value rows r, key dims d) stays fp32 in registers as S^T tiles, wave w owning value rows 16w..: every
// product with S takes a wave's own columns. Products take FP16 operands with fp32 sums. Not the scan's bits.
// Two kernels: what needs no state runs first over every (chunk, value head) at once, so the walk through the chunks
// keeps only the products with the state.
//   tf_gdn_chunk_prep: block (chunk, value head), 256 threads: A, P, T and Tb, written as FP16 32 x 32 tiles,
//               and the chunk's running log-gate sums.
//   tf_gdn_chunk_walk: block (value head, batch), 512 threads: waves 0..7 the state's walk, waves
//               8..15 stage the next chunk's Q, K, K^T, Tb, P and sums into the other of two buffers meanwhile; one
//               block barrier a chunk.
#include <type_traits>
#include "../ops/common.hpp"
#include "../common/arch.hpp"

namespace {
namespace gdnc {

constexpr int C = 32, DK = 128, DV = 128, W = 8;
constexpr int LQ = DK + 8, LT = C + 8, LP = C;     // FP16 row pitches (P and Tb unpadded: two buffers fit 64 KB)
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef unsigned u8v __attribute__((ext_vector_type(8)));

__device__ __forceinline__ h16 frag(const _Float16* p) {
    const uint4 lo = *reinterpret_cast<const uint4*>(p), hi = *reinterpret_cast<const uint4*>(p + 8);
    return __builtin_bit_cast(h16, u8v{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w});
}
__device__ __forceinline__ f8 mma(h16 a, h16 b, f8 c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c); }
__device__ __forceinline__ f8 zero8() { return f8{0, 0, 0, 0, 0, 0, 0, 0}; }
__device__ __forceinline__ h16 colfrag(f8 acc, int rh) {
    h16 out;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float mine = acc[i];
        const float other = __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(
            __builtin_bit_cast(int, mine), __builtin_bit_cast(int, mine), 0x76543210, 0xfedcba98, false, false));
        out[2 * i] = (_Float16)(rh ? other : mine);
        out[2 * i + 1] = (_Float16)(rh ? mine : other);
    }
    return out;
}

// Q, K rows of chunk rows t0.. as FP16 (rows past the prompt zero), K^T too where kt is given; `threads` threads.
__device__ __forceinline__ void stage_qk(const float* q, const float* k, int t0, int n, int key_heads, int key_head,
                                         int tid, int threads, _Float16 (*qs)[LQ], _Float16 (*ks)[LQ],
                                         _Float16 (*kt)[LT]) {
#pragma unroll 1
    for (int i = tid; i < C * DK / 4; i += threads) {
        const int t = i / (DK / 4), d = (i % (DK / 4)) * 4;
        float4 kv = {0, 0, 0, 0}, qv = {0, 0, 0, 0};
        if (t < n) {
            const size_t src = (static_cast<size_t>(t0 + t) * key_heads + key_head) * DK + d;
            kv = *reinterpret_cast<const float4*>(k + src);
            qv = *reinterpret_cast<const float4*>(q + src);
        }
        ks[t][d] = (_Float16)kv.x; ks[t][d + 1] = (_Float16)kv.y; ks[t][d + 2] = (_Float16)kv.z; ks[t][d + 3] = (_Float16)kv.w;
        qs[t][d] = (_Float16)qv.x; qs[t][d + 1] = (_Float16)qv.y; qs[t][d + 2] = (_Float16)qv.z; qs[t][d + 3] = (_Float16)qv.w;
        if (kt != nullptr) {
            kt[d][t] = (_Float16)kv.x; kt[d + 1][t] = (_Float16)kv.y; kt[d + 2][t] = (_Float16)kv.z; kt[d + 3][t] = (_Float16)kv.w;
        }
    }
}

// Block (value head, chunk, batch), 256 threads (a chunk's heads together: they read the same rows). tbg, psg: (batch, chunks, value heads, 32, 32) FP16; cumg: (batch,
// chunks, value heads, 32).
extern "C" __global__ void __launch_bounds__(256)
tf_gdn_chunk_prep(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ gate,
           const float* __restrict__ beta, _Float16* __restrict__ tbg, _Float16* __restrict__ psg, float* __restrict__ cumg,
           int length, int key_heads, int value_heads) {
#if TF_DEVICE_WMMA_GFX11
    __shared__ __attribute__((aligned(16))) _Float16 qs[C][LQ];
    __shared__ __attribute__((aligned(16))) _Float16 ks[C][LQ];
    __shared__ float as[C][C + 1];
    __shared__ float ts[C][C + 1];
    __shared__ float cum[C], bs[C];
    const int head = blockIdx.x, c = blockIdx.y, batch = blockIdx.z, tid = threadIdx.x;
    const int chunks = gridDim.y;
    q += static_cast<size_t>(batch) * length * key_heads * DK;
    k += static_cast<size_t>(batch) * length * key_heads * DK;
    gate += static_cast<size_t>(batch) * length * value_heads;
    beta += static_cast<size_t>(batch) * length * value_heads;
    const size_t tile = ((static_cast<size_t>(batch) * chunks + c) * value_heads + head);
    const int key_head = head / (value_heads / key_heads);
    const int lane = tid & 31, wave = tid >> 5, rl = lane & 15, rh = lane >> 4;
    const int t0 = c * C, n = min(C, length - t0);
    stage_qk(q, k, t0, n, key_heads, key_head, tid, 256, qs, ks, nullptr);
    if (wave == 0) {
        const size_t src = static_cast<size_t>(t0 + lane) * value_heads + head;
        const float lg = lane < n ? logf(fmaxf(gate[src], 1e-30f)) : 0.f;
        float s = lg;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { const float u = __shfl_up(s, o, 32); if (lane >= o) s += u; }
        cum[lane] = s;
        cumg[tile * C + lane] = s;
        bs[lane] = lane < n ? beta[src] : 0.f;
    }
    __syncthreads();
    _Float16* ps = psg + tile * C * C;
    if (wave < 6) {
        const bool qk = wave >= 3;
        const int tl = wave % 3, ti = tl == 0 ? 0 : 1, tj = tl == 2 ? 1 : 0;
        const _Float16 (*a)[LQ] = qk ? qs : ks;
        f8 acc = zero8();
#pragma unroll
        for (int kk = 0; kk < DK / 16; ++kk) {
            acc = mma(frag(&a[ti * 16 + rl][kk * 16]), frag(&ks[tj * 16 + rl][kk * 16]), acc);
            __builtin_amdgcn_sched_barrier(0);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int t = ti * 16 + 2 * i + rh, j = tj * 16 + rl;
            const float dcy = __expf(cum[t] - cum[j]);
            if (!qk) as[t][j] = j < t ? bs[t] * dcy * acc[i] : 0.f;
            else ps[t * C + j] = (_Float16)(j <= t ? dcy * acc[i] : 0.f);
        }
    } else {
        for (int i = tid - 192; i < 16 * 16; i += 64) {
            const int t = i / 16, j = 16 + i % 16;
            as[t][j] = 0.f;
            ps[t * C + j] = (_Float16)0.f;
        }
    }
    __syncthreads();
    if (tid < C) {
        const int j = tid & 15, o = tid & 16;
        float tc[16];
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            float s = t == j ? 1.f : 0.f;
#pragma unroll
            for (int i = 0; i < t; ++i) s -= as[o + t][o + i] * tc[i];
            tc[t] = s;
            __builtin_amdgcn_sched_barrier(0);
        }
#pragma unroll
        for (int t = 0; t < 16; ++t) ts[o + t][o + j] = tc[t];
    }
    __syncthreads();
    {
        const int t = tid >> 4, j = tid & 15;
        float s = 0.f;
#pragma unroll
        for (int m = 0; m < 16; ++m) s += as[16 + t][m] * ts[m][j];
        ts[t][16 + j] = s;
    }
    __syncthreads();
    {
        const int t = tid >> 4, j = tid & 15;
        float s = 0.f;
#pragma unroll
        for (int m = 0; m < 16; ++m) s += ts[16 + t][16 + m] * ts[m][16 + j];
        ts[16 + t][j] = -s;
    }
    __syncthreads();
    _Float16* tb = tbg + tile * C * C;
    for (int i = tid; i < C * C; i += 256) {
        const int t = i / C, j = i % C;
        tb[i] = (_Float16)(j <= t ? ts[t][j] * bs[j] : 0.f);
    }
#else
    (void)q; (void)k; (void)gate; (void)beta; (void)tbg; (void)psg; (void)cumg; (void)length; (void)key_heads; (void)value_heads;
#endif
}

// A block barrier over LDS only: __syncthreads also waits for the wave's global stores (the y rows) and loads.
__device__ __forceinline__ void lds_barrier() {
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

struct Buf {
    _Float16 qs[C][LQ];
    _Float16 ks[C][LQ];
    _Float16 kt[DK][LT];
    _Float16 ps[C][LP];
    _Float16 tb[C][LP];
    float cum[C];
};

// Block (value head, batch), 512 threads; tbg, psg, cumg from tf_gdn_chunk_prep.
extern "C" __global__ void __launch_bounds__(512)
tf_gdn_chunk_walk(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
           const _Float16* __restrict__ tbg, const _Float16* __restrict__ psg, const float* __restrict__ cumg,
           float* __restrict__ state, float* __restrict__ y, int length, int key_heads, int value_heads) {
#if TF_DEVICE_WMMA_GFX11
    __shared__ __attribute__((aligned(16))) Buf bufs[2];
    const int head = blockIdx.x, batch = blockIdx.y, tid0 = threadIdx.x;
    const int chunks = (length + C - 1) / C;
    q += static_cast<size_t>(batch) * length * key_heads * DK;
    k += static_cast<size_t>(batch) * length * key_heads * DK;
    v += static_cast<size_t>(batch) * length * value_heads * DV;
    y += static_cast<size_t>(batch) * length * value_heads * DV;
    state += static_cast<size_t>(batch) * value_heads * DV * DK;
    const int key_head = head / (value_heads / key_heads);
    const bool producer = tid0 >= W * 32;
    // the producers' staging of chunk cc into buffer b
    auto stage = [&](int cc, Buf& b) {
        const int ptid = tid0 - W * 32;
        const int t0 = cc * C, n = min(C, length - t0);
        stage_qk(q, k, t0, n, key_heads, key_head, ptid, W * 32, b.qs, b.ks, b.kt);
        const size_t tile = (static_cast<size_t>(batch) * chunks + cc) * value_heads + head;
        // P and Tb: 2 x 1024 FP16, 8 a thread
        {
            const uint4* src = reinterpret_cast<const uint4*>((ptid < 128 ? psg : tbg) + tile * C * C);
            const int i = ptid & 127;  // 8 FP16 a thread: row i / 4, columns (i % 4) 8..
            const uint4 x = src[i];
            _Float16 (*dst)[LP] = ptid < 128 ? b.ps : b.tb;
            *reinterpret_cast<uint4*>(&dst[i / 4][(i % 4) * 8]) = x;
        }
        if (ptid < C) b.cum[ptid] = cumg[tile * C + ptid];
    };
    if (producer) {
        stage(0, bufs[0]);
        __syncthreads();
        for (int c = 0; c < chunks; ++c) {
            if (c + 1 < chunks) stage(c + 1, bufs[(c + 1) & 1]);
            lds_barrier();
        }
        return;
    }
    const int lane0 = tid0 & 31, wave = tid0 >> 5, rl0 = lane0 & 15, rh0 = lane0 >> 4;
    const int r0 = wave * 16 + rl0;
    f8 st[DK / 16];
#pragma unroll
    for (int dt = 0; dt < DK / 16; ++dt)
#pragma unroll
        for (int i = 0; i < 8; ++i) st[dt][i] = state[(static_cast<size_t>(head) * DV + r0) * DK + dt * 16 + 2 * i + rh0];
    __syncthreads();
    const int row_stride = value_heads * DV;
    // a lane's v rows of a chunk (accumulator layout)
    auto load_v = [&](int cc, float (&dst)[2][8], auto full) {
        const int t0 = cc * C, n = min(C, length - t0);
        int tid = tid0;
        asm volatile("" : "+v"(tid));
        const int rh = (tid >> 4) & 1, r = wave * 16 + (tid & 15);
        const float* vb = v + (static_cast<size_t>(t0) * value_heads + head) * DV + r;
        asm volatile("" : "+v"(vb));
#pragma unroll
        for (int ti = 0; ti < 2; ++ti)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int t = ti * 16 + 2 * i + rh;
                dst[ti][i] = !decltype(full)::value && t >= n ? 0.f : vb[t * row_stride];
            }
    };
    // a chunk of the walk; `full`: all 32 rows in the prompt (every chunk but maybe the last), so the v loads and y
    // stores take no row predicate (with them in the loop the compiler interleaved the tail's branch and spilled)
    auto chunk = [&](int c, auto full) {
        // the buffer opaque: else the loop is unrolled by two and both buffers' fragment addresses are held
        unsigned bo = (c & 1) * static_cast<unsigned>(sizeof(Buf));
        asm volatile("" : "+s"(bo));
        const Buf& b = *reinterpret_cast<const Buf*>(reinterpret_cast<const char*>(bufs) + bo);
        const int t0 = c * C, n = min(C, length - t0);
        int tid = tid0;
        asm volatile("" : "+v"(tid));
        const int lane = tid & 31, rl = lane & 15, rh = lane >> 4;
        const int r = wave * 16 + rl;
        float vv[2][8];
        load_v(c, vv, full);
#pragma unroll
        for (int dt = 0; dt < DK / 16; ++dt) asm volatile("" : "+v"(st[dt]));
        f8 z[2], y0[2];
        z[0] = z[1] = y0[0] = y0[1] = zero8();
#pragma unroll
        for (int dt = 0; dt < DK / 16; ++dt) {
            const h16 sf = colfrag(st[dt], rh);
            z[0] = mma(frag(&b.ks[rl][dt * 16]), sf, z[0]);
            z[1] = mma(frag(&b.ks[16 + rl][dt * 16]), sf, z[1]);
            y0[0] = mma(frag(&b.qs[rl][dt * 16]), sf, y0[0]);
            y0[1] = mma(frag(&b.qs[16 + rl][dt * 16]), sf, y0[1]);
            __builtin_amdgcn_sched_barrier(0);
        }
#pragma unroll
        for (int ti = 0; ti < 2; ++ti)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float e = __expf(b.cum[ti * 16 + 2 * i + rh]);
                z[ti][i] = vv[ti][i] - e * z[ti][i];
                y0[ti][i] *= e;
            }
        __builtin_amdgcn_sched_barrier(0);
        const h16 zf0 = colfrag(z[0], rh), zf1 = colfrag(z[1], rh);
        f8 dd[2];
        dd[0] = mma(frag(&b.tb[rl][0]), zf0, zero8());
        dd[1] = mma(frag(&b.tb[16 + rl][16]), zf1, mma(frag(&b.tb[16 + rl][0]), zf0, zero8()));
        __builtin_amdgcn_sched_barrier(0);
        const h16 df0 = colfrag(dd[0], rh), df1 = colfrag(dd[1], rh);
        y0[0] = mma(frag(&b.ps[rl][0]), df0, y0[0]);
        y0[1] = mma(frag(&b.ps[16 + rl][16]), df1, mma(frag(&b.ps[16 + rl][0]), df0, y0[1]));
        float* yb = y + (static_cast<size_t>(t0) * value_heads + head) * DV + r;
        asm volatile("" : "+v"(yb));
#pragma unroll
        for (int ti = 0; ti < 2; ++ti)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int t = ti * 16 + 2 * i + rh;
                if (decltype(full)::value || t < n) yb[t * row_stride] = y0[ti][i];
            }
        __builtin_amdgcn_sched_barrier(0);
        const float cn = b.cum[C - 1], gn = __expf(cn);
#pragma unroll
        for (int ti = 0; ti < 2; ++ti)
#pragma unroll
            for (int i = 0; i < 8; ++i) dd[ti][i] *= __expf(cn - b.cum[ti * 16 + 2 * i + rh]);
        const h16 ef0 = colfrag(dd[0], rh), ef1 = colfrag(dd[1], rh);
        // K^T's rows opaque here: else their fragments are read during the products above and spilled
        const _Float16* ktr = &b.kt[rl][0];
        asm volatile("" : "+v"(ktr));
#pragma unroll
        for (int dt = 0; dt < DK / 16; ++dt) {
            f8 acc = st[dt] * gn;
            acc = mma(frag(ktr + dt * 16 * LT), ef0, acc);
            st[dt] = mma(frag(ktr + dt * 16 * LT + 16), ef1, acc);
            __builtin_amdgcn_sched_barrier(0);
        }
        lds_barrier();
    };
    const int whole = length / C;
    for (int c = 0; c < whole; ++c) chunk(c, std::true_type{});
    if (whole < chunks) chunk(whole, std::false_type{});
#pragma unroll
    for (int dt = 0; dt < DK / 16; ++dt)
#pragma unroll
        for (int i = 0; i < 8; ++i) state[(static_cast<size_t>(head) * DV + r0) * DK + dt * 16 + 2 * i + rh0] = st[dt][i];
#else
    (void)q; (void)k; (void)v; (void)tbg; (void)psg; (void)cumg; (void)state; (void)y; (void)length; (void)key_heads; (void)value_heads;
#endif
}

}  // namespace gdnc
}  // namespace
