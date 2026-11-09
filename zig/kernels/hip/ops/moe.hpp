#pragma once

#include "common.hpp"

// The MoE plan and the router's logits.

namespace {

// An expert's items of `c` pairs: at most `tile` pairs each; tile 256 is the tall plan, items of up to 256 pairs while
// more than 128 remain, then one of the rest (affine_wmma_fold_tall's items and the 128-row tile's).
__device__ __forceinline__ int route_tall_big(int c) { return c > 128 ? (c - 128 + 255) / 256 : 0; }
__device__ __forceinline__ int route_items(int c, int tile) {
    if (tile != 256) return (c + tile - 1) / tile;
    const int big = route_tall_big(c);
    return big + (c > big * 256 ? 1 : 0);
}
// item `within` of the expert's: its first pair (from the expert's start) and its count (0 past the expert's items)
__device__ __forceinline__ void route_cut(int c, int tile, int within, int& first, int& count) {
    first = within * tile;
    const int left = c - first;
    const int most = tile != 256 || within < route_tall_big(c) ? tile : 128;
    count = left < 0 ? 0 : (left > most ? most : left);
}

// Stable-sorts the pairs by expert into members and cuts items of at most `tile` pairs; one block, counted per segment.
extern "C" __global__ void tf_moe_route(const int* picks, int pairs, int experts, int tile, int* members, int* items,
                                 int capacity) {
    extern __shared__ int shared[];
    int* counts = shared;
    int* starts = shared + experts;
    int* ends = shared + 2 * experts;
    unsigned short* seg = reinterpret_cast<unsigned short*>(shared + 3 * experts);
    const int segs = experts <= 256 ? 64 : 16;
    const int span = (pairs + segs - 1) / segs;
    const int t = threadIdx.x;
    for (int i = t; i < segs * experts; i += blockDim.x) seg[i] = 0;
    __syncthreads();
    if (t < segs) {
        for (int p = t * span; p < pairs && p < (t + 1) * span; ++p) seg[t * experts + picks[p]] += 1;
    }
    __syncthreads();
    // an expert's total and each segment's first place for it (still relative to the expert's start)
    for (int e = t; e < experts; e += blockDim.x) {
        int run = 0;
        for (int g = 0; g < segs; ++g) {
            int c = seg[g * experts + e];
            seg[g * experts + e] = static_cast<unsigned short>(run);
            run += c;
        }
        counts[e] = run;
    }
    __syncthreads();
    // starts: the counts' exclusive sums; ends: the tiles' inclusive sums; a thread's experts, then the block's scan
    int* scan_c = reinterpret_cast<int*>(seg + segs * experts + (segs * experts) % 2);
    int* scan_n = scan_c + blockDim.x;
    const int per = (experts + blockDim.x - 1) / blockDim.x;
    const int lo = min(experts, t * per), hi = min(experts, lo + per);
    int own_c = 0, own_n = 0;
    for (int e = lo; e < hi; ++e) {
        own_c += counts[e];
        own_n += route_items(counts[e], tile);
    }
    scan_c[t] = own_c;
    scan_n[t] = own_n;
    __syncthreads();
    for (int off = 1; off < static_cast<int>(blockDim.x); off <<= 1) {
        const int vc = t >= off ? scan_c[t - off] : 0, vn = t >= off ? scan_n[t - off] : 0;
        __syncthreads();
        scan_c[t] += vc;
        scan_n[t] += vn;
        __syncthreads();
    }
    int start = scan_c[t] - own_c, end = scan_n[t] - own_n;
    for (int e = lo; e < hi; ++e) {
        starts[e] = start;
        start += counts[e];
        end += route_items(counts[e], tile);
        ends[e] = end;
    }
    __syncthreads();
    if (t < segs) {
        for (int p = t * span; p < pairs && p < (t + 1) * span; ++p) {
            int e = picks[p];
            members[starts[e] + seg[t * experts + e]] = p;
            seg[t * experts + e] += 1;
        }
    }
    for (int slot = t; slot < capacity; slot += blockDim.x) {
        int lo = 0, hi = experts;
        while (lo < hi) {                       // searchsorted(ends, slot, right=True)
            int mid = (lo + hi) / 2;
            if (ends[mid] <= slot) lo = mid + 1;
            else hi = mid;
        }
        int e = lo < experts - 1 ? lo : experts - 1;
        int reps = route_items(counts[e], tile);
        int within = slot - (ends[e] - reps);
        int first, count;
        route_cut(counts[e], tile, within, first, count);
        items[slot * 3] = e;
        items[slot * 3 + 1] = starts[e] + first;
        items[slot * 3 + 2] = count;
    }
}

// tf_moe_route's members and items (the same values) over many blocks, for a prompt's many pairs: (1) a one-wave
// block a span of `span` pairs counts its experts, (2) a wave an expert turns the spans' counts into each span's first
// place for it (its exclusive sum, in place) and the expert's total, (3) each span block places its pairs in order: a
// wave of 32 pairs at a time, a pair's place its span's next place for its expert plus the earlier lanes' picks of it
// (lanes of one expert found by a ballot a bit of the expert's id). The spans' block also cut the items.
extern "C" __global__ void __launch_bounds__(32) tf_moe_route_count(const int* picks, int pairs, int experts, int span,
                                                                   int* span_counts) {
    __shared__ int counts[1024];
    const int g = blockIdx.x, lane = threadIdx.x;
    for (int e = lane; e < experts; e += 32) counts[e] = 0;
    __builtin_amdgcn_wave_barrier();
    const int lo = g * span, hi = min(pairs, lo + span);
    for (int p = lo + lane; p < hi; p += 32) atomicAdd(&counts[picks[p]], 1);
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
    __builtin_amdgcn_wave_barrier();
    for (int e = lane; e < experts; e += 32) span_counts[static_cast<long long>(g) * experts + e] = counts[e];
}

extern "C" __global__ void __launch_bounds__(256) tf_moe_route_spans(int* span_counts, int spans, int experts, int* totals) {
    const int e = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (e >= experts) return;
    int run = 0;
    for (int g0 = 0; g0 < spans; g0 += 32) {
        const int g = g0 + lane;
        const int c = g < spans ? span_counts[static_cast<long long>(g) * experts + e] : 0;
        int s = c;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int u = __shfl_up(s, o, 32);
            if (lane >= o) s += u;
        }
        if (g < spans) span_counts[static_cast<long long>(g) * experts + e] = run + s - c;
        run += __shfl(s, 31, 32);
    }
    if (lane == 0) totals[e] = run;
}

extern "C" __global__ void __launch_bounds__(32) tf_moe_route_place(const int* picks, int pairs, int experts, int tile,
                                                                   int span, const int* span_firsts, const int* totals,
                                                                   int* members, int* items, int capacity) {
    __shared__ int next[1024];
    __shared__ int ends[1024];
    __shared__ int starts[1024];
    const int g = blockIdx.x, lane = threadIdx.x;
    // the experts' starts (the totals' exclusive sums) and their items' inclusive sums, 32 experts a step
    int start = 0, end = 0;
    for (int e0 = 0; e0 < experts; e0 += 32) {
        const int e = e0 + lane;
        const int c = e < experts ? totals[e] : 0, n = route_items(c, tile);
        int sc = c, sn = n;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int uc = __shfl_up(sc, o, 32), un = __shfl_up(sn, o, 32);
            if (lane >= o) { sc += uc; sn += un; }
        }
        if (e < experts) {
            starts[e] = start + sc - c;
            next[e] = start + sc - c + span_firsts[static_cast<long long>(g) * experts + e];
            ends[e] = end + sn;
        }
        start += __shfl(sc, 31, 32);
        end += __shfl(sn, 31, 32);
    }
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
    __builtin_amdgcn_wave_barrier();
    const int lo = g * span, hi = min(pairs, lo + span);
    const unsigned below = (1u << lane) - 1u;
    for (int p0 = lo; p0 < hi; p0 += 32) {
        const int p = p0 + lane;
        const bool live = p < hi;
        const int e = live ? picks[p] : 0;
        unsigned peers = static_cast<unsigned>(__ballot(live));
#pragma unroll
        for (int b = 0; b < 10; ++b) {
            const unsigned set = static_cast<unsigned>(__ballot((e >> b) & 1));
            peers &= (e >> b) & 1 ? set : ~set;
        }
        const int at = live ? next[e] : 0;
        __builtin_amdgcn_wave_barrier();
        if (live) {
            members[at + __popc(peers & below)] = p;
            if ((peers & below) == 0) next[e] = at + __popc(peers);  // the expert's first lane
        }
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();
    }
    // items: tf_moe_route's slot loop over the blocks
    for (int slot = g * 32 + lane; slot < capacity; slot += gridDim.x * 32) {
        int l = 0, h = experts;
        while (l < h) {
            const int mid = (l + h) / 2;
            if (ends[mid] <= slot) l = mid + 1;
            else h = mid;
        }
        const int e = l < experts - 1 ? l : experts - 1;
        const int c = totals[e];
        const int reps = route_items(c, tile);
        const int within = slot - (ends[e] - reps);
        int first, count;
        route_cut(c, tile, within, first, count);
        items[slot * 3] = e;
        items[slot * 3 + 1] = starts[e] + first;
        items[slot * 3 + 2] = count;
    }
}

// moe_router_kernel's logits with a wave's expert row read once for 8 token rows; each logit keeps its bits.
__device__ __forceinline__ float router_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x += __shfl_xor(x, mask, 32);
    return __shfl(x, 0, 32);
}

template <int KIND>
__device__ void router_rows(const void* x, const float* rows, float* logits, int r, int d, int e) {
    constexpr int kChunk = 1024;
    __shared__ float xs[8][kChunk];
    const int lane = threadIdx.x & 31;
    const int expert = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int row0 = blockIdx.y * 8;
    const float* w = rows + static_cast<long long>(expert < e ? expert : 0) * d;
    float acc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = 0.f;
    for (int base = 0; base < d; base += kChunk) {
        const int n = d - base < kChunk ? d - base : kChunk;
        __syncthreads();
        for (int q = threadIdx.x; q < 8 * kChunk; q += blockDim.x) {
            const int k = q / kChunk, i = q % kChunk;
            xs[k][i] = (row0 + k < r && i < n) ? load(x, KIND, static_cast<long long>(row0 + k) * d + base + i) : 0.f;
        }
        __syncthreads();
        // lane l keeps adding i = l, l + 32, ... in order across the chunks
        for (int i = lane; i < n; i += 32) {
            const float wv = w[base + i];
#pragma unroll
            for (int k = 0; k < 8; ++k) acc[k] = fmaf(xs[k][i], wv, acc[k]);
        }
    }
    if (expert >= e) return;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float sum = router_sum(acc[k]);
        if (lane == 0 && row0 + k < r) logits[static_cast<long long>(row0 + k) * e + expert] = sum;
    }
}

// The router's logits for many rows: a 64 x 64 tile of (row, expert) a block, 4 x 4 a thread, in fp32.
template <int KIND>
__device__ void router_tile(const void* x, const float* rows, float* logits, int r, int d, int e) {
    constexpr int kTile = 64, kStep = 32, kLd = kTile + 4;
    __shared__ alignas(16) float xs[kStep][kLd];
    __shared__ alignas(16) float ws[kStep][kLd];
    const int row0 = blockIdx.y * kTile, expert0 = blockIdx.x * kTile;
    const int ty = threadIdx.x / 16, tx = threadIdx.x % 16;
    // the staging: activation row ar at steps ak .. ak + 7, router rows wr and wr + 32 at steps wk .. wk + 3
    const int ar = threadIdx.x / 4, ak = (threadIdx.x % 4) * 8;
    const int wr = threadIdx.x / 8, wk = (threadIdx.x % 8) * 4;
    const uint16_t* xh = static_cast<const uint16_t*>(x);
    float acc[4][4] = {};
    for (int base = 0; base < d; base += kStep) {
        uint4 xv = {0, 0, 0, 0};
        if (row0 + ar < r) xv = *reinterpret_cast<const uint4*>(xh + static_cast<long long>(row0 + ar) * d + base + ak);
        float4 w0 = {0, 0, 0, 0}, w1 = {0, 0, 0, 0};
        if (expert0 + wr < e) w0 = *reinterpret_cast<const float4*>(rows + static_cast<long long>(expert0 + wr) * d + base + wk);
        if (expert0 + wr + 32 < e)
            w1 = *reinterpret_cast<const float4*>(rows + static_cast<long long>(expert0 + wr + 32) * d + base + wk);
        __syncthreads();
        const uint16_t* h = reinterpret_cast<const uint16_t*>(&xv);
#pragma unroll
        for (int j = 0; j < 8; ++j) xs[ak + j][ar] = KIND == 1 ? __half2float(__ushort_as_half(h[j])) : bf16_to_float(h[j]);
        const float a0[4] = {w0.x, w0.y, w0.z, w0.w}, a1[4] = {w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) ws[wk + j][wr] = a0[j], ws[wk + j][wr + 32] = a1[j];
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kStep; ++i) {
            const float4 a = *reinterpret_cast<const float4*>(&xs[i][4 * ty]);
            const float4 b = *reinterpret_cast<const float4*>(&ws[i][4 * tx]);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int w = 0; w < 4; ++w) acc[u][w] = fmaf(av[u], bv[w], acc[u][w]);
        }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int row = row0 + 4 * ty + u;
        if (row >= r) continue;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int expert = expert0 + 4 * tx + w;
            if (expert < e) logits[static_cast<long long>(row) * e + expert] = acc[u][w];
        }
    }
}

// router_tile's logits for up to 32 rows: a wave an expert, a lane a row, each logit one in-order fmaf chain.
template <int KIND>
__device__ void router_small(const void* x, const float* rows, float* logits, int r, int d, int e) {
    const int lane = threadIdx.x & 31;
    const int expert = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int row = blockIdx.y * 32 + lane;
    if (expert >= e) return;
    const uint16_t* xh = static_cast<const uint16_t*>(x) + static_cast<long long>(row < r ? row : 0) * d;
    const float* w = rows + static_cast<long long>(expert) * d;
    float acc = 0.f;
    for (int base = 0; base < d; base += 8) {
        const uint4 xv = *reinterpret_cast<const uint4*>(xh + base);
        const uint16_t* h = reinterpret_cast<const uint16_t*>(&xv);
        const float4 w0 = *reinterpret_cast<const float4*>(w + base);
        const float4 w1 = *reinterpret_cast<const float4*>(w + base + 4);
        const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc = fmaf(KIND == 1 ? __half2float(__ushort_as_half(h[j])) : bf16_to_float(h[j]), wv[j], acc);
        }
    }
    if (row < r) logits[static_cast<long long>(row) * e + expert] = acc;
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256) tf_router_small_f16(const void* x, const float* rows, float* logits,
                                                                      int r, int d, int e) {
    router_small<1>(x, rows, logits, r, d, e);
}

extern "C" __global__ void __launch_bounds__(256) tf_router_small_bf16(const void* x, const float* rows, float* logits,
                                                                       int r, int d, int e) {
    router_small<2>(x, rows, logits, r, d, e);
}

extern "C" __global__ void __launch_bounds__(256) tf_router_tile_f16(const void* x, const float* rows, float* logits,
                                                                     int r, int d, int e) {
    router_tile<1>(x, rows, logits, r, d, e);
}

extern "C" __global__ void __launch_bounds__(256) tf_router_tile_bf16(const void* x, const float* rows, float* logits,
                                                                      int r, int d, int e) {
    router_tile<2>(x, rows, logits, r, d, e);
}

extern "C" __global__ void __launch_bounds__(256) tf_router_rows_f16(const void* x, const float* rows, float* logits,
                                                                     int r, int d, int e) {
    router_rows<1>(x, rows, logits, r, d, e);
}

extern "C" __global__ void __launch_bounds__(256) tf_router_rows_bf16(const void* x, const float* rows, float* logits,
                                                                      int r, int d, int e) {
    router_rows<2>(x, rows, logits, r, d, e);
}
