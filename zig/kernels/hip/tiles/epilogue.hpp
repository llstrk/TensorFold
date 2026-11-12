#pragma once

// Epilogues: what a tile does with finished sums; GEMM tiles hand one fp32 value, the stream tile a lane's columns.

#include <hip/hip_fp16.h>

#include <type_traits>

#include "tiles/plan.hpp"

namespace tf {
namespace rocm {

// A float as the activation type, rounded to nearest even.
template <typename T>
__device__ inline typename T::elem narrow(float v) {
    if constexpr (std::is_same_v<typename T::elem, __half>) {
        return __float2half_rn(v);
    } else {
        const uint32_t u = __float_as_uint(v);
        const uint32_t bits = v != v ? 0x7fc0u : (u + (((u >> 16) & 1u) + 0x7fffu)) >> 16;
        return __builtin_bit_cast(typename T::elem, static_cast<unsigned short>(bits));
    }
}

// The product at its (row, col), the plan's pair row for a routed item: rounded to T into out16 when set, else fp32.
template <typename T>
struct GemmOut {
    template <class Args>
    __device__ static void store(const Args& a, int row, int col, float v) {
        if (a.out16 != nullptr) {
            static_cast<typename T::elem*>(static_cast<void*>(a.out16))[out_row(a, row) * a.n + col] = narrow<T>(v);
        } else {
            a.out[out_row(a, row) * a.n + col] = v;
        }
    }
};

// A packed gate/up product's pair, as tf_silu_mul over the plain product's two BF16 outputs: g and u rounded as GemmOut
// rounds them, silu(g) = g / (1 + expf(-g)) rounded, then that times u rounded; act (m, n / 2) at feature f.
template <typename T>
struct SiluPairOut {
    static constexpr bool pairs = true;
    __device__ static float widen(typename T::elem v) {
        if constexpr (std::is_same_v<typename T::elem, __half>) return __half2float(v);
        else return __uint_as_float(static_cast<uint32_t>(__builtin_bit_cast(unsigned short, v)) << 16);
    }
    template <class Args>
    __device__ static void store_pair(const Args& a, int row, int f, float g, float u) {
        const float gb = widen(narrow<T>(g)), ub = widen(narrow<T>(u));
        const float s = widen(narrow<T>(gb / (1.0f + expf(-gb))));
        static_cast<typename T::elem*>(static_cast<void*>(a.out16))[out_row(a, row) * (a.n / 2) + f] = narrow<T>(s * ub);
    }
};

// A float product rounded to T: FP16 is one rounding of the exact product (fma, zero addend), never fp32 then FP16.
template <typename T>
__device__ inline typename T::elem narrow_product(float a, float b) {
    if constexpr (std::is_same_v<typename T::elem, __half>) {
        uint32_t bits;
        asm("v_fma_mixlo_f16 %0, %1, %2, 0" : "=v"(bits) : "v"(a), "v"(b));
        return __builtin_bit_cast(typename T::elem, static_cast<unsigned short>(bits));
    } else {
        return narrow<T>(a * b);
    }
}

// silu(gate) * up in the activation type, both clamped by limit first when it is above 0 (moe_act_kernel's arithmetic).
template <typename T>
__device__ inline typename T::elem swiglu(float g, float u, float limit) {
    if (limit > 0.f) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    return narrow_product<T>(g / (1.f + expf(-g)), u);
}

// Stream tile epilogues take a lane's CB column sums for one row (`v`), their column `c` and the paired output width.

// The sum rounded to the activation type where `out16` is set (the decode tails' own rounding), else stored in fp32.
template <class T>
struct StreamRound {
    static constexpr bool kPair = false;

    template <int CB, class Args>
    __device__ static void put(const Args& a, long long row, int col, const float (&v)[CB], int c, int, float) {
        if (a.out16 != nullptr) {
            static_cast<typename T::elem*>(static_cast<void*>(a.out16))[row * a.n + col] = narrow<T>(v[c]);
        } else {
            a.out[row * a.n + col] = v[c];
        }
    }
};

// A lane's columns are CB / 2 gate columns and the same up columns; out is silu(gate) * up in T, `pair_cols` wide.
template <class T>
struct StreamSwiglu {
    static constexpr bool kPair = true;

    template <int CB, class Args>
    __device__ static void put(const Args& a, long long row, int col, const float (&v)[CB], int c, int pair_cols, float limit) {
        constexpr int CH = CB / 2;
        static_cast<typename T::elem*>(static_cast<void*>(a.out16))[row * pair_cols + col] = swiglu<T>(v[c], v[c + CH], limit);
    }
};

// tf_fn_hc_gates on the hc down projection's sums (bf16 activations): column c < low (`pair_cols`) writes the up
// projection's input silu(v) to out16 (rows, low), a later column the inject gate 2 sigmoid(v) to `out` read as
// (rows, n - low) bf16; v = round(round(sum) / 4) as tf_fn_hc_gates rounds z at four streams. The same bytes, no z.
template <class T>
struct StreamHcGates {
    static constexpr bool kPair = false;

    static __device__ __forceinline__ float bf16r(float f) {
        if (f != f) return __uint_as_float(0x7fc00000u);
        const uint32_t u = __float_as_uint(f);
        return __uint_as_float(((u + (((u >> 16) & 1u) + 0x7fffu)) >> 16) << 16);
    }
    static __device__ __forceinline__ uint16_t bf16(float f) { return static_cast<uint16_t>(__float_as_uint(bf16r(f)) >> 16); }

    template <int CB, class Args>
    __device__ static void put(const Args& a, long long row, int col, const float (&v)[CB], int c, int low, float) {
        constexpr float kStreams = 4.f;  // Flash Next's hyper-connection streams, the divisor tf_fn_hc_gates takes
        const float g = bf16r(bf16r(v[c]) / kStreams);
        if (col < low) {
            static_cast<uint16_t*>(static_cast<void*>(a.out16))[row * low + col] = bf16(g / (1.0f + expf(-g)));
        } else {
            static_cast<uint16_t*>(static_cast<void*>(a.out))[row * (a.n - low) + col - low] = bf16(2.f * bf16r(1.0f / (1.0f + expf(-g))));
        }
    }
};

// tf_nemo_relu2 on the sum: relu(the sum rounded to T)^2, the square rounded to fp32 before it is rounded to T.
template <class T>
struct StreamRelu2 {
    static constexpr bool kPair = false;

    template <int CB, class Args>
    __device__ static void put(const Args& a, long long row, int col, const float (&v)[CB], int c, int, float) {
        const typename T::elem r = narrow<T>(v[c]);
        float u;
        if constexpr (std::is_same_v<typename T::elem, __half>) {
            u = __half2float(r);
        } else {
            u = __uint_as_float(static_cast<uint32_t>(__builtin_bit_cast(unsigned short, r)) << 16);
        }
        u = u != u ? u : fmaxf(u, 0.f);  // a NaN stays NaN, as tf_nemo_relu2 keeps it
        float sq = u * u;
        // kept apart from the narrowing: fptrunc(fmul) would otherwise become one mixed-precision op
        asm("" : "+v"(sq));
        static_cast<typename T::elem*>(static_cast<void*>(a.out16))[row * a.n + col] = narrow<T>(sq);
    }
};

}  // namespace rocm
}  // namespace tf
