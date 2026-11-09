#pragma once

// MLX affine kernels: each shared tile over the MLX decoder, identity encoder and Dot, named as the launchers look up.

#include <type_traits>

#include "quant/act.hpp"
#include "quant/mlx_decoder.hpp"
#include "tiles/dot2.hpp"
#include "tiles/epilogue.hpp"
#include "tiles/gemm.hpp"
#include "tiles/gemm_kp.hpp"
#include "tiles/matrix_gemm.hpp"
#include "tiles/matrix_fold.hpp"
#include "tiles/matrix_fold_mix.hpp"
#include "tiles/matrix_fold_tall.hpp"
#include "tiles/matrix_item.hpp"
#include "tiles/stream.hpp"

namespace tf {
namespace rocm {

// ---- the prefill GEMM tile ----

template <typename T, int BITS, int RT>
__global__ void __launch_bounds__(256) affine_gemm_block(Affine a) {
    gemm_tile<MlxDecoder<BITS>, IdentityAct<T>, T, GemmOut<T>, RT>(a);
}

// ---- the K-parallel tiles of a short prompt ----

template <typename T, int BITS, int CB, int R, int WAVES, int RB, bool LOOP>
__global__ void __launch_bounds__(32 * WAVES) affine_gemm_kp(Affine a) {
    gemm_kp_tile<MlxDecoder<BITS>, IdentityAct<T>, T, GemmOut<T>, CB, R, WAVES, RB, LOOP>(a);
}

// ---- the prefill GEMM tile on the matrix cores of gfx11 ----

template <int BITS>
__global__ void __launch_bounds__(256) affine_wmma_gemm(Affine a) {
    wmma_gemm_tile<MlxDecoder<BITS>, BF16Act, WmmaBF16, GemmOut<WmmaBF16>>(a);
}

// ---- the same with the group terms folded into the staged weights (policy fold=on) ----

template <int BITS>
__global__ void __launch_bounds__(256) affine_wmma_fold(Affine a) {
    if (tall_item(a, blockIdx.z)) return;  // the tall plan's item of more than 128 rows: affine_wmma_fold_tall's
    wmma_fold_tile<MlxDecoder<BITS>, BF16Act, WmmaBF16, GemmOut<WmmaBF16>>(a);
}

// ---- the tall plan's items of more than 128 rows on the 256-row fold tile (the others on the 128-row one) ----

template <int BITS>
__global__ void __launch_bounds__(512) affine_wmma_fold_tall(Affine a) {
    wmma_fold_tall_tile<MlxDecoder<BITS>, BF16Act, WmmaBF16, FoldCols<GemmOut<WmmaBF16>>>(a, {a.n});
}

// ---- the fold tile on a dense weight's BF16 copy, and the decode that makes the copy (policy fold=bf16) ----

template <int BITS>
__global__ void __launch_bounds__(256) affine_wmma_fold16(Affine a, const uint16_t* w16) {
    wmma_fold16_tile<MlxDecoder<BITS>, BF16Act, WmmaBF16, GemmOut<WmmaBF16>>(a, w16);
}

// ...with x the hyper-connection norm's values of the streams (HcNorm): the hc down projection without a normed copy.
template <int BITS>
__global__ void __launch_bounds__(256) affine_wmma_fold16_norm(Affine a, const uint16_t* w16, const float* rinv,
                                                              const float* scale, int streams, int dims) {
    wmma_fold16_tile<MlxDecoder<BITS>, BF16Act, WmmaBF16, GemmOut<WmmaBF16>, HcNorm>(
        a, w16, HcNormed{static_cast<const uint16_t*>(a.x), rinv, scale, streams, dims});
}

// (n, k) BF16 weights as wmma_fold stages them: BF16 of fmaf(code, scale, bias), rounded half up.
template <int BITS>
__global__ void __launch_bounds__(256) affine_fold_decode(Affine a, uint16_t* w16) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(a.n) * a.k) return;
    const int col = static_cast<int>(i / a.k), kk = static_cast<int>(i % a.k);
    const int words_row = a.k * BITS / 32;
    const uint32_t code = affine_code(a.words + static_cast<long long>(col) * words_row, kk, BITS, words_row);
    using Dec = MlxDecoder<BITS>;
    const auto term = Dec::term(a, static_cast<long long>(col) * (a.k / a.group) + kk / a.group);
    const float w = fmaf(static_cast<float>(code), Dec::scale_value(a, term), Dec::bias_value(a, term));
    w16[i] = static_cast<uint16_t>((__float_as_uint(w) + 0x8000u) >> 16);
}

// ---- the fold tile of a hyper-connection's up projection with its stream mix as the epilogue ----

template <int BITS>
__global__ void __launch_bounds__(256) affine_wmma_fold_mix(Affine a, const uint16_t* xs, uint16_t* mixed, int dims) {
    wmma_fold_cols_tile<MlxDecoder<BITS>, BF16Act, WmmaBF16, HcMixCols>(a, HcMixCols::Extra{xs, mixed, dims});
}

// ...with the normed streams read through HcNormed (tf_fn_hc_norm_rinv's h and rinv, no normed copy)
template <int BITS>
__global__ void __launch_bounds__(256) affine_wmma_fold_mix_norm(Affine a, const uint16_t* h, const float* rinv,
                                                                const float* scale, uint16_t* mixed, int dims) {
    wmma_fold_cols_tile<MlxDecoder<BITS>, BF16Act, WmmaBF16, HcMixNormCols>(
        a, HcMixNormCols::Extra{HcNormed{h, rinv, scale, 4, dims}, mixed, dims});
}

// ---- the same for a routed gate and up with the activation as the epilogue ----

template <int BITS>
__global__ void __launch_bounds__(256) affine_wmma_fold_pair(Affine a, uint16_t* out, int width, float limit) {
    if (tall_item(a, blockIdx.z)) return;
    wmma_fold_cols_tile<MlxDecoder<BITS>, BF16Act, WmmaBF16, PairCols>(a, PairCols::Extra{out, width, limit});
}

template <int BITS>
__global__ void __launch_bounds__(512) affine_wmma_fold_pair_tall(Affine a, uint16_t* out, int width, float limit) {
    wmma_fold_tall_tile<MlxDecoder<BITS>, BF16Act, WmmaBF16, PairCols>(a, PairCols::Extra{out, width, limit});
}

// ---- the same on RM x 16 rows a block, for a routed item of few rows ----

template <int BITS, int RM>
__global__ void __launch_bounds__(256) affine_wmma_item(Affine a) {
    wmma_item_tile<MlxDecoder<BITS>, BF16Act, WmmaBF16, GemmOut<WmmaBF16>, RM>(a);
}

// ---- the decode stream tile ----

// Up to four products sharing x, K, width and group in one launch; side s owns blocks [first[s], first[s + 1]).
constexpr int kStreamSides = 4;

struct StreamSides {
    const uint32_t* words[kStreamSides];
    const void* scale[kStreamSides];
    const void* bias[kStreamSides];
    void* out[kStreamSides];
    int n[kStreamSides];
    int first[kStreamSides + 1];
    int count;  // 0 is the plain product of the Affine
    int out_half;
    int pair_cols;  // > 0: the (gate | up) rows of a stacked product with this many columns each, out is its activation
    float limit;    // the activation's clamp when above 0
};

// Item z of the plan (or the plain product, or a group's side) at CB columns a lane, R rows; PAIR fuses the activation.
template <typename T, int BITS, int R, int CB, bool PAIR>
__global__ void __launch_bounds__(32 * kStreamWaves) affine_dot2_stream(Affine a, StreamSides sides, int lpc_log2,
                                                                       int gshift) {
    using Epi = std::conditional_t<PAIR, StreamSwiglu<T>, StreamRound<T>>;
    stream_tile<MlxDecoder<BITS>, IdentityAct<T>, T, Epi, R, CB>(a, sides, lpc_log2, gshift);
}

// The plain product (or plan item) with Nemotron's relu^2 as the epilogue, into a.out16 in the activation type.
template <typename T, int BITS, int R, int CB>
__global__ void __launch_bounds__(32 * kStreamWaves) affine_dot2_stream_relu2(Affine a, StreamSides sides, int lpc_log2,
                                                                             int gshift) {
    stream_tile<MlxDecoder<BITS>, IdentityAct<T>, T, StreamRelu2<T>, R, CB>(a, sides, lpc_log2, gshift);
}

// Relu^2 products of one x in one launch: plan a (items on z) on the first `split` blocks, plain b past them at z 0.
template <typename T, int BITS, int R, int CB>
__global__ void __launch_bounds__(32 * kStreamWaves) affine_dot2_stream_relu2_pair(Affine a, Affine b, int split,
                                                                                  int lpc_log2, int gshift) {
    StreamSides sides{};
    if (static_cast<int>(blockIdx.x) < split) {
        stream_tile<MlxDecoder<BITS>, IdentityAct<T>, T, StreamRelu2<T>, R, CB>(a, sides, lpc_log2, gshift);
        return;
    }
    if (blockIdx.z != 0) return;
    sides.count = 1;
    sides.first[0] = split;
    sides.first[1] = gridDim.x;
    sides.words[0] = b.words;
    sides.scale[0] = b.scale.p;
    sides.bias[0] = b.bias.p;
    sides.out[0] = b.out16;
    sides.n[0] = b.n;
    sides.out_half = 1;
    stream_tile<MlxDecoder<BITS>, IdentityAct<T>, T, StreamRelu2<T>, R, CB>(b, sides, lpc_log2, gshift);
}


}  // namespace rocm
}  // namespace tf
