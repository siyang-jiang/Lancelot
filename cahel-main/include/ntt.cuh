#pragma once

#define SAMPLE_SIZE(n) \
    ({size_t SAMPLE_SIZE;                                                \
    switch (n)                                                        \
    {                                                                 \
    case 2048:                                                        \
    case 4096:                                                        \
        SAMPLE_SIZE =  64;                                            \
        break;                                                        \
    case 8192:                                                        \
    case 16384:                                                       \
        SAMPLE_SIZE = 128;                                            \
        break;                                                        \
    case 32768:                                                       \
    case 65536:                                                       \
    case 131072:                                                      \
        SAMPLE_SIZE = 256;                                            \
        break;                                                        \
    default:                                                          \
        throw std::invalid_argument("unsupported polynomial degree when selecting sample size"); \
        break;                                                        \
    }; SAMPLE_SIZE; })

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "gputype.h"
#include "gpucontext.h"
#include "uintmodmath.cuh"

void nwt_1d_radix2_forward_inplace(uint64_t *inout, const DNTTTable &ntt_tables, size_t coeff_modulus_size,
                                   size_t start_modulus_idx);

void nwt_1d_radix2_backward_inplace(uint64_t *inout, const DNTTTable &ntt_tables, size_t coeff_modulus_size,
                                    size_t start_modulus_idx);

//=============================

void nwt_2d_radix8_forward_inplace(uint64_t *inout, const DNTTTable &ntt_tables, size_t coeff_modulus_size,
                                   size_t start_modulus_idx);

void nwt_2d_radix8_forward_inplace_fuse_moddown(
        uint64_t *ct, const uint64_t *cx,
        const DMulUIntMod *bigPInv_mod_q,
        uint64_t *delta,
        const DNTTTable &ntt_tables,
        size_t coeff_modulus_size,
        size_t start_modulus_idx);

void
nwt_2d_radix8_forward_inplace_include_temp_mod(uint64_t *inout, const DNTTTable &ntt_tables, size_t coeff_modulus_size,
                                               size_t start_modulus_idx, size_t total_modulus_size);

void nwt_2d_radix8_forward_inplace_include_special_mod(uint64_t *inout, const DNTTTable &ntt_tables,
                                                       size_t coeff_modulus_size, size_t start_modulus_idx,
                                                       size_t size_QP, size_t size_P);

void nwt_2d_radix8_forward_inplace_include_special_mod_exclude_range(
        uint64_t *inout,
        const DNTTTable &ntt_tables,
        size_t coeff_modulus_size,
        size_t start_modulus_idx,
        size_t size_QP, size_t size_P,
        size_t excluded_range_start, size_t excluded_range_end);

void nwt_2d_radix8_forward_inplace_single_mod(uint64_t *inout, size_t modulus_index, const DNTTTable &ntt_tables,
                                              size_t coeff_modulus_size, size_t start_modulus_idx);

void nwt_2d_radix8_forward_modup_fuse(uint64_t *out,
                                      const uint64_t *in,
                                      size_t modulus_index,
                                      const DNTTTable &ntt_tables,
                                      size_t coeff_modulus_size,
                                      size_t start_modulus_idx);

void nwt_2d_radix8_forward_moddown_fuse(uint64_t *encrypted,
                                        const uint64_t *cx,
                                        const uint64_t *cx_last,
                                        uint64_t *reduced_cx_last,
                                        const DNTTTable &ntt_tables,
                                        const DMulUIntMod *inv_q_last_mod_q,
                                        size_t coeff_modulus_size,
                                        size_t q_coeff_count,
                                        size_t pq_coeff_count,
                                        size_t start_modulus_idx);

void nwt_2d_radix8_backward_inplace(uint64_t *inout, const DNTTTable &ntt_tables, size_t coeff_modulus_size,
                                    size_t start_modulus_idx);

void nwt_2d_radix8_backward(uint64_t *out, const uint64_t *in, const DNTTTable &ntt_tables, size_t coeff_modulus_size,
                            size_t start_modulus_idx);

void
nwt_2d_radix8_backward_scale(uint64_t *out, const uint64_t *in, const DNTTTable &ntt_tables, size_t coeff_modulus_size,
                             size_t start_modulus_idx, const DMulUIntMod *scale);

void nwt_2d_radix8_backward_inplace_scale(uint64_t *inout, const DNTTTable &ntt_tables, size_t coeff_modulus_size,
                                          size_t start_modulus_idx, const DMulUIntMod *scale);

void nwt_2d_radix8_backward_inplace_include_special_mod(uint64_t *inout, const DNTTTable &ntt_tables,
                                                        size_t coeff_modulus_size, size_t start_modulus_idx,
                                                        size_t size_QP, size_t size_P);

void nwt_2d_radix8_backward_inplace_include_temp_mod_scale(uint64_t *inout, const DNTTTable &ntt_tables,
                                                           size_t coeff_modulus_size, size_t start_modulus_idx,
                                                           size_t total_modulus_size, DMulUIntMod *scale);

//=============================

inline void
nwt_forward_inplace(uint64_t *inout, const DNTTTable &ntt_tables, size_t coeff_modulus_size, size_t start_modulus_idx) {
    size_t poly_degree = ntt_tables.n_;
    if (poly_degree > 2048) {
        nwt_2d_radix8_forward_inplace(inout, ntt_tables, coeff_modulus_size, start_modulus_idx);
    } else {
        nwt_1d_radix2_forward_inplace(inout, ntt_tables, coeff_modulus_size, start_modulus_idx);
    }
}

inline void nwt_backward_inplace(uint64_t *inout, const DNTTTable &ntt_tables, size_t coeff_modulus_size,
                                 size_t start_modulus_idx) {
    size_t poly_degree = ntt_tables.n_;
    if (poly_degree > 2048) {
        nwt_2d_radix8_backward_inplace(inout, ntt_tables, coeff_modulus_size, start_modulus_idx);
    } else {
        nwt_1d_radix2_backward_inplace(inout, ntt_tables, coeff_modulus_size, start_modulus_idx);
    }
}
