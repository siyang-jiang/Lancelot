#include "rns.cuh"
#include "ntt.cuh"
#include "polymath.cuh"
#include "rns_bconv.cuh"
#include "evaluate.h"
#include "mempool.cuh"
#include "util.cuh"

using namespace std;
using namespace cahel;
using namespace cahel::util;

__global__ void key_switch_inner_prod_c2_and_evk(
        uint64_t *dst, const uint64_t *c2,
        const uint64_t *const *evks,
        const DModulus *modulus,
        size_t n,
        size_t size_QP, size_t size_QP_n,
        size_t size_QlP, size_t size_QlP_n,
        size_t size_Q, size_t size_Ql,
        size_t beta,
        size_t reduction_threshold) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < size_QlP_n;
         tid += blockDim.x * gridDim.x) {
        size_t nid = tid / n;
        size_t twr = (nid >= size_Ql ? size_Q + (nid - size_Ql) : nid);
        // base_rns = {q0, q1, ..., qj, p}
        DModulus mod = modulus[twr];
        uint64_t evk_id = (tid % n) + twr * n;
        uint64_t c2_id = (tid % n) + nid * n;

        uint128_t prod0, prod1;
        uint128_t acc0, acc1;
        uint64_t res0, res1;

        // ct^x = ( <RNS-Decomp(c*_2), evk_b> , <RNS-Decomp(c*_2), evk_a>
        // evk[key_index][rns]
        //
        // RNS-Decomp(c*_2)[key_index + rns_indx * twr] =
        //           ( {c*_2 mod q0, c*_2 mod q1, ..., c*_2 mod qj} mod q0,
        //             {c*_2 mod q0, c*_2 mod q1, ..., c*_2 mod qj} mod q1,
        //             ...
        //             {c*_2 mod q0, c*_2 mod q1, ..., c*_2 mod qj} mod qj,
        //             {c*_2 mod q0, c*_2 mod q1, ..., c*_2 mod qj} mod p, )
        //
        // decomp_mod_size = number of evks

        // evk[0]_a
        acc0 = multiply_uint64_uint64(c2[c2_id], evks[0][evk_id]);
        // evk[0]_b
        acc1 = multiply_uint64_uint64(c2[c2_id], evks[0][evk_id + size_QP_n]);

        for (uint64_t i = 1; i < beta; i++) {
            if (i && reduction_threshold == 0) {
                acc0.lo = barrett_reduce_uint128_uint64(acc0, mod.value(), mod.const_ratio());
                acc0.hi = 0;

                acc1.lo = barrett_reduce_uint128_uint64(acc1, mod.value(), mod.const_ratio());
                acc1.hi = 0;
            }

            prod0 = multiply_uint64_uint64(c2[c2_id + i * size_QlP_n], evks[i][evk_id]);
            add_uint128_uint128(acc0, prod0, acc0);

            prod1 = multiply_uint64_uint64(c2[c2_id + i * size_QlP_n], evks[i][evk_id + size_QP_n]);
            add_uint128_uint128(acc1, prod1, acc1);
        }

        res0 = barrett_reduce_uint128_uint64(acc0, mod.value(), mod.const_ratio());
        dst[tid] = res0;

        res1 = barrett_reduce_uint128_uint64(acc1, mod.value(), mod.const_ratio());
        dst[tid + size_QlP_n] = res1;
    }
}


// cks refers to cipher to be key-switched
void switch_key_inplace(const CAHELGPUContext &context, CAHELGPUCiphertext &encrypted, uint64_t *cks,
                        const CAHELGPURelinKey &relin_keys, bool is_relin) {
#ifndef CAHEL_PROFILE
    if (relin_keys.parms_id() != (context.cpu_context_)->key_parms_id()) {
        throw invalid_argument("relin_keys is not valid for encryption parameters");
    }
#endif

    // Extract encryption parameters.
    auto cpu_context = context.cpu_context_;

    auto &key_context_data = cpu_context->get_context_data(0);
    auto &key_parms = key_context_data.parms();
    auto scheme = key_parms.scheme();
    auto n = key_parms.poly_modulus_degree();
    auto mul_tech = key_parms.mul_tech();
    auto &key_modulus = key_parms.coeff_modulus();
    size_t size_P = key_parms.special_modulus_size();
    size_t size_QP = key_modulus.size();

    // HPS and HPSOverQ does not drop modulus
    uint32_t levelsDropped;

    if (scheme == scheme_type::bfv) {
        levelsDropped = 0;
        if (mul_tech == mul_tech_type::hps_overq_leveled) {
            size_t depth = encrypted.GetNoiseScaleDeg();
            bool isKeySwitch = !is_relin;
            bool is_Asymmetric = encrypted.is_asymmetric();
            size_t levels = depth - 1;
            auto dcrtBits = static_cast<double>(context.gpu_rns_tool_vec()[1].qMSB_);

            // how many levels to drop
            levelsDropped = FindLevelsToDrop(context, levels, dcrtBits, isKeySwitch, is_Asymmetric);
        }
    } else if (scheme == scheme_type::bgv || scheme == scheme_type::ckks) {
        levelsDropped = encrypted.chain_index() - 1;
    } else {
        throw invalid_argument("unsupported scheme in switch_key_inplace");
    }

    auto &rns_tool = context.gpu_rns_tool_vec_[1 + levelsDropped];

    auto modulus_QP = context.gpu_rns_tables().modulus();

    size_t size_Ql = rns_tool.base_Ql_.size();
    size_t size_Q = size_QP - size_P;
    size_t size_QlP = size_Ql + size_P;

    auto size_Ql_n = size_Ql * n;
    auto size_QP_n = size_QP * n;
    auto size_QlP_n = size_QlP * n;

    if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
        Pointer<uint64_t> t_cks;
        t_cks.acquire(allocate<uint64_t>(Global(), size_Q * n));
        cudaMemcpyAsync(t_cks.get(), cks, size_Q * n * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        rns_tool.scaleAndRound_HPS_Q_Ql(cks, t_cks.get());
        t_cks.release();
    }

    // Prepare key
    auto &key_vector = relin_keys.public_keys_;
    auto key_poly_num = key_vector[0].pk_.size_;

    if (key_poly_num != 2)
        throw std::invalid_argument("key_poly_num != 2");

    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

    // mod up

    Pointer<uint64_t> t_mod_up;
    t_mod_up.acquire(allocate<uint64_t>(Global(), beta * size_QlP_n));

    rns_tool.modup(t_mod_up.get(),
                   cks,
                   context.gpu_rns_tables(),
                   scheme);

    // key switch
    Pointer<uint64_t> cx;
    cx.acquire(allocate<uint64_t>(Global(), 2 * size_QlP_n));

#ifdef CAHEL_PROFILE
    CUDATimer product_timer("product");
    product_timer.start();
#endif
    auto reduction_threshold =
            (1 << (bits_per_uint64 - static_cast<uint64_t>(log2(key_modulus.front().value())) - 1)) - 1;
    key_switch_inner_prod_c2_and_evk<<<size_QlP_n / blockDimGlb.x, blockDimGlb>>>(
            cx.get(), t_mod_up.get(),
            relin_keys.public_keys_ptr_.get(),
            modulus_QP,
            n,
            size_QP, size_QP_n,
            size_QlP, size_QlP_n,
            size_Q, size_Ql,
            beta,
            reduction_threshold);
#ifdef CAHEL_PROFILE
    product_timer.stop();
#endif

    // mod down
    for (size_t i = 0; i < 2; i++) {
        auto cx_i = cx.get() + i * size_QlP_n;
        rns_tool.moddown_from_NTT(cx_i,
                                  cx_i,
                                  context.gpu_rns_tables(),
                                  scheme);
    }

    for (size_t i = 0; i < 2; i++) {
        auto cx_i = cx.get() + i * size_QlP_n;

        if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
            auto ct_i = encrypted.data() + i * size_Q * n;
            Pointer<uint64_t> t_cx;
            t_cx.acquire(allocate<uint64_t>(Global(), size_Q * n));
            rns_tool.ExpandCRTBasis_Ql_Q(t_cx.get(), cx_i);
            add_to_ct_kernel<<<(size_Q * n) / blockDimGlb.x, blockDimGlb>>>(
                    ct_i,
                    t_cx.get(),
                    rns_tool.base_Q_.base(),
                    n, size_Q);
            t_cx.release();
        } else {
            auto ct_i = encrypted.data() + i * size_Ql_n;
            add_to_ct_kernel<<<size_Ql_n / blockDimGlb.x, blockDimGlb>>>(
                    ct_i,
                    cx_i,
                    rns_tool.base_Ql_.base(),
                    n, size_Ql);
        }
    }
}
