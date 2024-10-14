#include "rns.cuh"
#include "ntt.cuh"
#include "polymath.cuh"

using namespace std;
using namespace cahel;
using namespace cahel::util;

void DRNSTool::init(const RNSTool &cpu_rns_tool) {
    mul_tech_ = cpu_rns_tool.mul_tech();
    n_ = cpu_rns_tool.coeff_count();
    size_QP_ = cpu_rns_tool.total_modulus_size();
    size_P_ = cpu_rns_tool.special_modulus_size();
    size_t size_Q = size_QP_ - size_P_;

    if (cpu_rns_tool.base() != nullptr)
        base_.init(*(cpu_rns_tool.base()));

    if (cpu_rns_tool.base_Q() != nullptr)
        base_Q_.init(*(cpu_rns_tool.base_Q()));

    if (cpu_rns_tool.base_Ql() != nullptr)
        base_Ql_.init(*(cpu_rns_tool.base_Ql()));

    if (cpu_rns_tool.base_QlP() != nullptr)
        base_QlP_.init(*(cpu_rns_tool.base_QlP()));

    size_t base_size = base_.size();
    size_t size_Ql = base_Ql_.size();

    // inv_q_last_mod_q_: q[last]^(-1) mod q[i] for i = 0..last-1
    if (size_Ql > 1) {
        inv_q_last_mod_q_.acquire(allocate<DMulUIntMod>(Global(), (size_Ql - 1)));
        CUDA_CHECK(cudaMemcpy(inv_q_last_mod_q(), cpu_rns_tool.inv_q_last_mod_q(),
                              (size_Ql - 1) * sizeof(DMulUIntMod), cudaMemcpyHostToDevice));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // hybrid key-switching
    ////////////////////////////////////////////////////////////////////////////////////////////////

    if (size_P_ != 0) {
        size_t beta = cpu_rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

        bigP_mod_q_.acquire(allocate<DMulUIntMod>(Global(), size_Ql));
        CUDA_CHECK(cudaMemcpy(bigP_mod_q_.get(), cpu_rns_tool.bigP_mod_q().data(), size_Ql * sizeof(DMulUIntMod),
                              cudaMemcpyHostToDevice));

        bigPInv_mod_q_.acquire(allocate<DMulUIntMod>(Global(), size_Ql));
        CUDA_CHECK(cudaMemcpy(bigPInv_mod_q_.get(), cpu_rns_tool.bigPInv_mod_q().data(), size_Ql * sizeof(DMulUIntMod),
                              cudaMemcpyHostToDevice));

        if (base_size <= size_Q) {
            partQlHatInv_mod_Ql_concat_.acquire(allocate<DMulUIntMod>(Global(), size_Ql));
            if (!cpu_rns_tool.partQlHatInv_mod_Ql_concat().empty())
                CUDA_CHECK(
                        cudaMemcpy(partQlHatInv_mod_Ql_concat_.get(), cpu_rns_tool.partQlHatInv_mod_Ql_concat().data(),
                                   size_Ql * sizeof(DMulUIntMod),
                                   cudaMemcpyHostToDevice));

            v_base_part_Ql_to_compl_part_QlP_conv_.resize(beta);
            for (size_t i = 0; i < beta; i++)
                v_base_part_Ql_to_compl_part_QlP_conv_[i].init(*(cpu_rns_tool.base_part_Ql_to_compl_part_QlP_conv(i)));
        }

        base_P_to_Ql_conv_.init(*(cpu_rns_tool.base_P_to_Ql_conv()));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // plain modulus related (BFV/BGV)
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Modulus t_;
    t_ = DModulus(cpu_rns_tool.t().value(),
                  cpu_rns_tool.t().const_ratio().at(0),
                  cpu_rns_tool.t().const_ratio().at(1));
    q_last_mod_t_ = cpu_rns_tool.q_last_mod_t();
    inv_q_last_mod_t_ = cpu_rns_tool.inv_q_last_mod_t();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // BGV correction factor
    ////////////////////////////////////////////////////////////////////////////////////////////////

    if (!cpu_rns_tool.t().is_zero() && mul_tech_ == mul_tech_type::none) {
        // Base converter: q --> t
        if (cpu_rns_tool.base_q_to_t_conv() != nullptr)
            base_q_to_t_conv_.init(*(cpu_rns_tool.base_q_to_t_conv()));

        if (size_P_ != 0) {
            pjInv_mod_q_.acquire(allocate<DMulUIntMod>(Global(), size_Ql * size_P_));
            CUDA_CHECK(cudaMemcpy(pjInv_mod_q_.get(), cpu_rns_tool.pjInv_mod_q().data(),
                                  size_Ql * size_P_ * sizeof(DMulUIntMod), cudaMemcpyHostToDevice));

            pjInv_mod_t_.acquire(allocate<DMulUIntMod>(Global(), size_P_));
            CUDA_CHECK(cudaMemcpy(pjInv_mod_t_.get(), cpu_rns_tool.pjInv_mod_t().data(),
                                  size_P_ * sizeof(DMulUIntMod), cudaMemcpyHostToDevice));

            bigPInv_mod_t_.operand_ = cpu_rns_tool.bigPInv_mod_t().operand;
            bigPInv_mod_t_.quotient_ = cpu_rns_tool.bigPInv_mod_t().quotient;

            base_P_to_t_conv_.init(*(cpu_rns_tool.base_P_to_t_conv()));
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // BFV enc/add/sub
    ////////////////////////////////////////////////////////////////////////////////////////////////

    negQl_mod_t_.operand_ = cpu_rns_tool.negQl_mod_t().operand;
    negQl_mod_t_.quotient_ = cpu_rns_tool.negQl_mod_t().quotient;

    if (!cpu_rns_tool.tInv_mod_q().empty()) {
        tInv_mod_q_.acquire(allocate<DMulUIntMod>(Global(), size_Ql));
        CUDA_CHECK(cudaMemcpy(tInv_mod_q_.get(), cpu_rns_tool.tInv_mod_q().data(),
                              size_Ql * sizeof(DMulUIntMod), cudaMemcpyHostToDevice));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // BEHZ
    ////////////////////////////////////////////////////////////////////////////////////////////////

    if (mul_tech_ == mul_tech_type::behz && base_size <= size_Q) {
        // Modulus gamma_;
        gamma_ = DModulus(cpu_rns_tool.gamma().value(),
                          cpu_rns_tool.gamma().const_ratio().at(0),
                          cpu_rns_tool.gamma().const_ratio().at(1));

        if (cpu_rns_tool.base_t_gamma() != nullptr)
            base_t_gamma_.init(*(cpu_rns_tool.base_t_gamma()));

        // Base converter: q --> {t, gamma}
        if (cpu_rns_tool.base_q_to_t_gamma_conv() != nullptr)
            base_q_to_t_gamma_conv_.init(*(cpu_rns_tool.base_q_to_t_gamma_conv()));

        // gamma^(-1) mod t
        inv_gamma_mod_t_.operand_ = cpu_rns_tool.inv_gamma_mod_t().operand;
        inv_gamma_mod_t_.quotient_ = cpu_rns_tool.inv_gamma_mod_t().quotient;

        // -prod(q)^(-1) mod {t, gamma}
        auto neg_inv_q_mod_t_gamma_size = cpu_rns_tool.neg_inv_q_mod_t_gamma().size();
        if (neg_inv_q_mod_t_gamma_size > 0) {
            neg_inv_q_mod_t_gamma_.acquire(allocate<DMulUIntMod>(Global(), neg_inv_q_mod_t_gamma_size));
            // CUDA_CHECK(cudaStreamAttachMemAsync(NULL, neg_inv_q_mod_t_gamma(), 0, cudaMemAttachGlobal));
            CUDA_CHECK(cudaMemcpy(neg_inv_q_mod_t_gamma(), cpu_rns_tool.neg_inv_q_mod_t_gamma().data(),
                                  neg_inv_q_mod_t_gamma_size * sizeof(DMulUIntMod), cudaMemcpyHostToDevice));
        }

        // prod({t, gamma}) mod q
        auto prod_t_gamma_mod_q_size = cpu_rns_tool.prod_t_gamma_mod_q().size();
        if (prod_t_gamma_mod_q_size > 0) {
            prod_t_gamma_mod_q_.acquire(allocate<DMulUIntMod>(Global(), prod_t_gamma_mod_q_size));
            CUDA_CHECK(cudaMemcpy(prod_t_gamma_mod_q(), cpu_rns_tool.prod_t_gamma_mod_q().data(),
                                  prod_t_gamma_mod_q_size * sizeof(DMulUIntMod), cudaMemcpyHostToDevice));
        }

    }

    if (mul_tech_ == mul_tech_type::behz && base_size == size_Q) {
        // multiply can only be used at top data level, because BFV doesn't require mod-switching

        base_B_.init(*(cpu_rns_tool.base_B()));
        base_Bsk_.init(*(cpu_rns_tool.base_Bsk()));
        base_Bsk_m_tilde_.init(*(cpu_rns_tool.base_Bsk_m_tilde()));
        base_q_to_Bsk_conv_.init(*(cpu_rns_tool.base_q_to_Bsk_conv()));
        base_q_to_m_tilde_conv_.init(*(cpu_rns_tool.base_q_to_m_tilde_conv()));
        base_B_to_q_conv_.init(*(cpu_rns_tool.base_B_to_q_conv()));
        base_B_to_m_sk_conv_.init(*(cpu_rns_tool.base_B_to_m_sk_conv()));

        // NTTTables for Bsk
        auto &base_Bsk_ntt_tables_cpu = cpu_rns_tool.base_Bsk_ntt_tables();
        auto size_Bsk = base_Bsk_ntt_tables_cpu.size();
        gpu_Bsk_tables_.init(n_, size_Bsk);
        for (size_t i = 0; i < size_Bsk; i++) {
            auto coeff_modulus = base_Bsk_ntt_tables_cpu[i].modulus();
            auto modulus = DModulus(coeff_modulus.value(), coeff_modulus.const_ratio()[0],
                                    coeff_modulus.const_ratio()[1]);
            gpu_Bsk_tables_.set(&(modulus),
                                (uint64_t *) (base_Bsk_ntt_tables_cpu[i].get_from_root_powers().data()),
                                (uint64_t *) (base_Bsk_ntt_tables_cpu[i].get_from_inv_root_powers_div2().data()),
                                i);
        }

        tModBsk_.acquire(allocate<DMulUIntMod>(Global(), cpu_rns_tool.tModBsk().size()));
        CUDA_CHECK(cudaMemcpy(tModBsk_.get(), cpu_rns_tool.tModBsk().data(),
                              cpu_rns_tool.tModBsk().size() * sizeof(DMulUIntMod),
                              cudaMemcpyHostToDevice));

        // inv_prod_q_mod_Bsk_ = prod(q)^(-1) mod Bsk
        auto &inv_prod_q_mod_Bsk_cpu = cpu_rns_tool.inv_prod_q_mod_Bsk();
        inv_prod_q_mod_Bsk_.acquire(allocate<DMulUIntMod>(Global(), inv_prod_q_mod_Bsk_cpu.size()));
        CUDA_CHECK(cudaMemcpy(inv_prod_q_mod_Bsk(), inv_prod_q_mod_Bsk_cpu.data(),
                              inv_prod_q_mod_Bsk_cpu.size() * sizeof(DMulUIntMod),
                              cudaMemcpyHostToDevice));

        // prod(q)^(-1) mod m_tilde
        neg_inv_prod_q_mod_m_tilde_.operand_ = cpu_rns_tool.neg_inv_prod_q_mod_m_tilde().operand;
        neg_inv_prod_q_mod_m_tilde_.quotient_ = cpu_rns_tool.neg_inv_prod_q_mod_m_tilde().quotient;

        // prod(B)^(-1) mod m_sk
        inv_prod_B_mod_m_sk_.operand_ = cpu_rns_tool.inv_prod_B_mod_m_sk().operand;
        inv_prod_B_mod_m_sk_.quotient_ = cpu_rns_tool.inv_prod_B_mod_m_sk().quotient;

        // prod(B) mod q
        prod_B_mod_q_.acquire(allocate<uint64_t>(Global(), cpu_rns_tool.prod_B_mod_q().size()));
        // CUDA_CHECK(cudaStreamAttachMemAsync(NULL, prod_B_mod_q(), 0, cudaMemAttachGlobal));
        CUDA_CHECK(cudaMemcpy(prod_B_mod_q(), cpu_rns_tool.prod_B_mod_q().data(),
                              cpu_rns_tool.prod_B_mod_q().size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

        // m_tilde * QHatInvModq
        auto &m_tilde_QHatInvModq_cpu = cpu_rns_tool.m_tilde_QHatInvModq();
        m_tilde_QHatInvModq_.acquire(allocate<DMulUIntMod>(Global(), m_tilde_QHatInvModq_cpu.size()));
        CUDA_CHECK(cudaMemcpy(m_tilde_QHatInvModq(), m_tilde_QHatInvModq_cpu.data(),
                              sizeof(DMulUIntMod) * m_tilde_QHatInvModq_cpu.size(), cudaMemcpyHostToDevice));

        // m_tilde^(-1) mod Bsk
        auto &inv_m_tilde_mod_Bsk_cpu = cpu_rns_tool.inv_m_tilde_mod_Bsk();
        inv_m_tilde_mod_Bsk_.acquire(allocate<DMulUIntMod>(Global(), inv_m_tilde_mod_Bsk_cpu.size()));
        CUDA_CHECK(cudaMemcpy(inv_m_tilde_mod_Bsk(), inv_m_tilde_mod_Bsk_cpu.data(),
                              sizeof(DMulUIntMod) * inv_m_tilde_mod_Bsk_cpu.size(), cudaMemcpyHostToDevice));

        // prod(q) mod Bsk
        prod_q_mod_Bsk_.acquire(allocate<uint64_t>(Global(), cpu_rns_tool.prod_q_mod_Bsk().size()));
        CUDA_CHECK(cudaMemcpy(prod_q_mod_Bsk(), cpu_rns_tool.prod_q_mod_Bsk().data(),
                              cpu_rns_tool.prod_q_mod_Bsk().size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

        // Modulus m_tilde_;
        m_tilde_ = DModulus(cpu_rns_tool.m_tilde().value(),
                            cpu_rns_tool.m_tilde().const_ratio().at(0),
                            cpu_rns_tool.m_tilde().const_ratio().at(1));
        // Modulus m_sk_;
        m_sk_ = DModulus(cpu_rns_tool.m_sk().value(),
                         cpu_rns_tool.m_sk().const_ratio().at(0),
                         cpu_rns_tool.m_sk().const_ratio().at(1));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // HPS
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // HPS Decrypt Scale&Round
    if ((mul_tech_ == mul_tech_type::hps ||
         mul_tech_ == mul_tech_type::hps_overq ||
         mul_tech_ == mul_tech_type::hps_overq_leveled) && (base_size <= size_Q)) {

        qMSB_ = cpu_rns_tool.qMSB();
        sizeQMSB_ = cpu_rns_tool.sizeQMSB();
        tMSB_ = cpu_rns_tool.tMSB();

        t_QHatInv_mod_q_div_q_mod_t_.acquire(
                allocate<DMulUIntMod>(Global(), cpu_rns_tool.t_QHatInv_mod_q_div_q_mod_t().size()));
        CUDA_CHECK(cudaMemcpy(t_QHatInv_mod_q_div_q_mod_t(), cpu_rns_tool.t_QHatInv_mod_q_div_q_mod_t().data(),
                              cpu_rns_tool.t_QHatInv_mod_q_div_q_mod_t().size() * sizeof(DMulUIntMod),
                              cudaMemcpyHostToDevice));

        t_QHatInv_mod_q_div_q_frac_.acquire(
                allocate<double>(Global(), cpu_rns_tool.t_QHatInv_mod_q_div_q_frac().size()));
        CUDA_CHECK(cudaMemcpy(t_QHatInv_mod_q_div_q_frac(), cpu_rns_tool.t_QHatInv_mod_q_div_q_frac().data(),
                              cpu_rns_tool.t_QHatInv_mod_q_div_q_frac().size() * sizeof(double),
                              cudaMemcpyHostToDevice));

        t_QHatInv_mod_q_B_div_q_mod_t_.acquire(
                allocate<DMulUIntMod>(Global(), cpu_rns_tool.t_QHatInv_mod_q_B_div_q_mod_t().size()));
        CUDA_CHECK(cudaMemcpy(t_QHatInv_mod_q_B_div_q_mod_t(), cpu_rns_tool.t_QHatInv_mod_q_B_div_q_mod_t().data(),
                              cpu_rns_tool.t_QHatInv_mod_q_B_div_q_mod_t().size() * sizeof(DMulUIntMod),
                              cudaMemcpyHostToDevice));

        t_QHatInv_mod_q_B_div_q_frac_.acquire(
                allocate<double>(Global(), cpu_rns_tool.t_QHatInv_mod_q_B_div_q_frac().size()));
        CUDA_CHECK(cudaMemcpy(t_QHatInv_mod_q_B_div_q_frac(), cpu_rns_tool.t_QHatInv_mod_q_B_div_q_frac().data(),
                              cpu_rns_tool.t_QHatInv_mod_q_B_div_q_frac().size() * sizeof(double),
                              cudaMemcpyHostToDevice));
    }

    // HPS multiply
    // HPS or HPSOverQ don't need to pre-compute at levels other than first data level
    // HPSOverQLeveled doesn't need to pre-compute at the key level
    // otherwise, pre-computations are needed
    // note that if base size equals to Q size, it is the first data level
    if ((mul_tech_ == mul_tech_type::hps && base_size == size_Q) ||
        (mul_tech_ == mul_tech_type::hps_overq && base_size == size_Q) ||
        (mul_tech_ == mul_tech_type::hps_overq_leveled && base_size <= size_Q)) {

        base_Rl_.init(*(cpu_rns_tool.base_Rl()));
        base_QlRl_.init(*(cpu_rns_tool.base_QlRl()));

        // NTTTables for QlRl
        auto &base_QlRl_ntt_tables_cpu = cpu_rns_tool.base_QlRl_ntt_tables();
        auto size_QlRl = base_QlRl_ntt_tables_cpu.size();
        gpu_QlRl_tables_.init(n_, size_QlRl);
        for (size_t i = 0; i < size_QlRl; i++) {
            auto coeff_modulus = base_QlRl_ntt_tables_cpu[i].modulus();
            auto modulus = DModulus(coeff_modulus.value(), coeff_modulus.const_ratio()[0],
                                    coeff_modulus.const_ratio()[1]);
            gpu_QlRl_tables_.set(&(modulus),
                                 (uint64_t *) (base_QlRl_ntt_tables_cpu[i].get_from_root_powers().data()),
                                 (uint64_t *) (base_QlRl_ntt_tables_cpu[i].get_from_inv_root_powers_div2().data()),
                                 i);
        }

        if (cpu_rns_tool.base_Ql_to_Rl_conv())
            base_Ql_to_Rl_conv_.init(*(cpu_rns_tool.base_Ql_to_Rl_conv()));

        if (cpu_rns_tool.base_Rl_to_Ql_conv() != nullptr)
            base_Rl_to_Ql_conv_.init(*(cpu_rns_tool.base_Rl_to_Ql_conv()));
    }

    if (mul_tech_ == mul_tech_type::hps && base_size == size_Q) {
        tRSHatInvModsDivsFrac_.acquire(
                allocate<double>(Global(), cpu_rns_tool.tRSHatInvModsDivsFrac().size()));
        CUDA_CHECK(cudaMemcpy(tRSHatInvModsDivsFrac(), cpu_rns_tool.tRSHatInvModsDivsFrac().data(),
                              cpu_rns_tool.tRSHatInvModsDivsFrac().size() * sizeof(double),
                              cudaMemcpyHostToDevice));

        tRSHatInvModsDivsModr_.acquire(
                allocate<DMulUIntMod>(Global(), cpu_rns_tool.tRSHatInvModsDivsModr().size()));
        CUDA_CHECK(cudaMemcpy(tRSHatInvModsDivsModr(), cpu_rns_tool.tRSHatInvModsDivsModr().data(),
                              cpu_rns_tool.tRSHatInvModsDivsModr().size() * sizeof(DMulUIntMod),
                              cudaMemcpyHostToDevice));
    }

    if ((mul_tech_ == mul_tech_type::hps_overq && base_size == size_Q) ||
        (mul_tech_ == mul_tech_type::hps_overq_leveled && base_size <= size_Q)) {

        tQlSlHatInvModsDivsFrac_.acquire(
                allocate<double>(Global(), cpu_rns_tool.tQlSlHatInvModsDivsFrac().size()));
        CUDA_CHECK(cudaMemcpy(tQlSlHatInvModsDivsFrac(), cpu_rns_tool.tQlSlHatInvModsDivsFrac().data(),
                              cpu_rns_tool.tQlSlHatInvModsDivsFrac().size() * sizeof(double),
                              cudaMemcpyHostToDevice));

        tQlSlHatInvModsDivsModq_.acquire(
                allocate<DMulUIntMod>(Global(), cpu_rns_tool.tQlSlHatInvModsDivsModq().size()));
        CUDA_CHECK(cudaMemcpy(tQlSlHatInvModsDivsModq(), cpu_rns_tool.tQlSlHatInvModsDivsModq().data(),
                              cpu_rns_tool.tQlSlHatInvModsDivsModq().size() * sizeof(DMulUIntMod),
                              cudaMemcpyHostToDevice));

        if (mul_tech_ == mul_tech_type::hps_overq_leveled && base_size < size_Q) {

            base_QlDrop_.init(*(cpu_rns_tool.base_QlDrop()));

            if (cpu_rns_tool.base_Q_to_Rl_conv())
                base_Q_to_Rl_conv_.init(*(cpu_rns_tool.base_Q_to_Rl_conv()));

            if (cpu_rns_tool.base_Ql_to_QlDrop_conv())
                base_Ql_to_QlDrop_conv_.init(*(cpu_rns_tool.base_Ql_to_QlDrop_conv()));

            QlQHatInvModqDivqFrac_.acquire(
                    allocate<double>(Global(), cpu_rns_tool.QlQHatInvModqDivqFrac().size()));
            CUDA_CHECK(cudaMemcpy(QlQHatInvModqDivqFrac(), cpu_rns_tool.QlQHatInvModqDivqFrac().data(),
                                  cpu_rns_tool.QlQHatInvModqDivqFrac().size() * sizeof(double),
                                  cudaMemcpyHostToDevice));

            QlQHatInvModqDivqModq_.acquire(
                    allocate<DMulUIntMod>(Global(), cpu_rns_tool.QlQHatInvModqDivqModq().size()));
            CUDA_CHECK(cudaMemcpy(QlQHatInvModqDivqModq(), cpu_rns_tool.QlQHatInvModqDivqModq().data(),
                                  cpu_rns_tool.QlQHatInvModqDivqModq().size() * sizeof(DMulUIntMod),
                                  cudaMemcpyHostToDevice));
        }
    }
}

__global__ void perform_final_multiplication(uint64_t *dst, const uint64_t *src, const DMulUIntMod inv_gamma_mod_t,
                                             const uint64_t poly_degree, const DModulus *base_t_gamma) {

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree;
         tid += blockDim.x * gridDim.x) {
        DModulus t = base_t_gamma[0];
        uint64_t gamma_value = base_t_gamma[1].value();
        uint64_t threshold_value = gamma_value >> 1;
        uint64_t temp;
        // Need correction because of centered mod
        if (src[poly_degree + tid] > threshold_value) {
            // Compute -(gamma - a) instead of (a - gamma)
            temp = barrett_reduce_uint64_uint64((gamma_value - src[poly_degree + tid]), t.value(),
                                                t.const_ratio()[1]);
            temp = add_uint64_uint64_mod(src[tid], temp, t.value());
        } else {
            // No correction needed
            temp = barrett_reduce_uint64_uint64(src[poly_degree + tid], t.value(), t.const_ratio()[1]);
            temp = sub_uint64_uint64_mod(src[tid], temp, t.value());
        }
        // If this coefficient was non-zero, multiply by t^(-1)
        dst[tid] = multiply_and_reduce_shoup(temp, inv_gamma_mod_t, t.value());
    }
}

void
DRNSTool::behz_decrypt_scale_and_round(uint64_t *src, uint64_t *temp, const DNTTTable &rns_table,
                                       uint64_t temp_mod_size,
                                       uint64_t poly_modulus_degree, uint64_t *dst) const {
    size_t base_q_size = base_Ql_.size();
    size_t base_t_gamma_size = base_t_gamma_.size();
    size_t coeff_mod_size = rns_table.size();

    // Compute |gamma * t|_qi * ct(s)
    uint64_t gridDimGlb;
    gridDimGlb = n_ * base_q_size / blockDimGlb.x;
    multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb>>>(src, prod_t_gamma_mod_q(), base_Ql_.base(), temp,
                                                          n_,
                                                          base_q_size);

    // Do not need additional memory
    if (temp_mod_size >= base_t_gamma_size) {
        // Convert from q to {t, gamma}
        base_q_to_t_gamma_conv_.bConv_BEHZ(temp, temp, n_);

        // Multiply by -prod(q)^(-1) mod {t, gamma}
        if (coeff_mod_size >= base_t_gamma_size) {
            gridDimGlb = n_ * base_t_gamma_size / blockDimGlb.x;
            multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb>>>(temp, neg_inv_q_mod_t_gamma(),
                                                                  base_t_gamma_.base(),
                                                                  temp, n_, base_t_gamma_size);
        } else {
            // coeff_mod_size = 1
            gridDimGlb = n_ * coeff_mod_size / blockDimGlb.x;
            multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb>>>(temp, neg_inv_q_mod_t_gamma(),
                                                                  base_t_gamma_.base(),
                                                                  temp, n_, coeff_mod_size);
        }

        // Need to correct values in temp_t_gamma (gamma component only) which are
        // larger than floor(gamma/2)

        // Now compute the subtraction to remove error and perform final multiplication by
        // gamma inverse mod t
        gridDimGlb = n_ / blockDimGlb.x;
        perform_final_multiplication<<<gridDimGlb, blockDimGlb>>>(dst, temp, inv_gamma_mod_t_, n_,
                                                                  base_t_gamma_.base());
    } else {
        // Need additional memory
        Pointer<uint64_t> t_gamma;
        t_gamma.acquire(allocate<uint64_t>(Global(), base_t_gamma_size * n_));

        // Convert from q to {t, gamma}
        base_q_to_t_gamma_conv_.bConv_BEHZ(t_gamma.get(), temp, n_);

        // Multiply by -prod(q)^(-1) mod {t, gamma}
        if (coeff_mod_size >= base_t_gamma_size) {
            gridDimGlb = n_ * base_t_gamma_size / blockDimGlb.x;
            multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb>>>(t_gamma.get(), neg_inv_q_mod_t_gamma(),
                                                                  base_t_gamma_.base(), t_gamma.get(), n_,
                                                                  base_t_gamma_size);
        } else {
            gridDimGlb = n_ * coeff_mod_size / blockDimGlb.x;
            multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb>>>(t_gamma.get(), neg_inv_q_mod_t_gamma(),
                                                                  base_t_gamma_.base(), t_gamma.get(), n_,
                                                                  coeff_mod_size);
        }

        // Need to correct values in temp_t_gamma (gamma component only) which are
        // larger than floor(gamma/2)

        // Now compute the subtraction to remove error and perform final multiplication by
        // gamma inverse mod t
        gridDimGlb = n_ / blockDimGlb.x;
        perform_final_multiplication<<<gridDimGlb, blockDimGlb>>>(dst, t_gamma.get(), inv_gamma_mod_t_, n_,
                                                                  base_t_gamma_.base());
    }
}

__global__ void divide_and_round_q_last_kernel(uint64_t *dst, const uint64_t *src, const DModulus *base_q,
                                               const DMulUIntMod *inv_q_last_mod_q, const uint64_t poly_degree,
                                               const uint64_t next_base_q_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * next_base_q_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        DModulus mod = base_q[twr];
        DMulUIntMod q_last_inv = inv_q_last_mod_q[twr];

        // uint64_t q_last_value = base_q[next_base_q_size].value();
        uint64_t c_last_coeff = src[(tid % poly_degree) + next_base_q_size * poly_degree];

        uint64_t temp;

        // q_last^(-1) * (ci[j] - (ci[last] mod qj)) mod qj
        temp = barrett_reduce_uint64_uint64(c_last_coeff, mod.value(), mod.const_ratio()[1]);
        temp = sub_uint64_uint64_mod(src[tid], temp, mod.value());

        // q_last^(-1) * (ci[j] + (-ci[last] mod qlast)) mod qj
        // sub_uint64_uint64(q_last_value, c_last_coeff, temp);
        // add_uint64_uint64(temp, src[tid], temp);
        // temp = barrett_reduce_uint64_uint64(temp, mod.value(), mod.const_ratio()[1]);
        dst[tid] = multiply_and_reduce_shoup(temp, q_last_inv, mod.value());
    }
}

/**
 * N: poly_modulus_degree_
 * base_q_size: coeff_modulus_size_
 */
void
DRNSTool::divide_and_round_q_last(const uint64_t *src, size_t cipher_size, uint64_t *dst) const {
    size_t size_Ql = base_Ql_.size();
    size_t next_size_Ql = size_Ql - 1;
    // Add (qj-1)/2 to change from flooring to rounding
    // qlast^(-1) * (ci[j] - ci[last]) mod qj
    uint64_t gridDimGlb = n_ * next_size_Ql / blockDimGlb.x;
    for (size_t i = 0; i < cipher_size; i++) {
        divide_and_round_q_last_kernel<<<gridDimGlb, blockDimGlb>>>(
                dst + i * next_size_Ql * n_,
                src + i * size_Ql * n_,
                base_Ql_.base(), inv_q_last_mod_q(), n_, next_size_Ql);
    }
}

__global__ void divide_and_round_reduce_q_last_kernel(uint64_t *dst,
                                                      const uint64_t *src,
                                                      const DModulus *base_q,
                                                      const uint64_t poly_degree,
                                                      const uint64_t next_base_q_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * next_base_q_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        DModulus mod = base_q[twr];
        uint64_t c_last_coeff = src[(tid % poly_degree) + next_base_q_size * poly_degree];

        // ci[last] mod qj
        dst[tid] = barrett_reduce_uint64_uint64(c_last_coeff, mod.value(), mod.const_ratio()[1]);
    }
}

__global__ void divide_and_round_ntt_inv_scalar_kernel(uint64_t *dst,
                                                       const uint64_t *src,
                                                       const DModulus *base_q,
                                                       const DMulUIntMod *inv_q_last_mod_q,
                                                       const uint64_t poly_degree,
                                                       const uint64_t next_base_q_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * next_base_q_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        uint64_t mod_value = base_q[twr].value();
        DMulUIntMod q_last_inv = inv_q_last_mod_q[twr];

        uint64_t temp;

        temp = sub_uint64_uint64_mod(src[tid], dst[tid], mod_value);
        dst[tid] = multiply_and_reduce_shoup(temp, q_last_inv, mod_value);
    }
}

void DRNSTool::divide_and_round_q_last_ntt(const uint64_t *src,
                                           size_t cipher_size, const DNTTTable &rns_tables, uint64_t *dst) const {
    size_t base_q_size = base_Ql_.size();
    auto next_base_q_size = base_q_size - 1;
    uint64_t gridDimGlb = n_ * next_base_q_size / blockDimGlb.x;

    for (size_t i = 0; i < cipher_size; i++) {
        uint64_t *ci_in = (uint64_t *) src + i * n_ * base_q_size;
        uint64_t *ci_out = dst + i * n_ * next_base_q_size;

        //  Convert ci[last] to non-NTT form
        nwt_2d_radix8_backward_inplace(ci_in, rns_tables, 1, base_q_size - 1);

        // ci[last] mod qj
        divide_and_round_reduce_q_last_kernel<<<gridDimGlb, blockDimGlb>>>(ci_out, ci_in, base_Ql_.base(), n_,
                                                                           next_base_q_size);

        // Convert to NTT form
        nwt_2d_radix8_forward_inplace(ci_out, rns_tables, next_base_q_size, 0);

        // qlast^(-1) * (ci[j] - (ci[last] mod qj)) mod qj
        divide_and_round_ntt_inv_scalar_kernel<<<gridDimGlb, blockDimGlb>>>(
                ci_out, ci_in,
                base_Ql_.base(),
                inv_q_last_mod_q(), n_, next_base_q_size);
    }
}

void DRNSTool::decrypt_mod_t(uint64_t *dst, const uint64_t *src, const uint64_t poly_degree) const {
    base_q_to_t_conv_.exact_convert_array(dst, src, poly_degree);
}

/**
 * Optimization1: merge m_tilde into BConv phase 1 (BEHZ16)
 * Optimization2: calculate phase 1 once
 * Original: call two BConv (Q -> Bsk, Q -> m_tilde)
 * @param dst Output in base Bsk U {m_tilde}
 * @param src Input in base q
 */
void DRNSTool::fastbconv_m_tilde(uint64_t *dst, uint64_t *src) const {
    size_t base_Q_size = base_Ql_.size();
    size_t base_Bsk_size = base_Bsk_.size();
    auto n = n_;

    Pointer<uint64_t> temp_bconv;
    temp_bconv.acquire(allocate<uint64_t>(Global(), base_Q_size * n));

    constexpr int unroll_factor = 2;

    // multiply HatInv
    uint64_t gridDimGlb = base_Q_size * n / unroll_factor / blockDimGlb.x;
    bconv_mult_unroll2_kernel<<<gridDimGlb, blockDimGlb>>>(
            temp_bconv.get(),
            src,
            m_tilde_QHatInvModq(),
            base_Q_.base(), base_Q_size,
            n);

    // convert to Bsk
    gridDimGlb = base_Bsk_size * n / unroll_factor / blockDimGlb.x;
    bconv_matmul_unroll2_kernel<<<gridDimGlb, blockDimGlb, sizeof(uint64_t) * base_Bsk_size * base_Q_size>>>(
            dst,
            temp_bconv.get(),
            base_q_to_Bsk_conv_.QHatModp(),
            base_Q_.base(), base_Q_size,
            base_Bsk_.base(), base_Bsk_size,
            n);

    // convert to m_tilde
    gridDimGlb = 1 * n / unroll_factor / blockDimGlb.x; // m_tilde size is 1
    bconv_matmul_unroll2_kernel<<<gridDimGlb, blockDimGlb, sizeof(uint64_t) * 1 * base_Q_size>>>(
            dst + n * base_Bsk_size,
            temp_bconv.get(),
            base_q_to_m_tilde_conv_.QHatModp(),
            base_Q_.base(), base_Q_size,
            base_q_to_m_tilde_conv_.obase().base(), 1,
            n);
}

/** used in BFV BEHZ: result = (input + prod_q_mod_Bsk_elt * r_m_tilde)* inv_m_tilde_mod_Bsk mod modulus
 *@notice m_tilde_div_2 and m_tilde is used for ensure r_m_tilde <= m_tilde_div_2
 * @param[out] dst The buff to hold the result
 * @param[in] src in size N
 * @param[in] neg_inv_prod_q_mod_m_tilde
 * @param[in] m_tilde_ptr in size N
 * @param[in] prod_q_mod_Bsk
 * @param[in] inv_m_tilde_mod_Bsk
 * @param[in] modulus
 * @param[in] poly_degree
 */
__global__ void sm_mrq_kernel(uint64_t *dst, const uint64_t *src,
                              const uint64_t m_tilde,
                              const DMulUIntMod neg_inv_prod_q_mod_m_tilde,
                              const DModulus *base_Bsk,
                              const uint64_t *prod_q_mod_Bsk,
                              const DMulUIntMod *inv_m_tilde_mod_Bsk,
                              const uint64_t poly_degree, const uint64_t base_Bsk_size) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * base_Bsk_size;
         tid += blockDim.x * gridDim.x) {
        size_t twr = tid / poly_degree;
        DModulus Bski = base_Bsk[twr];
        uint64_t prod_q_mod_Bski = prod_q_mod_Bsk[twr];
        DMulUIntMod inv_m_tilde_mod_Bski = inv_m_tilde_mod_Bsk[twr];

        // the last component of input mod m_tilde (c''_m_tilde)
        uint64_t r_m_tilde = src[(tid % poly_degree) + base_Bsk_size * poly_degree];
        // compute r_m_tilde = - in[last] * q^(-1) mod m_tilde
        r_m_tilde = multiply_and_reduce_shoup(r_m_tilde, neg_inv_prod_q_mod_m_tilde, m_tilde);
        // set r_m_tilde within range [-m_tilde/2, m_tilde/2)
        if (r_m_tilde >= m_tilde >> 1) {
            r_m_tilde += Bski.value() - m_tilde;
        }
        // c'_Bsk = (c''_Bsk + q * (r_m_tilde mod Bsk)) * m_tilde^(-1) mod Bsk
        uint128_t temp;
        temp = multiply_uint64_uint64(r_m_tilde, prod_q_mod_Bski);
        temp = add_uint128_uint64(temp, src[tid]);
        temp.lo = barrett_reduce_uint128_uint64(temp, Bski.value(), Bski.const_ratio());
        dst[tid] = multiply_and_reduce_shoup(temp.lo, inv_m_tilde_mod_Bski, Bski.value());
    }
}

/*
 Require: Input in base Bsk U {m_tilde}
 Ensure: Output in base Bsk
*/
void DRNSTool::sm_mrq(uint64_t *dst, const uint64_t *src) const {
    size_t base_Bsk_size = base_Bsk_.size();
    // The last component of the input is mod m_tilde
    // Compute (in + q * r_m_tilde) * m_tilde^(-1) mod Bsk
    uint64_t gridDimGlb = n_ * base_Bsk_size / blockDimGlb.x;
    sm_mrq_kernel<<<gridDimGlb, blockDimGlb>>>(dst, src, m_tilde_.value(),
                                               neg_inv_prod_q_mod_m_tilde_, // -q^(-1) mod m_tilde
                                               base_Bsk_.base(), // mod
                                               prod_q_mod_Bsk(), // q mod Bsk
                                               inv_m_tilde_mod_Bsk(), // m_tilde^(-1) mod Bsk
                                               n_, base_Bsk_size);
}

__global__ static void bconv_fuse_sub_mul_unroll2_kernel(uint64_t *dst,
                                                         const uint64_t *xi_qiHatInv_mod_qi,
                                                         const uint64_t *input_base_Bsk,
                                                         const DMulUIntMod *inv_prod_q_mod_Bsk,
                                                         const uint64_t *QHatModp,
                                                         const DModulus *ibase, uint64_t ibase_size,
                                                         const DModulus *obase, uint64_t obase_size,
                                                         uint64_t n) {
    constexpr const int unroll_number = 2;
    extern __shared__ uint64_t s_QHatModp[];
    for (size_t idx = threadIdx.x; idx < obase_size * ibase_size; idx += blockDim.x) {
        s_QHatModp[idx] = QHatModp[idx];
    }
    __syncthreads();

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < obase_size * n / unroll_number;
         tid += blockDim.x * gridDim.x) {
        const size_t degree_idx = unroll_number * (tid / obase_size);
        const size_t out_prime_idx = tid % obase_size;

        uint128_t2 accum = base_convert_acc_unroll2(xi_qiHatInv_mod_qi,
                                                    s_QHatModp, out_prime_idx,
                                                    n,
                                                    ibase_size,
                                                    degree_idx);

        uint64_t obase_value = obase[out_prime_idx].value();
        uint64_t obase_ratio[2] = {obase[out_prime_idx].const_ratio()[0], obase[out_prime_idx].const_ratio()[1]};
        auto &scale = inv_prod_q_mod_Bsk[out_prime_idx];
        uint64_t out1, out2;
        uint64_t input1, input2;

        out1 = barrett_reduce_uint128_uint64(accum.x, obase_value, obase_ratio);
        out2 = barrett_reduce_uint128_uint64(accum.y, obase_value, obase_ratio);
        ld_two_uint64(input1, input2, input_base_Bsk + out_prime_idx * n + degree_idx);

        sub_uint64_uint64(obase_value, out1, out1);
        add_uint64_uint64(input1, out1, out1);
        out1 = multiply_and_reduce_shoup(out1, scale, obase_value);

        sub_uint64_uint64(obase_value, out2, out2);
        add_uint64_uint64(input2, out2, out2);
        out2 = multiply_and_reduce_shoup(out2, scale, obase_value);

        st_two_uint64(dst + out_prime_idx * n + degree_idx, out1, out2);
    }
}

/**
 * BEHZ step 7: divide by q and floor, producing a result in base Bsk
 * Optimization: fuse sub_and_scale_rns_poly with Q->Bsk BConv phase 2
 * @param input_base_q
 * @param input_base_Bsk
 * @param out_base_Bsk
 * @param temp
 */
void
DRNSTool::fast_floor(uint64_t *input_base_q, uint64_t *input_base_Bsk, uint64_t *out_base_Bsk) const {
    size_t base_Bsk_size = base_Bsk_.size();
    size_t base_Q_size = base_Q_.size();
    auto n = n_;

    // Convert q -> Bsk

    Pointer<uint64_t> temp_bconv;
    temp_bconv.acquire(allocate<uint64_t>(Global(), base_Q_size * n));

    constexpr int unroll_factor = 2;

    // multiply HatInv
    uint64_t gridDimGlb = base_Q_size * n / unroll_factor / blockDimGlb.x;
    bconv_mult_unroll2_kernel<<<gridDimGlb, blockDimGlb>>>(
            temp_bconv.get(),
            input_base_q,
            base_Q_.QHatInvModq(),
            base_Q_.base(), base_Q_size,
            n);

    // convert to Bsk
    gridDimGlb = base_Bsk_size * n / unroll_factor / blockDimGlb.x;
    bconv_fuse_sub_mul_unroll2_kernel<<<gridDimGlb, blockDimGlb, sizeof(uint64_t) * base_Bsk_size * base_Q_size>>>(
            out_base_Bsk,
            temp_bconv.get(),
            input_base_Bsk,
            inv_prod_q_mod_Bsk(),
            base_q_to_Bsk_conv_.QHatModp(),
            base_Q_.base(), base_Q_size,
            base_Bsk_.base(), base_Bsk_size,
            n);
}

__global__ static void bconv_fuse_sub_mul_single_unroll2_kernel(uint64_t *dst,
                                                                const uint64_t *xi_qiHatInv_mod_qi,
                                                                const uint64_t *input_base_Bsk,
                                                                DMulUIntMod inv_prod_q_mod_Bsk,
                                                                const uint64_t *QHatModp,
                                                                const DModulus *ibase, uint64_t ibase_size,
                                                                DModulus obase, uint64_t obase_size,
                                                                uint64_t n) {
    constexpr const int unroll_number = 2;
    extern __shared__ uint64_t s_QHatModp[];
    for (size_t idx = threadIdx.x; idx < obase_size * ibase_size; idx += blockDim.x) {
        s_QHatModp[idx] = QHatModp[idx];
    }
    __syncthreads();

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < obase_size * n / unroll_number;
         tid += blockDim.x * gridDim.x) {
        const size_t degree_idx = unroll_number * (tid / obase_size);
        const size_t out_prime_idx = tid % obase_size;

        uint128_t2 accum = base_convert_acc_unroll2(xi_qiHatInv_mod_qi,
                                                    s_QHatModp, out_prime_idx,
                                                    n,
                                                    ibase_size,
                                                    degree_idx);

        uint64_t obase_value = obase.value();
        uint64_t obase_ratio[2] = {obase.const_ratio()[0], obase.const_ratio()[1]};
        auto &scale = inv_prod_q_mod_Bsk;
        uint64_t out1, out2;
        uint64_t input1, input2;

        out1 = barrett_reduce_uint128_uint64(accum.x, obase_value, obase_ratio);
        out2 = barrett_reduce_uint128_uint64(accum.y, obase_value, obase_ratio);
        ld_two_uint64(input1, input2, input_base_Bsk + out_prime_idx * n + degree_idx);

        sub_uint64_uint64(obase_value, input1, input1);
        add_uint64_uint64(input1, out1, out1);
        out1 = multiply_and_reduce_shoup(out1, scale, obase_value);

        sub_uint64_uint64(obase_value, input2, input2);
        add_uint64_uint64(input2, out2, out2);
        out2 = multiply_and_reduce_shoup(out2, scale, obase_value);

        st_two_uint64(dst + out_prime_idx * n + degree_idx, out1, out2);
    }
}

/**
* BEHZ step 8: use Shenoy-Kumaresan method to convert the result (base Bsk) to base q
 * Optimization1: reuse BConv phase 1
 * Optimization2: fuse sub_and_scale_single_mod_poly with B->m_sk BConv phase 2
* @param input_base_Bsk Input in base Bsk
* @param out_base_q Output in base q
*/
void
DRNSTool::fastbconv_sk(uint64_t *input_base_Bsk, uint64_t *out_base_q) const {
    uint64_t gridDimGlb;

    size_t size_B = base_B_.size();
    size_t size_Bsk = base_Bsk_.size();
    size_t size_Q = base_Q_.size();
    auto n = n_;

    uint64_t *input_base_m_sk = input_base_Bsk + size_B * n;

    Pointer<uint64_t> temp_bconv;
    temp_bconv.acquire(allocate<uint64_t>(Global(), size_B * n));

    Pointer<uint64_t> temp_m_sk;
    temp_m_sk.acquire(allocate<uint64_t>(Global(), n));

    constexpr int unroll_factor = 2;

    // multiply HatInv
    gridDimGlb = size_B * n / unroll_factor / blockDimGlb.x;
    bconv_mult_unroll2_kernel<<<gridDimGlb, blockDimGlb>>>(
            temp_bconv.get(),
            input_base_Bsk,
            base_B_.QHatInvModq(),
            base_B_.base(), size_B,
            n);

    // convert to m_sk
    gridDimGlb = 1 * n / unroll_factor / blockDimGlb.x;
    bconv_fuse_sub_mul_single_unroll2_kernel<<<gridDimGlb, blockDimGlb, sizeof(uint64_t) * 1 * size_B>>>(
            temp_m_sk.get(),
            temp_bconv.get(),
            input_base_m_sk,
            inv_prod_B_mod_m_sk_,
            base_B_to_m_sk_conv_.QHatModp(),
            base_B_.base(), size_B,
            m_sk_, 1,
            n);

    // convert to Q
    gridDimGlb = size_Q * n / unroll_factor / blockDimGlb.x;
    bconv_matmul_unroll2_kernel<<<gridDimGlb, blockDimGlb, sizeof(uint64_t) * size_Q * size_B>>>(
            out_base_q,
            temp_bconv.get(),
            base_B_to_q_conv_.QHatModp(),
            base_B_.base(), size_B,
            base_Q_.base(), size_Q,
            n);

    // (3) Compute FastBconvSK(x, Bsk, q) = (FastBconv(x, B, q) - alpha_sk * B) mod q
    // alpha_sk (stored in temp) is now ready for the Shenoy-Kumaresan conversion; however, note that our
    // alpha_sk here is not a centered reduction, so we need to apply a correction below.
    // TODO: fuse multiply_and_negated_add_rns_poly with B->Q BConv phase 2
    gridDimGlb = n_ * base_Ql_.size() / blockDimGlb.x;
    multiply_and_negated_add_rns_poly<<<gridDimGlb, blockDimGlb>>>(temp_m_sk.get(), m_sk_.value(), prod_B_mod_q(),
                                                                   out_base_q,
                                                                   base_Ql_.base(), out_base_q, n_,
                                                                   base_Ql_.size());
}

__global__ void hps_decrypt_scale_and_round_kernel_small(uint64_t *dst, const uint64_t *src,
                                                         const DMulUIntMod *t_QHatInv_mod_q_div_q_mod_t,
                                                         const double *t_QHatInv_mod_q_div_q_frac,
                                                         uint64_t t, size_t n, size_t size_Ql) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < n;
         tid += blockDim.x * gridDim.x) {
        double floatSum = 0.0;
        uint64_t intSum = 0;
        uint64_t tmp;
        double tInv = 1. / static_cast<double>(t);

        for (size_t i = 0; i < size_Ql; i++) {
            tmp = src[i * n + tid];
            floatSum += static_cast<double>(tmp) * t_QHatInv_mod_q_div_q_frac[i];
            intSum += multiply_and_reduce_shoup(tmp, t_QHatInv_mod_q_div_q_mod_t[i], t);
        }
        // compute modulo reduction by finding the quotient using doubles
        // and then subtracting quotient * t
        floatSum += static_cast<double>(intSum);
        auto quot = static_cast<uint64_t>(floatSum * tInv);
        floatSum -= static_cast<double>(t * quot);
        // rounding
        dst[tid] = llround(floatSum);
    }
}

__global__ void hps_decrypt_scale_and_round_kernel_small_lazy(uint64_t *dst, const uint64_t *src,
                                                              const DMulUIntMod *t_QHatInv_mod_q_div_q_mod_t,
                                                              const double *t_QHatInv_mod_q_div_q_frac,
                                                              uint64_t t, size_t n, size_t size_Ql) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < n;
         tid += blockDim.x * gridDim.x) {
        double floatSum = 0.0;
        uint64_t intSum = 0;
        uint64_t tmp;
        double tInv = 1. / static_cast<double>(t);

        for (size_t i = 0; i < size_Ql; i++) {
            tmp = src[i * n + tid];
            floatSum += static_cast<double>(tmp) * t_QHatInv_mod_q_div_q_frac[i];
            intSum += tmp * t_QHatInv_mod_q_div_q_mod_t[i].operand();
        }
        // compute modulo reduction by finding the quotient using doubles
        // and then subtracting quotient * t
        floatSum += static_cast<double>(intSum);
        auto quot = static_cast<uint64_t>(floatSum * tInv);
        floatSum -= static_cast<double>(t * quot);
        // rounding
        dst[tid] = llround(floatSum);
    }
}

__global__ void hps_decrypt_scale_and_round_kernel_large(uint64_t *dst, const uint64_t *src,
                                                         const DMulUIntMod *t_QHatInv_mod_q_div_q_mod_t,
                                                         const double *t_QHatInv_mod_q_div_q_frac,
                                                         const DMulUIntMod *t_QHatInv_mod_q_B_div_q_mod_t,
                                                         const double *t_QHatInv_mod_q_B_div_q_frac,
                                                         uint64_t t, size_t n, size_t size_Ql, size_t qMSBHf) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < n;
         tid += blockDim.x * gridDim.x) {
        double floatSum = 0.0;
        uint64_t intSum = 0;
        uint64_t tmpLo, tmpHi;
        double tInv = 1. / static_cast<double>(t);

        for (size_t i = 0; i < size_Ql; i++) {
            uint64_t tmp = src[i * n + tid];
            tmpHi = tmp >> qMSBHf;
            tmpLo = tmp & ((1ULL << qMSBHf) - 1);
            floatSum += static_cast<double>(tmpLo) * t_QHatInv_mod_q_div_q_frac[i];
            floatSum += static_cast<double>(tmpHi) * t_QHatInv_mod_q_B_div_q_frac[i];
            intSum += multiply_and_reduce_shoup(tmpLo, t_QHatInv_mod_q_div_q_mod_t[i], t);
            intSum += multiply_and_reduce_shoup(tmpHi, t_QHatInv_mod_q_B_div_q_mod_t[i], t);
        }
        // compute modulo reduction by finding the quotient using doubles
        // and then subtracting quotient * t
        floatSum += static_cast<double>(intSum);
        auto quot = static_cast<uint64_t>(floatSum * tInv);
        floatSum -= static_cast<double>(t * quot);
        // rounding
        dst[tid] = llround(floatSum);
    }
}

__global__ void hps_decrypt_scale_and_round_kernel_large_lazy(uint64_t *dst, const uint64_t *src,
                                                              const DMulUIntMod *t_QHatInv_mod_q_div_q_mod_t,
                                                              const double *t_QHatInv_mod_q_div_q_frac,
                                                              const DMulUIntMod *t_QHatInv_mod_q_B_div_q_mod_t,
                                                              const double *t_QHatInv_mod_q_B_div_q_frac,
                                                              uint64_t t, size_t n, size_t size_Ql, size_t qMSBHf) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < n;
         tid += blockDim.x * gridDim.x) {
        double floatSum = 0.0;
        uint64_t intSum = 0;
        uint64_t tmpLo, tmpHi;
        double tInv = 1. / static_cast<double>(t);

        for (size_t i = 0; i < size_Ql; i++) {
            uint64_t tmp = src[i * n + tid];
            tmpHi = tmp >> qMSBHf;
            tmpLo = tmp & ((1ULL << qMSBHf) - 1);
            floatSum += static_cast<double>(tmpLo) * t_QHatInv_mod_q_div_q_frac[i];
            floatSum += static_cast<double>(tmpHi) * t_QHatInv_mod_q_B_div_q_frac[i];
            intSum += tmpLo * t_QHatInv_mod_q_div_q_mod_t[i].operand();
            intSum += tmpHi * t_QHatInv_mod_q_B_div_q_mod_t[i].operand();
        }
        // compute modulo reduction by finding the quotient using doubles
        // and then subtracting quotient * t
        floatSum += static_cast<double>(intSum);
        auto quot = static_cast<uint64_t>(floatSum * tInv);
        floatSum -= static_cast<double>(t * quot);
        // rounding
        dst[tid] = llround(floatSum);
    }
}

void DRNSTool::hps_decrypt_scale_and_round(uint64_t *dst, const uint64_t *src) const {
    uint64_t gridDimGlb = n_ / blockDimGlb.x;
    uint64_t t = t_.value();
    size_t n = n_;
    size_t size_Ql = base_Ql_.size();

    // We try to keep floating point error of
    // \sum x_i*tQHatInvModqDivqFrac[i] small.
    if (qMSB_ + sizeQMSB_ < 52) {
        // In our settings x_i <= q_i/2 and for double type floating point
        // error is bounded by 2^{-53}. Thus the floating point error is bounded
        // by sizeQ * q_i/2 * 2^{-53}. In case of qMSB + sizeQMSB < 52 the error
        // is bounded by 1/4, and the rounding will be correct.
        if ((qMSB_ + tMSB_ + sizeQMSB_) < 52) {
            // No intermediate modulo reductions are needed in this case
            // we fit in 52 bits, so we can do multiplications and
            // additions without modulo reduction, and do modulo reduction
            // only once using floating point techniques
            hps_decrypt_scale_and_round_kernel_small_lazy<<<gridDimGlb, blockDimGlb>>>(
                    dst, src,
                    t_QHatInv_mod_q_div_q_mod_t_.get(),
                    t_QHatInv_mod_q_div_q_frac_.get(),
                    t, n, size_Ql);
        } else {
            // In case of qMSB + sizeQMSB >= 52 we decompose x_i in the basis
            // B=2^{qMSB/2} And split the sum \sum x_i*tQHatInvModqDivqFrac[i] to
            // the sum \sum xLo_i*tQHatInvModqDivqFrac[i] +
            // xHi_i*tQHatInvModqBDivqFrac[i] with also precomputed
            // tQHatInvModqBDivqFrac = Frac{t*QHatInv_i*B/q_i} In our settings q_i <
            // 2^60, so xLo_i, xHi_i < 2^30 and for double type floating point error
            // is bounded by 2^{-53}. Thus the floating point error is bounded by
            // sizeQ * 2^30 * 2^{-53}. We always have sizeQ < 2^11, which means the
            // error is bounded by 1/4, and the rounding will be correct.
            // only once using floating point techniques
            hps_decrypt_scale_and_round_kernel_small<<<gridDimGlb, blockDimGlb>>>(
                    dst, src,
                    t_QHatInv_mod_q_div_q_mod_t_.get(),
                    t_QHatInv_mod_q_div_q_frac_.get(),
                    t, n, size_Ql);
        }
    } else { // qMSB_ + sizeQMSB_ >= 52
        size_t qMSBHf = qMSB_ >> 1;
        if ((qMSBHf + tMSB_ + sizeQMSB_) < 52) {
            // No intermediate modulo reductions are needed in this case
            // we fit in 52 bits, so we can do multiplications and
            // additions without modulo reduction, and do modulo reduction
            // only once using floating point techniques
            hps_decrypt_scale_and_round_kernel_large_lazy<<<gridDimGlb, blockDimGlb>>>(
                    dst, src,
                    t_QHatInv_mod_q_div_q_mod_t_.get(),
                    t_QHatInv_mod_q_div_q_frac_.get(),
                    t_QHatInv_mod_q_B_div_q_mod_t_.get(),
                    t_QHatInv_mod_q_B_div_q_frac_.get(),
                    t, n, size_Ql, qMSBHf);
        } else {
            hps_decrypt_scale_and_round_kernel_large<<<gridDimGlb, blockDimGlb>>>(
                    dst, src,
                    t_QHatInv_mod_q_div_q_mod_t_.get(),
                    t_QHatInv_mod_q_div_q_frac_.get(),
                    t_QHatInv_mod_q_B_div_q_mod_t_.get(),
                    t_QHatInv_mod_q_B_div_q_frac_.get(),
                    t, n, size_Ql, qMSBHf);
        }
    }
}

__device__ inline bool is64BitOverflow(double d) {
    // std::numeric_limits<double>::epsilon();
    constexpr double epsilon = 0.000001;
    // std::nextafter(static_cast<double>(std::numeric_limits<int64_t>::max()), 0.0);
    constexpr int64_t safe_double = 9223372036854775295;
    return ((std::abs(d) - static_cast<double>(safe_double)) > epsilon);
}

// QR -> R
__global__ void scaleAndRound_HPS_QR_R_kernel(uint64_t *dst, const uint64_t *src,
                                              const DMulUIntMod *t_R_SHatInv_mod_s_div_s_mod_r,
                                              const double *t_R_SHatInv_mod_s_div_s_frac,
                                              const DModulus *base_Rl,
                                              size_t n, size_t size_Ql, size_t size_Rl) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < n;
         tid += blockDim.x * gridDim.x) {
        auto src_Ql = src;
        auto src_Rl = src + size_Ql * n;

        double nu = 0.5;
        for (size_t i = 0; i < size_Ql; i++) {
            uint64_t xi = src_Ql[i * n + tid];
            nu += static_cast<double>(xi) * t_R_SHatInv_mod_s_div_s_frac[i];
        }

//        if (!is64BitOverflow(nu)) {
        auto alpha = static_cast<uint64_t>(nu);

        for (size_t j = 0; j < size_Rl; j++) {
            uint128_t curValue = {0, 0};
            auto rj = base_Rl[j].value();
            auto rj_ratio = base_Rl[j].const_ratio();
            auto t_R_SHatInv_mod_s_div_s_mod_rj = t_R_SHatInv_mod_s_div_s_mod_r + j * (size_Ql + 1);

            for (size_t i = 0; i < size_Ql; i++) {
                uint64_t xi = src_Ql[i * n + tid];
                uint128_t temp = multiply_uint64_uint64(xi, t_R_SHatInv_mod_s_div_s_mod_rj[i].operand());
                add_uint128_uint128(temp, curValue, curValue);
            }

            uint64_t xi = src_Rl[j * n + tid];
            uint128_t temp = multiply_uint64_uint64(xi, t_R_SHatInv_mod_s_div_s_mod_rj[size_Ql].operand());
            add_uint128_uint128(temp, curValue, curValue);

            uint64_t curNativeValue = barrett_reduce_uint128_uint64(curValue, rj, rj_ratio);
            alpha = barrett_reduce_uint64_uint64(alpha, rj, rj_ratio[1]);
            dst[j * n + tid] = add_uint64_uint64_mod(curNativeValue, alpha, rj);
        }
//        } else {
//            int exp;
//            double mant = std::frexp(nu, &exp);
//            auto mantissa = static_cast<uint64_t>(mant * (1ULL << 53));
//            auto exponent = static_cast<uint64_t>(1ULL << (exp - 53));
//            uint128_t alpha = multiply_uint64_uint64(mantissa, exponent);
//
//            for (size_t j = 0; j < size_Rl; j++) {
//                uint128_t curValue = {0, 0};
//                auto rj = base_Rl[j].value();
//                auto rj_ratio = base_Rl[j].const_ratio();
//                auto t_R_SHatInv_mod_s_div_s_mod_rj = t_R_SHatInv_mod_s_div_s_mod_r + j * (size_Ql + 1);
//
//                for (size_t i = 0; i < size_Ql; i++) {
//                    uint64_t xi = src_Ql[i * n + tid];
//                    uint128_t temp = multiply_uint64_uint64(xi, t_R_SHatInv_mod_s_div_s_mod_rj[i].operand());
//                    add_uint128_uint128(temp, curValue, curValue);
//                }
//
//                uint64_t xi = src_Rl[j * n + tid];
//                uint128_t temp = multiply_uint64_uint64(xi, t_R_SHatInv_mod_s_div_s_mod_rj[size_Ql].operand());
//                add_uint128_uint128(temp, curValue, curValue);
//
//                uint64_t curNativeValue = barrett_reduce_uint128_uint64(curValue, rj, rj_ratio);
//                uint64_t alphaNativeValue = barrett_reduce_uint128_uint64(alpha, rj, rj_ratio);
//                dst[j * n + tid] = add_uint64_uint64_mod(curNativeValue, alphaNativeValue, rj);
//            }
//        }
    }
}

void DRNSTool::scaleAndRound_HPS_QR_R(uint64_t *dst, const uint64_t *src) const {
    uint64_t gridDimGlb = n_ / blockDimGlb.x;
    size_t n = n_;
    size_t size_Ql = base_Ql_.size();
    size_t size_Rl = base_Rl_.size();
    scaleAndRound_HPS_QR_R_kernel<<<gridDimGlb, blockDimGlb>>>(
            dst, src,
            tRSHatInvModsDivsModr(),
            tRSHatInvModsDivsFrac(),
            base_Rl_.base(),
            n, size_Ql, size_Rl);
}

// QlRl -> Ql
__global__ void scaleAndRound_HPS_QlRl_Ql_kernel(uint64_t *dst, const uint64_t *src,
                                                 const DMulUIntMod *tQlSlHatInvModsDivsModq,
                                                 const double *tQlSlHatInvModsDivsFrac,
                                                 const DModulus *base_Ql,
                                                 size_t n, size_t size_Ql, size_t size_Rl) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < n;
         tid += blockDim.x * gridDim.x) {
        auto src_Ql = src;
        auto src_Rl = src + size_Ql * n;

        double nu = 0.5;
        for (size_t j = 0; j < size_Rl; j++) {
            uint64_t xj = src_Rl[j * n + tid];
            nu += static_cast<double>(xj) * tQlSlHatInvModsDivsFrac[j];
        }

//        if (!is64BitOverflow(nu)) {
        auto alpha = static_cast<uint64_t>(nu);

        for (size_t i = 0; i < size_Ql; i++) {
            uint128_t curValue = {0, 0};

            auto tQlSlHatInvModsDivsModqi = tQlSlHatInvModsDivsModq + i * (size_Rl + 1);

            for (size_t j = 0; j < size_Rl; j++) {
                uint64_t xj = src_Rl[j * n + tid];
                uint128_t temp = multiply_uint64_uint64(xj, tQlSlHatInvModsDivsModqi[j].operand());
                add_uint128_uint128(temp, curValue, curValue);
            }

            uint64_t xi = src_Ql[i * n + tid];
            uint128_t temp = multiply_uint64_uint64(xi, tQlSlHatInvModsDivsModqi[size_Rl].operand());
            add_uint128_uint128(temp, curValue, curValue);

            auto qi = base_Ql[i].value();
            auto qi_ratio = base_Ql[i].const_ratio();
            uint64_t curNativeValue = barrett_reduce_uint128_uint64(curValue, qi, qi_ratio);
            alpha = barrett_reduce_uint64_uint64(alpha, qi, qi_ratio[1]);
            dst[i * n + tid] = add_uint64_uint64_mod(curNativeValue, alpha, qi);
        }
//        } else {
//            int exp;
//            double mant = std::frexp(nu, &exp);
//            auto mantissa = static_cast<uint64_t>(mant * (1ULL << 53));
//            auto exponent = static_cast<uint64_t>(1ULL << (exp - 53));
//            uint128_t alpha = multiply_uint64_uint64(mantissa, exponent);
//
//            for (size_t i = 0; i < size_Ql; i++) {
//                uint128_t curValue = {0, 0};
//
//                auto tQlSlHatInvModsDivsModqi = tQlSlHatInvModsDivsModq + i * (size_Rl + 1);
//
//                for (size_t j = 0; j < size_Rl; j++) {
//                    uint64_t xj = src_Rl[j * n + tid];
//                    uint128_t temp = multiply_uint64_uint64(xj, tQlSlHatInvModsDivsModqi[j].operand());
//                    add_uint128_uint128(temp, curValue, curValue);
//                }
//
//                uint64_t xi = src_Ql[i * n + tid];
//                uint128_t temp = multiply_uint64_uint64(xi, tQlSlHatInvModsDivsModqi[size_Rl].operand());
//                add_uint128_uint128(temp, curValue, curValue);
//
//                auto qi = base_Ql[i].value();
//                auto qi_ratio = base_Ql[i].const_ratio();
//                uint64_t curNativeValue = barrett_reduce_uint128_uint64(curValue, qi, qi_ratio);
//                uint64_t alphaNativeValue = barrett_reduce_uint128_uint64(alpha, qi, qi_ratio);
//                dst[i * n + tid] = add_uint64_uint64_mod(curNativeValue, alphaNativeValue, qi);
//            }
//        }
    }
}

void DRNSTool::scaleAndRound_HPS_QlRl_Ql(uint64_t *dst, const uint64_t *src) const {
    uint64_t gridDimGlb = n_ / blockDimGlb.x;
    size_t n = n_;
    size_t size_Ql = base_Ql_.size();
    size_t size_Rl = base_Rl_.size();
    scaleAndRound_HPS_QlRl_Ql_kernel<<<gridDimGlb, blockDimGlb>>>(
            dst, src,
            tQlSlHatInvModsDivsModq(),
            tQlSlHatInvModsDivsFrac(),
            base_Ql_.base(),
            n, size_Ql, size_Rl);
}

// reuse scaleAndRound_HPS_QlRl_Ql_kernel
void DRNSTool::scaleAndRound_HPS_Q_Ql(uint64_t *dst, const uint64_t *src) const {
    uint64_t gridDimGlb = n_ / blockDimGlb.x;
    size_t n = n_;
    size_t size_Ql = base_Ql_.size();
    size_t size_QlDrop = base_QlDrop_.size();
    scaleAndRound_HPS_QlRl_Ql_kernel<<<gridDimGlb, blockDimGlb>>>(
            dst, src,
            QlQHatInvModqDivqModq(),
            QlQHatInvModqDivqFrac(),
            base_Ql_.base(),
            n, size_Ql, size_QlDrop);
}

/*
 * dst = src * scale % base
 */
__global__ void ExpandCRTBasisQlHat_kernel(uint64_t *out, const uint64_t *in,
                                           const DMulUIntMod *QlDropModq,
                                           const DModulus *base_Ql, size_t size_Ql, size_t size_Q,
                                           uint64_t n) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < n * size_Q;
         tid += blockDim.x * gridDim.x) {
        size_t i = tid / n;
        if (i < size_Ql) {
            auto modulus = base_Ql[i].value();
            out[tid] = multiply_and_reduce_shoup(in[tid], QlDropModq[i], modulus);
        } else {
            out[tid] = 0;
        }
    }
}

void DRNSTool::ExpandCRTBasis_Ql_Q(uint64_t *dst, const uint64_t *src) const {
    size_t size_Ql = base_Ql_.size();
    size_t size_Q = base_Q_.size();

    const DMulUIntMod *QlDropModq = base_Ql_to_QlDrop_conv_.PModq();

    size_t n = n_;
    uint64_t gridDimGlb = n * size_Q / blockDimGlb.x;
    ExpandCRTBasisQlHat_kernel<<<gridDimGlb, blockDimGlb>>>(
            dst, src,
            QlDropModq,
            base_Ql_.base(), size_Ql, size_Q,
            n);
}

__global__ void ExpandCRTBasisQlHat_add_to_ct_kernel(uint64_t *out, const uint64_t *in,
                                                     const DMulUIntMod *QlDropModq,
                                                     const DModulus *base_Ql, size_t size_Ql,
                                                     uint64_t n) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < n * size_Ql;
         tid += blockDim.x * gridDim.x) {
        size_t i = tid / n;
        auto modulus = base_Ql[i].value();
        uint64_t tmp = multiply_and_reduce_shoup(in[tid], QlDropModq[i], modulus);
        out[tid] = add_uint64_uint64_mod(tmp, out[tid], modulus);
    }
}

void DRNSTool::ExpandCRTBasis_Ql_Q_add_to_ct(uint64_t *dst, const uint64_t *src) const {
    size_t size_Ql = base_Ql_.size();
    size_t size_Q = base_Q_.size();

    const DMulUIntMod *QlDropModq = base_Ql_to_QlDrop_conv_.PModq();

    size_t n = n_;
    uint64_t gridDimGlb = n * size_Ql / blockDimGlb.x;
    ExpandCRTBasisQlHat_add_to_ct_kernel<<<gridDimGlb, blockDimGlb>>>(
            dst, src,
            QlDropModq,
            base_Ql_.base(), size_Ql,
            n);
}
