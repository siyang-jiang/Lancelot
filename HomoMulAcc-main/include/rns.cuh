#pragma once

#include "util/rns.h"
#include "gputype.h"
#include "polymath.cuh"
#include "util/encryptionparams.h"
#include "rns_base.cuh"
#include "rns_bconv.cuh"

typedef struct DRNSTool {
    cahel::mul_tech_type mul_tech_ = cahel::mul_tech_type::behz;

    std::size_t n_ = 0;
    std::size_t size_QP_ = 0;
    std::size_t size_P_ = 0;

    DRNSBase base_;
    DRNSBase base_Q_;
    DRNSBase base_Ql_;
    DRNSBase base_QlP_;
    // q[last]^(-1) mod q[i] for i = 0..last-1
    cahel::util::Pointer<DMulUIntMod> inv_q_last_mod_q_;

    // hybrid key-switching
    cahel::util::Pointer<DMulUIntMod> bigP_mod_q_;
    cahel::util::Pointer<DMulUIntMod> bigPInv_mod_q_;
    cahel::util::Pointer<DMulUIntMod> partQlHatInv_mod_Ql_concat_;
    std::vector<DBaseConverter> v_base_part_Ql_to_compl_part_QlP_conv_{};
    DBaseConverter base_P_to_Ql_conv_;

    // plain modulus related (BFV/BGV)
    DModulus t_;
    uint64_t q_last_mod_t_ = 1;
    uint64_t inv_q_last_mod_t_ = 1;
    // Base converter: q --> t
    DBaseConverter base_q_to_t_conv_;

    // BGV correction factor
    cahel::util::Pointer<DMulUIntMod> pjInv_mod_q_;
    cahel::util::Pointer<DMulUIntMod> pjInv_mod_t_;
    DMulUIntMod bigPInv_mod_t_;
    DBaseConverter base_P_to_t_conv_;

    // BFV enc/add/sub
    DMulUIntMod negQl_mod_t_; // Ql mod t
    cahel::util::Pointer<DMulUIntMod> tInv_mod_q_; // t^(-1) mod q

    // BFV BEHZ
    DRNSBase base_B_;
    DRNSBase base_Bsk_;
    DRNSBase base_Bsk_m_tilde_;
    DRNSBase base_t_gamma_;
    DModulus m_tilde_;
    DModulus m_sk_;
    DModulus gamma_;
    DNTTTable gpu_Bsk_tables_;
    cahel::util::Pointer<DMulUIntMod> tModBsk_;
    // Base converter: q --> B_sk
    DBaseConverter base_q_to_Bsk_conv_;
    // Base converter: q --> {m_tilde}
    DBaseConverter base_q_to_m_tilde_conv_;
    // Base converter: B --> q
    DBaseConverter base_B_to_q_conv_;
    // Base converter: B --> {m_sk}
    DBaseConverter base_B_to_m_sk_conv_;
    // Base converter: q --> {t, gamma}
    DBaseConverter base_q_to_t_gamma_conv_;
    // prod(q)^(-1) mod Bsk
    cahel::util::Pointer<DMulUIntMod> inv_prod_q_mod_Bsk_;
    // prod(q)^(-1) mod m_tilde
    DMulUIntMod neg_inv_prod_q_mod_m_tilde_;
    // prod(B)^(-1) mod m_sk
    DMulUIntMod inv_prod_B_mod_m_sk_;
    // gamma^(-1) mod t
    DMulUIntMod inv_gamma_mod_t_;
    // prod(B) mod q
    cahel::util::Pointer<uint64_t> prod_B_mod_q_;
    // m_tilde_QHatInvModq
    cahel::util::Pointer<DMulUIntMod> m_tilde_QHatInvModq_;
    // m_tilde^(-1) mod Bsk
    cahel::util::Pointer<DMulUIntMod> inv_m_tilde_mod_Bsk_;
    // prod(q) mod Bsk
    cahel::util::Pointer<uint64_t> prod_q_mod_Bsk_;
    // -prod(q)^(-1) mod {t, gamma}
    cahel::util::Pointer<DMulUIntMod> neg_inv_q_mod_t_gamma_;
    // prod({t, gamma}) mod q
    cahel::util::Pointer<DMulUIntMod> prod_t_gamma_mod_q_;

    // BFV HPS
    // decrypt scale&round
    size_t qMSB_ = 0;
    size_t sizeQMSB_ = 0;
    size_t tMSB_ = 0;
    cahel::util::Pointer<DMulUIntMod> t_QHatInv_mod_q_div_q_mod_t_;
    cahel::util::Pointer<double> t_QHatInv_mod_q_div_q_frac_;
    cahel::util::Pointer<DMulUIntMod> t_QHatInv_mod_q_B_div_q_mod_t_;
    cahel::util::Pointer<double> t_QHatInv_mod_q_B_div_q_frac_;
    // multiply
    DRNSBase base_Rl_;
    DRNSBase base_QlRl_;
    DRNSBase base_QlDrop_;
    DNTTTable gpu_QlRl_tables_;
    DBaseConverter base_Ql_to_Rl_conv_;
    DBaseConverter base_Rl_to_Ql_conv_;
    DBaseConverter base_Q_to_Rl_conv_;
    DBaseConverter base_Ql_to_QlDrop_conv_;
    cahel::util::Pointer<double> tRSHatInvModsDivsFrac_;
    cahel::util::Pointer<DMulUIntMod> tRSHatInvModsDivsModr_;
    cahel::util::Pointer<double> tQlSlHatInvModsDivsFrac_;
    cahel::util::Pointer<DMulUIntMod> tQlSlHatInvModsDivsModq_;
    cahel::util::Pointer<double> QlQHatInvModqDivqFrac_;
    cahel::util::Pointer<DMulUIntMod> QlQHatInvModqDivqModq_;

    DRNSTool() {
        inv_prod_q_mod_Bsk_ = cahel::util::Pointer<DMulUIntMod>();
        prod_B_mod_q_ = cahel::util::Pointer<uint64_t>();
        m_tilde_QHatInvModq_ = cahel::util::Pointer<DMulUIntMod>();
        inv_m_tilde_mod_Bsk_ = cahel::util::Pointer<DMulUIntMod>();
        prod_q_mod_Bsk_ = cahel::util::Pointer<uint64_t>();
        neg_inv_q_mod_t_gamma_ = cahel::util::Pointer<DMulUIntMod>();
        prod_t_gamma_mod_q_ = cahel::util::Pointer<DMulUIntMod>();
        inv_q_last_mod_q_ = cahel::util::Pointer<DMulUIntMod>();

        // hybrid key-switching
        bigP_mod_q_ = cahel::util::Pointer<DMulUIntMod>();
        bigPInv_mod_q_ = cahel::util::Pointer<DMulUIntMod>();
        partQlHatInv_mod_Ql_concat_ = cahel::util::Pointer<DMulUIntMod>();
        pjInv_mod_q_ = cahel::util::Pointer<DMulUIntMod>();
        pjInv_mod_t_ = cahel::util::Pointer<DMulUIntMod>();

        // BFV enc/add/sub
        tInv_mod_q_ = cahel::util::Pointer<DMulUIntMod>();

        // HPS decrypt scale&round
        t_QHatInv_mod_q_div_q_mod_t_ = cahel::util::Pointer<DMulUIntMod>();
        t_QHatInv_mod_q_div_q_frac_ = cahel::util::Pointer<double>();
        t_QHatInv_mod_q_B_div_q_mod_t_ = cahel::util::Pointer<DMulUIntMod>();
        t_QHatInv_mod_q_B_div_q_frac_ = cahel::util::Pointer<double>();

        // HPS multiply scale&round
        tRSHatInvModsDivsFrac_ = cahel::util::Pointer<double>();
        tRSHatInvModsDivsModr_ = cahel::util::Pointer<DMulUIntMod>();
        tQlSlHatInvModsDivsFrac_ = cahel::util::Pointer<double>();
        tQlSlHatInvModsDivsModq_ = cahel::util::Pointer<DMulUIntMod>();
        QlQHatInvModqDivqFrac_ = cahel::util::Pointer<double>();
        QlQHatInvModqDivqModq_ = cahel::util::Pointer<DMulUIntMod>();
    }

    DRNSTool(DRNSTool &source) {
        inv_prod_q_mod_Bsk_.acquire(source.inv_prod_q_mod_Bsk_);
        prod_B_mod_q_.acquire(source.prod_B_mod_q_);
        m_tilde_QHatInvModq_.acquire(source.m_tilde_QHatInvModq_);
        inv_m_tilde_mod_Bsk_.acquire(source.inv_m_tilde_mod_Bsk_);
        prod_q_mod_Bsk_.acquire(source.prod_q_mod_Bsk_);
        neg_inv_q_mod_t_gamma_.acquire(source.neg_inv_q_mod_t_gamma_);
        prod_t_gamma_mod_q_.acquire(source.prod_t_gamma_mod_q_);
        inv_q_last_mod_q_.acquire(source.inv_q_last_mod_q_);

        // hybrid key-switching
        bigP_mod_q_.acquire(source.bigP_mod_q_);
        bigPInv_mod_q_.acquire(source.bigPInv_mod_q_);
        partQlHatInv_mod_Ql_concat_.acquire(source.partQlHatInv_mod_Ql_concat_);
        pjInv_mod_q_.acquire(source.pjInv_mod_q_);
        pjInv_mod_t_.acquire(source.pjInv_mod_t_);

        // BFV enc/add/sub
        tInv_mod_q_.acquire(source.tInv_mod_q_);

        // HPS decrypt scale&round
        t_QHatInv_mod_q_div_q_mod_t_.acquire(source.t_QHatInv_mod_q_div_q_mod_t_);
        t_QHatInv_mod_q_div_q_frac_.acquire(source.t_QHatInv_mod_q_div_q_frac_);
        t_QHatInv_mod_q_B_div_q_mod_t_.acquire(source.t_QHatInv_mod_q_B_div_q_mod_t_);
        t_QHatInv_mod_q_B_div_q_frac_.acquire(source.t_QHatInv_mod_q_B_div_q_frac_);

        // HPS multiply scale&round
        tRSHatInvModsDivsFrac_.acquire(source.tRSHatInvModsDivsFrac_);
        tRSHatInvModsDivsModr_.acquire(source.tRSHatInvModsDivsModr_);
        tQlSlHatInvModsDivsFrac_.acquire(source.tQlSlHatInvModsDivsFrac_);
        tQlSlHatInvModsDivsModq_.acquire(source.tQlSlHatInvModsDivsModq_);
        QlQHatInvModqDivqFrac_.acquire(source.QlQHatInvModqDivqFrac_);
        QlQHatInvModqDivqModq_.acquire(source.QlQHatInvModqDivqModq_);
    }

    DRNSTool(DRNSTool &&source) noexcept {
        inv_prod_q_mod_Bsk_.acquire(source.inv_prod_q_mod_Bsk_);
        prod_B_mod_q_.acquire(source.prod_B_mod_q_);
        m_tilde_QHatInvModq_.acquire(source.m_tilde_QHatInvModq_);
        inv_m_tilde_mod_Bsk_.acquire(source.inv_m_tilde_mod_Bsk_);
        prod_q_mod_Bsk_.acquire(source.prod_q_mod_Bsk_);
        neg_inv_q_mod_t_gamma_.acquire(source.neg_inv_q_mod_t_gamma_);
        prod_t_gamma_mod_q_.acquire(source.prod_t_gamma_mod_q_);
        inv_q_last_mod_q_.acquire(source.inv_q_last_mod_q_);

        // hybrid key-switching
        bigP_mod_q_.acquire(source.bigP_mod_q_);
        bigPInv_mod_q_.acquire(source.bigPInv_mod_q_);
        partQlHatInv_mod_Ql_concat_.acquire(source.partQlHatInv_mod_Ql_concat_);
        pjInv_mod_q_.acquire(source.pjInv_mod_q_);
        pjInv_mod_t_.acquire(source.pjInv_mod_t_);

        // BFV enc/add/sub
        tInv_mod_q_.acquire(source.tInv_mod_q_);

        // HPS decrypt divide&round
        t_QHatInv_mod_q_div_q_mod_t_.acquire(source.t_QHatInv_mod_q_div_q_mod_t_);
        t_QHatInv_mod_q_div_q_frac_.acquire(source.t_QHatInv_mod_q_div_q_frac_);
        t_QHatInv_mod_q_B_div_q_mod_t_.acquire(source.t_QHatInv_mod_q_B_div_q_mod_t_);
        t_QHatInv_mod_q_B_div_q_frac_.acquire(source.t_QHatInv_mod_q_B_div_q_frac_);

        // HPS multiply scale&round
        tRSHatInvModsDivsFrac_.acquire(source.tRSHatInvModsDivsFrac_);
        tRSHatInvModsDivsModr_.acquire(source.tRSHatInvModsDivsModr_);
        tQlSlHatInvModsDivsFrac_.acquire(source.tQlSlHatInvModsDivsFrac_);
        tQlSlHatInvModsDivsModq_.acquire(source.tQlSlHatInvModsDivsModq_);
        QlQHatInvModqDivqFrac_.acquire(source.QlQHatInvModqDivqFrac_);
        QlQHatInvModqDivqModq_.acquire(source.QlQHatInvModqDivqModq_);
    }

    ~DRNSTool() = default;

    void init(const cahel::util::RNSTool &cpu_rns_tool);

    void modup(uint64_t *dst,
               const uint64_t *cks,
               const DNTTTable &ntt_tables,
               const cahel::scheme_type &scheme) const;

    void moddown(uint64_t *ct_i,
                 uint64_t *cx_i,
                 const DNTTTable &ntt_tables,
                 const cahel::scheme_type &scheme) const;

    void moddown_from_NTT(uint64_t *ct_i,
                          uint64_t *cx_i,
                          const DNTTTable &ntt_tables,
                          const cahel::scheme_type &scheme) const;

    void behz_decrypt_scale_and_round(uint64_t *src, uint64_t *temp, const DNTTTable &rns_table, uint64_t temp_mod_size,
                                      uint64_t poly_modulus_degree, uint64_t *dst) const;

    void hps_decrypt_scale_and_round(uint64_t *dst, const uint64_t *src) const;

    void scaleAndRound_HPS_QR_R(uint64_t *dst, const uint64_t *src) const;
    void scaleAndRound_HPS_QlRl_Ql(uint64_t *dst, const uint64_t *src) const;
    void scaleAndRound_HPS_Q_Ql(uint64_t *dst, const uint64_t *src) const;

    void ExpandCRTBasis_Ql_Q(uint64_t *dst, const uint64_t *src) const;
    void ExpandCRTBasis_Ql_Q_add_to_ct(uint64_t *dst, const uint64_t *src) const;

    void divide_and_round_q_last(const uint64_t *src, size_t cipher_size, uint64_t *dst) const;

    void divide_and_round_q_last_ntt(const uint64_t *src, size_t cipher_size, const DNTTTable &rns_tables,
                                     uint64_t *dst) const;

    /**
    Compute mod t
    **/
    void decrypt_mod_t(uint64_t *dst, const uint64_t *src, const uint64_t poly_degree) const;

    // BEHZ step 1: Convert from base q to base Bsk U {m_tilde}
    void fastbconv_m_tilde(uint64_t *dst, uint64_t *src) const;

    // BEHZ step 2: Reduce q-overflows in with Montgomery reduction, switching base to Bsk
    void sm_mrq(uint64_t *dst, const uint64_t *src) const;

    // BEHZ step 7: divide by q and floor, producing a result in base Bsk
    void fast_floor(uint64_t *input_base_q, uint64_t *input_base_Bsk, uint64_t *out_base_Bsk) const;

    // BEHZ step 8: use Shenoy-Kumaresan method to convert the result to base q
    void fastbconv_sk(uint64_t *input_base_Bsk, uint64_t *out_base_q) const;

    __host__ __device__ inline DMulUIntMod *inv_prod_q_mod_Bsk() const {
        return (DMulUIntMod *) (inv_prod_q_mod_Bsk_.get());
    }

    __host__ __device__ inline uint64_t *prod_B_mod_q() const {
        return (uint64_t *) (prod_B_mod_q_.get());
    }

    __host__ __device__ inline DMulUIntMod *m_tilde_QHatInvModq() const {
        return (DMulUIntMod *) (m_tilde_QHatInvModq_.get());
    }

    __host__ __device__ inline DMulUIntMod *inv_m_tilde_mod_Bsk() const {
        return (DMulUIntMod *) (inv_m_tilde_mod_Bsk_.get());
    }

    __host__ __device__ inline uint64_t *prod_q_mod_Bsk() const {
        return (uint64_t *) (prod_q_mod_Bsk_.get());
    }

    __host__ __device__ inline DMulUIntMod *neg_inv_q_mod_t_gamma() const {
        return (DMulUIntMod *) (neg_inv_q_mod_t_gamma_.get());
    }

    __host__ __device__ inline DMulUIntMod *prod_t_gamma_mod_q() const {
        return (DMulUIntMod *) (prod_t_gamma_mod_q_.get());
    }

    __host__ __device__ inline DMulUIntMod *inv_q_last_mod_q() const {
        return (DMulUIntMod *) (inv_q_last_mod_q_.get());
    }

    // hybrid key-switching

    __host__ __device__ inline DMulUIntMod *bigP_mod_q() const noexcept {
        return (DMulUIntMod *) (bigP_mod_q_.get());
    }

    __host__ __device__ inline DMulUIntMod *bigPInv_mod_q() const noexcept {
        return (DMulUIntMod *) (bigPInv_mod_q_.get());
    }

    __host__ __device__ inline DMulUIntMod *pjInv_mod_q() const noexcept {
        return (DMulUIntMod *) (pjInv_mod_q_.get());
    }

    __host__ __device__ inline DMulUIntMod *pjInv_mod_t() const noexcept {
        return (DMulUIntMod *) (pjInv_mod_t_.get());
    }

    __host__ __device__ inline auto &v_base_part_Ql_to_compl_part_QlP_conv() const noexcept {
        return v_base_part_Ql_to_compl_part_QlP_conv_;
    }

    __host__ inline auto &base_part_Ql_to_compl_part_QlP_conv(std::size_t index) const noexcept {
        return v_base_part_Ql_to_compl_part_QlP_conv_.at(index);
    }

    __host__ __device__ inline auto &base_P_to_Ql_conv() const noexcept {
        return base_P_to_Ql_conv_;
    }

    __host__ __device__ inline auto &base_P_to_t_conv() const noexcept {
        return base_P_to_t_conv_;
    }

    // HPS

    // decrypt scale and round

    __host__ __device__ inline DMulUIntMod *t_QHatInv_mod_q_div_q_mod_t() const noexcept {
        return (DMulUIntMod *) (t_QHatInv_mod_q_div_q_mod_t_.get());
    }

    __host__ __device__ inline double *t_QHatInv_mod_q_div_q_frac() const noexcept {
        return (double *) (t_QHatInv_mod_q_div_q_frac_.get());
    }

    __host__ __device__ inline DMulUIntMod *t_QHatInv_mod_q_B_div_q_mod_t() const noexcept {
        return (DMulUIntMod *) (t_QHatInv_mod_q_B_div_q_mod_t_.get());
    }

    __host__ __device__ inline double *t_QHatInv_mod_q_B_div_q_frac() const noexcept {
        return (double *) (t_QHatInv_mod_q_B_div_q_frac_.get());
    }

    // multiply scale and round

    inline DNTTTable &gpu_QlRl_tables() {
        return gpu_QlRl_tables_;
    }

    __host__ __device__ inline double *tRSHatInvModsDivsFrac() const noexcept {
        return (double *) (tRSHatInvModsDivsFrac_.get());
    }

    __host__ __device__ inline DMulUIntMod *tRSHatInvModsDivsModr() const noexcept {
        return (DMulUIntMod *) (tRSHatInvModsDivsModr_.get());
    }

    __host__ __device__ inline double *tQlSlHatInvModsDivsFrac() const noexcept {
        return (double *) (tQlSlHatInvModsDivsFrac_.get());
    }

    __host__ __device__ inline DMulUIntMod *tQlSlHatInvModsDivsModq() const noexcept {
        return (DMulUIntMod *) (tQlSlHatInvModsDivsModq_.get());
    }

    __host__ __device__ inline double *QlQHatInvModqDivqFrac() const noexcept {
        return (double *) (QlQHatInvModqDivqFrac_.get());
    }

    __host__ __device__ inline DMulUIntMod *QlQHatInvModqDivqModq() const noexcept {
        return (DMulUIntMod *) (QlQHatInvModqDivqModq_.get());
    }

} DRNSTool;
