#pragma once

#include "encryptionparams.h"
#include "modulus.h"
#include "ntt.h"
#include "uintarithsmallmod.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cahel::util {
    class RNSBase {
    public:
        RNSBase() : size_(0) {}

        // Construct the RNSBase from the parm, calculate
        // 1. the product of all coeff (big_Q_)
        // 2. the product of all coeff except myself (big_qiHat_)
        // 3. the inverse of the above product mod myself (qiHatInv_mod_qi_)
        explicit RNSBase(const std::vector<Modulus> &rnsbase);

        // Copy from the copy RNSBase
        RNSBase(const RNSBase &copy);

        // Move from the source
        RNSBase(RNSBase &&source) = default;

        RNSBase &operator=(const RNSBase &assign) = delete;

        // Get the index coeff modulus
        [[nodiscard]] inline const Modulus &operator[](std::size_t index) const {
            if (index >= size_) {
                throw std::out_of_range("index is out of range");
            }
            return mod_[index];
        }

        // Returns the number of coeff modulus
        [[nodiscard]] inline std::size_t size() const noexcept {
            return size_;
        }

        // Returns whether the specified modulus exists in the Q_
        [[nodiscard]] bool contains(const Modulus &value) const noexcept;

        // Return whether my Q_ is the subset of the provided superbase.Q_
        [[nodiscard]] bool is_subbase_of(const RNSBase &superbase) const noexcept;

        // Return whether the provided superbase.Q_ is a subset of my Q_
        [[nodiscard]] inline bool is_superbase_of(const RNSBase &subbase) const noexcept {
            return subbase.is_subbase_of(*this);
        }

        // A faster check compared with is_subbase_of
        [[maybe_unused]] [[nodiscard]] inline bool is_proper_subbase_of(const RNSBase &superbase) const noexcept {
            return (size_ < superbase.size_) && is_subbase_of(superbase);
        }

        // A faster check compared with is_superbase_of
        [[maybe_unused]] [[nodiscard]] inline bool is_proper_superbase_of(const RNSBase &subbase) const noexcept {
            return (size_ > subbase.size_) && !is_subbase_of(subbase);
        }

        // Add a modulus to my RNSBase
        [[nodiscard]] RNSBase extend(const Modulus &value) const;

        // Add other RNSBase to my RNSBase
        [[nodiscard]] RNSBase extend(const RNSBase &other) const;

        // Delete the last coeff and re-generate the RNSBase
        [[nodiscard]] RNSBase drop() const;

        // Delete the specified Modulus and re-generate the RNSBase
        [[nodiscard]] RNSBase drop(const Modulus &value) const;

        // Delete the specified Moduli and re-generate the RNSBase
        [[nodiscard]] RNSBase drop(const std::vector<Modulus> &values) const;

        // The CRT decompose, i.e., value % each modulus (Q_[0], Q_[1], ...)
        void decompose(std::uint64_t *value) const;

        // When the poly degree is count, perform the CRT in one invocation
        void decompose_array(std::uint64_t *value, std::size_t count) const;

        // CRT compose
        void compose(std::uint64_t *value) const;

        // When the poly degree is count, perform the CRT compose in one invocation
        void compose_array(std::uint64_t *value, std::size_t count) const;

        [[nodiscard]] inline const Modulus *base() const noexcept {
            return mod_.data();
        }

        [[nodiscard]] inline const std::uint64_t *big_modulus() const noexcept {
            return prod_mod_.data();
        }

        [[nodiscard]] inline const std::uint64_t *big_qiHat() const noexcept {
            return prod_hat_.data();
        }

        [[nodiscard]] inline const MultiplyUIntModOperand *qiHat_mod_qi() const noexcept {
            return hat_mod_.data();
        }

        [[nodiscard]] inline const MultiplyUIntModOperand *QHatInvModq() const noexcept {
            return hatInv_mod_.data();
        }

        [[nodiscard]] inline const double *inv() const noexcept {
            return inv_.data();
        }

    private:
        bool initialize();

        // total number of small modulus in this base
        std::size_t size_;
        // vector of small modulus in this base
        std::vector<Modulus> mod_;
        // product of all small modulus in this base, stored in 1d vector
        std::vector<std::uint64_t> prod_mod_;
        // product of all small modulus's hat in this base, stored in 2d vector
        std::vector<std::uint64_t> prod_hat_;
        // vector of qiHat mod qi
        std::vector<MultiplyUIntModOperand> hat_mod_;
        // vector of qiHatInv mod qi
        std::vector<MultiplyUIntModOperand> hatInv_mod_;
        // vector of 1.0 / qi
        std::vector<double> inv_;
    };

    class BaseConverter {
    public:
        BaseConverter(RNSBase ibase, RNSBase obase) : ibase_(std::move(ibase)), obase_(std::move(obase)) {
            initialize();
        }

        BaseConverter(BaseConverter &&source) = delete;

        BaseConverter &operator=(const BaseConverter &assign) = delete;

        BaseConverter &operator=(BaseConverter &&assign) = delete;

        [[nodiscard]] inline std::size_t ibase_size() const noexcept {
            return ibase_.size();
        }

        [[nodiscard]] inline std::size_t obase_size() const noexcept {
            return obase_.size();
        }

        [[nodiscard]] inline const RNSBase &ibase() const noexcept {
            return ibase_;
        }

        [[nodiscard]] inline const RNSBase &obase() const noexcept {
            return obase_;
        }

        [[nodiscard]] inline std::uint64_t *QHatModp(size_t index) {
            if (index >= obase_size())
                throw std::out_of_range("QHatModp index is out of range");

            return QHatModp_[index].data();
        }

        [[nodiscard]] inline std::uint64_t *alphaQModp(size_t index) {
            if (index >= ibase_size() + 1)
                throw std::out_of_range("alphaQModp index is out of range");

            return alphaQModp_[index].data();
        }

        [[nodiscard]] inline auto *negPQHatInvModq() {
            return negPQHatInvModq_.data();
        }

        [[nodiscard]] inline std::uint64_t *QInvModp(size_t index) {
            if (index >= obase_size())
                throw std::out_of_range("QInvModp index is out of range");

            return QInvModp_[index].data();
        }

        [[nodiscard]] inline auto *PModq() {
            return PModq_.data();
        }

    private:
        void initialize();

        RNSBase ibase_;
        RNSBase obase_;
        std::vector<std::vector<std::uint64_t>> QHatModp_;
        std::vector<std::vector<std::uint64_t>> alphaQModp_;
        std::vector<MultiplyUIntModOperand> negPQHatInvModq_;
        std::vector<std::vector<std::uint64_t>> QInvModp_;
        std::vector<MultiplyUIntModOperand> PModq_;
    };

    class RNSTool {
    public:
        /**
        @throws std::invalid_argument if poly_modulus_degree is out of range, coeff_modulus is not valid, or pool is
        invalid.
        @throws std::logic_error if coeff_modulus and extended bases do not support NTT or are not coprime.
        */
        RNSTool(size_t n, size_t size_P, const RNSBase &base_Ql,
                const std::vector<Modulus> &modulus_QP, const Modulus &t, mul_tech_type mul_tech);

        RNSTool(const RNSTool &copy) = delete;

        RNSTool(RNSTool &&source) = delete;

        RNSTool &operator=(const RNSTool &assign) = delete;

        RNSTool &operator=(RNSTool &&assign) = delete;

        [[nodiscard]] inline auto mul_tech() const noexcept {
            return mul_tech_;
        }

        [[nodiscard]] inline auto is_key_level() const noexcept {
            return is_key_level_;
        }

        [[nodiscard]] inline auto coeff_count() const noexcept {
            return n_;
        }

        [[nodiscard]] inline auto total_modulus_size() const noexcept {
            return size_QP_;
        }

        [[nodiscard]] inline auto special_modulus_size() const noexcept {
            return size_P_;
        }

        [[nodiscard]] inline auto &negQl_mod_t() const noexcept {
            return negQl_mod_t_;
        }

        [[nodiscard]] inline auto &tInv_mod_q() const noexcept {
            return tInv_mod_q_;
        }

        [[nodiscard]] inline auto base_q_to_t_gamma_conv() const noexcept {
            return base_q_to_t_gamma_conv_;
        }

        [[nodiscard]] inline auto base_q_to_t_conv() const noexcept {
            return base_q_to_t_conv_;
        }

        [[nodiscard]] inline auto base_q_to_Bsk_conv() const noexcept {
            return base_q_to_Bsk_conv_;
        }

        [[nodiscard]] inline auto base_q_to_m_tilde_conv() const noexcept {
            return base_q_to_m_tilde_conv_;
        }

        [[nodiscard]] inline auto base_B_to_q_conv() const noexcept {
            return base_B_to_q_conv_;
        }

        [[nodiscard]] inline auto base_B_to_m_sk_conv() const noexcept {
            return base_B_to_m_sk_conv_;
        }

        [[nodiscard]] inline auto &prod_t_gamma_mod_q() const noexcept {
            return prod_t_gamma_mod_q_;
        }

        [[nodiscard]] inline auto &neg_inv_q_mod_t_gamma() const noexcept {
            return neg_inv_q_mod_t_gamma_;
        }

        [[nodiscard]] inline auto &inv_gamma_mod_t() const noexcept {
            return inv_gamma_mod_t_;
        }

        [[nodiscard]] inline auto &neg_inv_prod_q_mod_m_tilde() const noexcept {
            return neg_inv_prod_q_mod_m_tilde_;
        }

        [[nodiscard]] inline auto inv_q_last_mod_q() const noexcept {
            return inv_q_last_mod_q_.data();
        }

        [[nodiscard]] inline auto &base_Bsk_ntt_tables() const noexcept {
            return base_Bsk_ntt_tables_;
        }

        [[nodiscard]] inline auto &m_tilde_QHatInvModq() const noexcept {
            return m_tilde_QHatInvModq_;
        }

        [[nodiscard]] inline auto &tModBsk() const noexcept {
            return tModBsk_;
        }

        [[nodiscard]] inline auto base() const noexcept {
            return base_.get();
        }

        [[nodiscard]] inline auto base_Q() const noexcept {
            return base_Q_.get();
        }

        [[nodiscard]] inline auto base_Ql() const noexcept {
            return base_Ql_.get();
        }

        [[nodiscard]] inline auto base_QlP() const noexcept {
            return base_QlP_.get();
        }

        [[nodiscard]] inline auto base_B() const noexcept {
            return base_B_.get();
        }

        [[nodiscard]] inline auto base_Bsk() const noexcept {
            return base_Bsk_.get();
        }

        [[nodiscard]] inline auto base_Bsk_m_tilde() const noexcept {
            return base_Bsk_m_tilde_.get();
        }

        [[nodiscard]] inline auto base_t_gamma() const noexcept {
            return base_t_gamma_.get();
        }

        [[nodiscard]] inline auto &m_tilde() const noexcept {
            return m_tilde_;
        }

        [[nodiscard]] inline auto &m_sk() const noexcept {
            return m_sk_;
        }

        [[nodiscard]] inline auto &t() const noexcept {
            return t_;
        }

        [[nodiscard]] inline auto &gamma() const noexcept {
            return gamma_;
        }

        [[nodiscard]] inline auto &inv_q_last_mod_t() const noexcept {
            return inv_q_last_mod_t_;
        }

        [[nodiscard]] inline const uint64_t &q_last_mod_t() const noexcept {
            return q_last_mod_t_;
        }

        [[nodiscard]] inline auto &inv_m_tilde_mod_Bsk() const noexcept {
            return inv_m_tilde_mod_Bsk_;
        }

        [[nodiscard]] inline auto &prod_q_mod_Bsk() const noexcept {
            return prod_q_mod_Bsk_;
        }

        [[nodiscard]] inline auto &prod_B_mod_q() const noexcept {
            return prod_B_mod_q_;
        }

        [[nodiscard]] inline auto &inv_prod_q_mod_Bsk() const noexcept {
            return inv_prod_q_mod_Bsk_;
        }

        [[nodiscard]] inline auto &inv_prod_B_mod_m_sk() const noexcept {
            return inv_prod_B_mod_m_sk_;
        }

        // hybrid key-switching

        [[nodiscard]] inline auto &bigP_mod_q() const noexcept {
            return bigP_mod_q_;
        }

        [[nodiscard]] inline auto &bigPInv_mod_q() const noexcept {
            return bigPInv_mod_q_;
        }

        [[nodiscard]] inline auto &partQlHatInv_mod_Ql_concat() const noexcept {
            return partQlHatInv_mod_Ql_concat_;
        }

        [[nodiscard]] inline auto &pjInv_mod_t() const noexcept {
            return pjInv_mod_t_;
        }

        [[nodiscard]] inline auto &bigPInv_mod_t() const noexcept {
            return bigPInv_mod_t_;
        }

        [[nodiscard]] inline auto &pjInv_mod_q() const noexcept {
            return pjInv_mod_q_;
        }

        [[nodiscard]] inline auto base_P_to_t_conv() const noexcept {
            return base_P_to_t_conv_.get();
        }

        [[nodiscard]] inline auto &v_base_part_Ql_to_compl_part_QlP_conv() const noexcept {
            return v_base_part_Ql_to_compl_part_QlP_conv_;
        }

        [[nodiscard]] inline auto base_part_Ql_to_compl_part_QlP_conv(std::size_t index) const noexcept {
            return v_base_part_Ql_to_compl_part_QlP_conv_.at(index).get();
        }

        [[nodiscard]] inline auto base_P_to_Ql_conv() const noexcept {
            return base_P_to_Ql_conv_.get();
        }

        // HPS

        // decrypt

        [[nodiscard]] inline auto qMSB() const noexcept {
            return qMSB_;
        }

        [[nodiscard]] inline auto sizeQMSB() const noexcept {
            return sizeQMSB_;
        }

        [[nodiscard]] inline auto tMSB() const noexcept {
            return tMSB_;
        }

        [[nodiscard]] inline auto &t_QHatInv_mod_q_div_q_mod_t() const noexcept {
            return t_QHatInv_mod_q_div_q_mod_t_;
        }

        [[nodiscard]] inline auto &t_QHatInv_mod_q_div_q_frac() const noexcept {
            return t_QHatInv_mod_q_div_q_frac_;
        }

        [[nodiscard]] inline auto &t_QHatInv_mod_q_B_div_q_mod_t() const noexcept {
            return t_QHatInv_mod_q_B_div_q_mod_t_;
        }

        [[nodiscard]] inline auto &t_QHatInv_mod_q_B_div_q_frac() const noexcept {
            return t_QHatInv_mod_q_B_div_q_frac_;
        }

        // multiply

        [[nodiscard]] inline auto base_Rl() const noexcept {
            return base_Rl_.get();
        }

        [[nodiscard]] inline auto base_QlRl() const noexcept {
            return base_QlRl_.get();
        }

        [[nodiscard]] inline auto base_QlDrop() const noexcept {
            return base_QlDrop_.get();
        }

        [[nodiscard]] inline auto &base_QlRl_ntt_tables() const noexcept {
            return base_QlRl_ntt_tables_;
        }

        [[nodiscard]] inline auto base_Ql_to_Rl_conv() const noexcept {
            return base_Ql_to_Rl_conv_;
        }

        [[nodiscard]] inline auto base_Rl_to_Ql_conv() const noexcept {
            return base_Rl_to_Ql_conv_;
        }

        [[nodiscard]] inline auto base_Q_to_Rl_conv() const noexcept {
            return base_Q_to_Rl_conv_;
        }

        [[nodiscard]] inline auto base_Ql_to_QlDrop_conv() const noexcept {
            return base_Ql_to_QlDrop_conv_;
        }

        [[nodiscard]] inline auto &tRSHatInvModsDivsFrac() const noexcept {
            return tRSHatInvModsDivsFrac_;
        }

        [[nodiscard]] inline auto &tRSHatInvModsDivsModr() const noexcept {
            return tRSHatInvModsDivsModr_;
        }

        [[nodiscard]] inline auto &tQlSlHatInvModsDivsFrac() const noexcept {
            return tQlSlHatInvModsDivsFrac_;
        }

        [[nodiscard]] inline auto &tQlSlHatInvModsDivsModq() const noexcept {
            return tQlSlHatInvModsDivsModq_;
        }

        [[nodiscard]] inline auto &QlQHatInvModqDivqFrac() const noexcept {
            return QlQHatInvModqDivqFrac_;
        }

        [[nodiscard]] inline auto &QlQHatInvModqDivqModq() const noexcept {
            return QlQHatInvModqDivqModq_;
        }

    private:
        mul_tech_type mul_tech_;

        std::size_t n_ = 0;
        std::size_t size_QP_ = 2;
        std::size_t size_P_ = 1;
        bool is_key_level_ = false;

        std::shared_ptr<RNSBase> base_;
        std::shared_ptr<RNSBase> base_Q_;
        std::shared_ptr<RNSBase> base_Ql_;
        std::shared_ptr<RNSBase> base_QlP_;
        std::vector<MultiplyUIntModOperand> inv_q_last_mod_q_; // q[last]^(-1) mod q[i] for i = 0..last-1

        // hybrid key-switching
        std::vector<MultiplyUIntModOperand> bigP_mod_q_{};
        std::vector<MultiplyUIntModOperand> bigPInv_mod_q_{};
        std::vector<MultiplyUIntModOperand> partQlHatInv_mod_Ql_concat_{};
        std::vector<std::shared_ptr<BaseConverter>> v_base_part_Ql_to_compl_part_QlP_conv_{};
        std::shared_ptr<BaseConverter> base_P_to_Ql_conv_;

        // plain modulus related (BFV/BGV)
        Modulus t_;
        std::uint64_t inv_q_last_mod_t_ = 1;
        std::uint64_t q_last_mod_t_ = 1;

        // BGV
        std::shared_ptr<BaseConverter> base_q_to_t_conv_;           // Base converter: q --> t
        std::vector<MultiplyUIntModOperand> pjInv_mod_q_{};
        std::vector<MultiplyUIntModOperand> pjInv_mod_t_{};
        MultiplyUIntModOperand bigPInv_mod_t_{};
        std::shared_ptr<BaseConverter> base_P_to_t_conv_;

        // BFV enc/add/sub
        MultiplyUIntModOperand negQl_mod_t_{}; // Ql mod t
        std::vector<MultiplyUIntModOperand> tInv_mod_q_{}; // t^(-1) mod q

        // BFV BEHZ
        // decrypt (every data level)
        Modulus gamma_;
        std::shared_ptr<RNSBase> base_t_gamma_;
        std::shared_ptr<BaseConverter> base_q_to_t_gamma_conv_; // Base converter: q --> {t, gamma}
        MultiplyUIntModOperand inv_gamma_mod_t_{}; // gamma^(-1) mod t
        std::vector<MultiplyUIntModOperand> prod_t_gamma_mod_q_; // prod({t, gamma}) mod q
        std::vector<MultiplyUIntModOperand> neg_inv_q_mod_t_gamma_; // -prod(q)^(-1) mod {t, gamma}

        // multiply (only need to generate top data level)
        std::shared_ptr<RNSBase> base_B_;
        std::shared_ptr<RNSBase> base_Bsk_;
        std::shared_ptr<RNSBase> base_Bsk_m_tilde_;
        std::shared_ptr<BaseConverter> base_q_to_Bsk_conv_; // Base converter: q --> B_sk
        std::shared_ptr<BaseConverter> base_q_to_m_tilde_conv_; // Base converter: q --> {m_tilde}
        std::shared_ptr<BaseConverter> base_B_to_q_conv_; // Base converter: B --> q
        std::shared_ptr<BaseConverter> base_B_to_m_sk_conv_; // Base converter: B --> {m_sk}
        std::vector<MultiplyUIntModOperand> inv_prod_q_mod_Bsk_; // prod(q)^(-1) mod Bsk
        MultiplyUIntModOperand neg_inv_prod_q_mod_m_tilde_{}; // prod(q)^(-1) mod m_tilde
        MultiplyUIntModOperand inv_prod_B_mod_m_sk_{}; // prod(B)^(-1) mod m_sk
        std::vector<std::uint64_t> prod_B_mod_q_; // prod(B) mod q
        std::vector<MultiplyUIntModOperand> inv_m_tilde_mod_Bsk_; // m_tilde^(-1) mod Bsk
        std::vector<std::uint64_t> prod_q_mod_Bsk_; // prod(q) mod Bsk
        std::vector<NTTTables> base_Bsk_ntt_tables_; // NTTTables for Bsk
        std::vector<MultiplyUIntModOperand> m_tilde_QHatInvModq_;
        std::vector<MultiplyUIntModOperand> tModBsk_;
        Modulus m_tilde_;
        Modulus m_sk_;

        // BFV HPS

        // decrypt scale&round
        size_t min_q_idx_ = 0;
        size_t qMSB_ = 0;
        size_t sizeQMSB_ = 0;
        size_t tMSB_ = 0;
        std::vector<MultiplyUIntModOperand> t_QHatInv_mod_q_div_q_mod_t_;
        std::vector<double> t_QHatInv_mod_q_div_q_frac_;
        std::vector<MultiplyUIntModOperand> t_QHatInv_mod_q_B_div_q_mod_t_;
        std::vector<double> t_QHatInv_mod_q_B_div_q_frac_;
        // multiply
        std::shared_ptr<RNSBase> base_Rl_;
        std::shared_ptr<RNSBase> base_QlRl_;
        std::shared_ptr<RNSBase> base_QlDrop_;
        std::vector<NTTTables> base_QlRl_ntt_tables_;
        std::shared_ptr<BaseConverter> base_Ql_to_Rl_conv_;
        std::shared_ptr<BaseConverter> base_Rl_to_Ql_conv_;
        std::shared_ptr<BaseConverter> base_Q_to_Rl_conv_;
        std::shared_ptr<BaseConverter> base_Ql_to_QlDrop_conv_;
        std::vector<MultiplyUIntModOperand> tRSHatInvModsDivsModr_;
        std::vector<double> tRSHatInvModsDivsFrac_;
        std::vector<MultiplyUIntModOperand> tQlSlHatInvModsDivsModq_;
        std::vector<double> tQlSlHatInvModsDivsFrac_;
        std::vector<MultiplyUIntModOperand> QlQHatInvModqDivqModq_;
        std::vector<double> QlQHatInvModqDivqFrac_;
        std::vector<MultiplyUIntModOperand> neg_Rl_QHatInv_mod_q_; // FastExpandCRTBasisPloverQ
    };

} // namespace cahel::util
