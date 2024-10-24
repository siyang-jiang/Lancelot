// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "util/common.h"
#include "util/numth.h"
#include "util/rns.h"
#include "util/uintarithmod.h"
#include "util/uintarithsmallmod.h"
#include <algorithm>
#include <cstdio>

using namespace std;

namespace cahel::util {
    // Construct the RNSBase from the parm, calcuate
    // 1. the product of all coeff (big_Q_)
    // 2. the product of all coeff except myself (big_qiHat_)
    // 3. the inverse of the above product mod myself (qiHatInv_mod_qi_)
    RNSBase::RNSBase(const vector <Modulus> &rnsbase)
            : size_(rnsbase.size()) {
        if (!size_) {
            throw invalid_argument("rnsbase cannot be empty");
        }

        for (size_t i = 0; i < rnsbase.size(); i++) {
            // The base elements cannot be zero
            if (rnsbase[i].is_zero()) {
                throw invalid_argument("rnsbase is invalid");
            }

            for (size_t j = 0; j < i; j++) {
                // The base must be coprime
                if (!are_coprime(rnsbase[i].value(), rnsbase[j].value())) {
                    throw invalid_argument("rnsbase is invalid");
                }
            }
        }

        // Base is good; now copy it over to rnsbase_
        mod_.resize(size_);
        copy_n(rnsbase.cbegin(), size_, mod_.data());

        // Initialize CRT data
        if (!initialize())
            throw invalid_argument("rnsbase is invalid");
    }

    RNSBase::RNSBase(const RNSBase &copy) : size_(copy.size_) {

        // Copy over the base
        mod_.resize(size_);
        copy_n(copy.mod_.data(), size_, mod_.data());

        // Copy over CRT data
        prod_mod_.resize(size_);
        set_uint(copy.prod_mod_.data(), size_, prod_mod_.data());

        prod_hat_.resize(size_ * size_);
        set_uint(copy.prod_hat_.data(), size_ * size_, prod_hat_.data());

        hat_mod_.resize(size_);
        copy_n(copy.hat_mod_.data(), size_, hat_mod_.data());

        hatInv_mod_.resize(size_);
        copy_n(copy.hatInv_mod_.data(), size_, hatInv_mod_.data());

        inv_.resize(size_);
        copy_n(copy.inv_.data(), size_, inv_.data());
    }

    bool RNSBase::contains(const Modulus &value) const noexcept {
        bool result = false;

        for (size_t i = 0; i < size_; i++)
            result = result || (mod_[i] == value);
        return result;
    }

    bool RNSBase::is_subbase_of(const RNSBase &superbase) const noexcept {
        bool result = true;
        for (size_t i = 0; i < size_; i++)
            result = result && superbase.contains(mod_[i]);
        return result;
    }

    RNSBase RNSBase::extend(const Modulus &value) const {
        if (value.is_zero()) {
            throw invalid_argument("value cannot be zero");
        }

        for (size_t i = 0; i < size_; i++) {
            if (!are_coprime(mod_[i].value(), value.value())) {
                throw logic_error("cannot extend by given value");
            }
        }

        // Copy over this base
        RNSBase newbase;
        newbase.size_ = size_ + 1;
        newbase.mod_.resize(newbase.size_);
        copy_n(mod_.data(), size_, newbase.mod_.data());

        // Extend with value
        newbase.mod_[newbase.size_ - 1] = value;

        // Initialize CRT data
        if (!newbase.initialize()) {
            throw logic_error("cannot extend by given value");
        }

        return newbase;
    }

    RNSBase RNSBase::extend(const RNSBase &other) const {
        // The bases must be coprime
        for (size_t i = 0; i < other.size_; i++) {
            for (size_t j = 0; j < size_; j++) {
                if (!are_coprime(other[i].value(), mod_[j].value())) {
                    throw invalid_argument("rnsbase is invalid");
                }
            }
        }

        // Copy over this base
        RNSBase newbase;
        newbase.size_ = size_ + other.size_;
        newbase.mod_.resize(newbase.size_);
        copy_n(mod_.data(), size_, newbase.mod_.data());

        // Extend with other base
        copy_n(other.mod_.data(), other.size_, newbase.mod_.data() + size_);

        // Initialize CRT data
        if (!newbase.initialize()) {
            throw logic_error("cannot extend by given base");
        }

        return newbase;
    }

    RNSBase RNSBase::drop() const {
        if (size_ == 1) {
            throw logic_error("cannot drop from base of size 1");
        }

        // Copy over this base
        RNSBase newbase;
        newbase.size_ = size_ - 1;
        newbase.mod_.resize(newbase.size_);
        copy_n(mod_.data(), size_ - 1, newbase.mod_.data());

        // Initialize CRT data
        newbase.initialize();

        return newbase;
    }

    RNSBase RNSBase::drop(const Modulus &value) const {
        if (size_ == 1) {
            throw logic_error("cannot drop from base of size 1");
        }
        if (!contains(value)) {
            throw logic_error("base does not contain value");
        }

        // Copy over this base
        RNSBase newbase;
        newbase.size_ = size_ - 1;
        newbase.mod_.resize(newbase.size_);
        size_t source_index = 0;
        size_t dest_index = 0;
        while (dest_index < size_ - 1) {
            if (mod_[source_index] != value) {
                newbase.mod_[dest_index] = mod_[source_index];
                dest_index++;
            }
            source_index++;
        }

        // Initialize CRT data
        newbase.initialize();
        return newbase;
    }

    RNSBase RNSBase::drop(const std::vector<Modulus> &values) const {
        size_t drop_size = values.size();

        if (size_ < drop_size + 1) {
            throw logic_error("RNSBase should contain at least one modulus after dropping");
        }

        for (auto &value: values) {
            if (!contains(value)) {
                throw logic_error("base does not contain value");
            }
        }

        // Copy over this base
        RNSBase new_base;
        new_base.size_ = size_ - drop_size;
        new_base.mod_.resize(new_base.size_);
        size_t source_index = 0;
        size_t dest_index = 0;
        while (dest_index < new_base.size_) {
            if (!std::count(values.begin(), values.end(), mod_[source_index])) {
                new_base.mod_[dest_index++] = mod_[source_index];
            }
            source_index++;
        }

        // Initialize CRT data
        new_base.initialize();
        return new_base;
    }

    // Calculate big_Q_, big_qiHat_, and qiHatInv_mod_qi_
    // Also perform the validation.
    bool RNSBase::initialize() {
        prod_mod_.resize(size_);
        prod_hat_.resize(size_ * size_);
        hat_mod_.resize(size_);
        hatInv_mod_.resize(size_);
        inv_.resize(size_);

        if (size_ > 1) {
            std::vector<uint64_t> rnsbase_values(size_);
            for (size_t i = 0; i < size_; i++)
                rnsbase_values[i] = mod_[i].value();

            // Create punctured products
            for (size_t i = 0; i < size_; i++) {
                multiply_many_uint64_except(rnsbase_values.data(), size_, i, prod_hat_.data() + i * size_);
            }

            // Compute the full product, i.e., qiHat[0] * Q_[0]
            auto temp_mpi = std::vector<uint64_t>(size_);
            multiply_uint(prod_hat_.data(), size_, mod_[0].value(), size_, temp_mpi.data());
            set_uint(temp_mpi.data(), size_, prod_mod_.data());

            // Compute inverses of punctured products mod primes
            for (size_t i = 0; i < size_; i++) {
                //punctured_prod[i] % qi
                uint64_t qiHat_mod_qi = modulo_uint(prod_hat_.data() + i * size_, size_, mod_[i]);
                //qiHat_mod_qi = qiHat_mod_qi^{-1} % qi
                uint64_t qiHatInv_mod_qi;
                if (!try_invert_uint_mod(qiHat_mod_qi, mod_[i], qiHatInv_mod_qi))
                    throw invalid_argument("invalid modulus");

                hat_mod_[i].set(qiHat_mod_qi, mod_[i]);
                hatInv_mod_[i].set(qiHatInv_mod_qi, mod_[i]);
            }

            // compute 1.0 / qi
            for (size_t i = 0; i < size_; i++) {
                uint64_t qi = mod_[i].value();
                double inv = 1.0 / static_cast<double>(qi);
                inv_[i] = inv;
            }

            return true;
        }

        // Only one single modulus
        prod_mod_[0] = mod_[0].value();
        prod_hat_[0] = 1;
        hat_mod_[0].set(1, mod_[0]);
        hatInv_mod_[0].set(1, mod_[0]);
        inv_[0] = 1.0 / static_cast<double>(mod_[0].value());
        return true;
    }

    void RNSBase::decompose(uint64_t *value) const {
        if (!value) {
            throw invalid_argument("value cannot be null");
        }

        if (size_ > 1) {
            // Copy the value
            auto value_copy = std::vector<uint64_t>(size_);
            std::copy_n(value, size_, value_copy.data());
            for (size_t i = 0; i < size_; i++) {
                value[i] = modulo_uint(value_copy.data(), size_, mod_[i]);
            }
        }
    }

    // input value is assumed that [1, 2, size_Q_] ... [1, 2, size_Q_], "count" elements in total
    // output value is in the form of [1, 2, count] ... [1, 2, count], "size_Q_" elements in total
    // This happens when the degree is count
    void RNSBase::decompose_array(uint64_t *value, size_t count) const {
        if (!value) {
            throw invalid_argument("value cannot be null");
        }

        if (size_ > 1) {
            // Decompose an array of multi-precision integers into an array of arrays, one per each base element
            std::vector<uint64_t> aa(count * size_);
            uint64_t *value_copy = aa.data();
            std::copy_n(value, count * size_, value_copy);
            for (size_t i = 0; i < size_; i++) {
                for (size_t j = 0; j < count; j++)
                    *(value + i * count + j) = modulo_uint(value_copy + j * size_, size_, mod_[i]);
            }
        }
    }

    // According to CRT: x = a1 * x1 * y1 + x2 * x2 * y2 + a3 * x3 * y3 % (product of all primes)
    // where x1 is the product of all prime except prime 1, i.e., big_qiHat_[0]
    // y1 is the inverse of x1 mod prime 1, i.e., qiHatInv_mod_qi_[0]
    void RNSBase::compose(uint64_t *value) const {
        if (!value) {
            throw invalid_argument("value cannot be null");
        }

        if (size_ > 1) {
            // Copy the value
            std::vector<uint64_t> aa(size_);
            uint64_t *copy_value = aa.data();
            std::copy_n(value, size_, copy_value);

            // Clear the result
            set_zero_uint(size_, value);

            auto temp_vec = std::vector<uint64_t>(size_);
            uint64_t *temp_mpi = temp_vec.data();
            uint64_t *punctured_prod = (uint64_t *) (prod_hat_.data());
            for (size_t i = 0; i < size_; i++) {
                uint64_t temp_prod = multiply_uint_mod(copy_value[i], hatInv_mod_[i],
                                                       mod_[i]);
                multiply_uint(punctured_prod + i * size_, size_, temp_prod, size_, temp_mpi);
                add_uint_uint_mod(temp_mpi, value, prod_mod_.data(), size_, value);
            }
        }
    }

    void RNSBase::compose_array(uint64_t *value, size_t count) const {
        if (!value) {
            throw invalid_argument("value cannot be null");

        }
        if (size_ > 1) {
            // Merge the coefficients first
            std::vector<uint64_t> temp_array(count * size_);
            for (size_t i = 0; i < count; i++) {
                for (size_t j = 0; j < size_; j++) {
                    temp_array[j + (i * size_)] = value[(j * count) + i];
                }
            }

            // Clear the result
            set_zero_uint(count * size_, value);

            uint64_t *temp_array_iter;
            uint64_t *value_iter;
            uint64_t *punctured_prod = (uint64_t *) (prod_hat_.data());

            // Compose an array of RNS integers into a single array of multi-precision integers
            auto temp_mpi = std::vector<uint64_t>(size_);

            for (size_t i = 0; i < count; i++) {
                value_iter = value + i * size_;
                temp_array_iter = temp_array.data() + i * size_;

                for (size_t j = 0; j < size_; j++) {
                    uint64_t temp_prod = multiply_uint_mod(*(temp_array_iter + j),
                                                           hatInv_mod_[j], mod_[j]);
                    multiply_uint(punctured_prod + j * size_, size_, temp_prod, size_, temp_mpi.data());
                    add_uint_uint_mod(temp_mpi.data(), value_iter, prod_mod_.data(), size_, value_iter);
                }
            }
        }
    }

    void BaseConverter::initialize() {
        // Verify that the size_QP is not too large
        size_t size_Q = ibase_.size();
        size_t size_P = obase_.size();
        auto size_QP = util::mul_safe(size_Q, size_P);
        if (!fits_in<std::size_t>(size_QP)) {
            throw logic_error("invalid parameters");
        }

        // Create the base-change matrix rows
        QHatModp_.resize(size_P);
        for (size_t j = 0; j < size_P; j++) {
            QHatModp_[j].resize(size_Q);
            auto ibase_big_qiHat = ibase_.big_qiHat();
            auto &pj = obase_.base()[j];
            for (size_t i = 0; i < size_Q; i++) {
                // Base-change matrix contains the punctured products of ibase elements modulo the obase
                QHatModp_[j][i] = modulo_uint(ibase_big_qiHat + i * size_Q,
                                              size_Q,
                                              pj);
            }
        }

        alphaQModp_.resize(size_Q + 1);
        for (size_t j = 0; j < size_P; j++) {
            auto big_Q = ibase_.big_modulus();
            auto &pj = obase_.base()[j];
            uint64_t big_Q_mod_pj = modulo_uint(big_Q, size_Q, pj);
            for (size_t alpha = 0; alpha < size_Q + 1; alpha++) {
                alphaQModp_[alpha].push_back(multiply_uint_mod(alpha, big_Q_mod_pj, pj));
            }
        }

        negPQHatInvModq_.resize(size_Q);
        PModq_.resize(size_Q);
        for (size_t i = 0; i < size_Q; i++) {
            auto &qi = ibase_.base()[i];
            auto QHatInvModqi = ibase_.QHatInvModq()[i];
            auto P = obase_.big_modulus();
            uint64_t PModqi = modulo_uint(P, size_P, qi);
            PModq_[i].set(PModqi, qi);
            uint64_t PQHatInvModqi = multiply_uint_mod(PModqi, QHatInvModqi, qi);
            uint64_t negPQHatInvModqi = qi.value() - PQHatInvModqi;
            negPQHatInvModq_[i].set(negPQHatInvModqi, qi);
        }

        QInvModp_.resize(size_P);
        for (size_t j = 0; j < size_P; j++) {
            QInvModp_[j].resize(size_Q);
            auto &pj = obase_.base()[j];
            for (size_t i = 0; i < size_Q; i++) {
                auto &qi = ibase_.base()[i];
                if (!try_invert_uint_mod(qi.value(), pj, QInvModp_[j][i])) {
                    throw logic_error("invalid rns bases in computing QInvModp");
                }
            }
        }
    }

    RNSTool::RNSTool(size_t n, size_t size_P, const RNSBase &base,
                     const std::vector<Modulus> &modulus_QP, const Modulus &t, mul_tech_type mul_tech) {
        // Return if base is out of bounds
        size_t base_size = base.size();
        base_ = make_shared<RNSBase>(base);

        if (base_size < CAHEL_COEFF_MOD_COUNT_MIN || base_size > CAHEL_COEFF_MOD_COUNT_MAX) {
            throw invalid_argument("RNSBase is invalid");
        }

        // Return if coeff_count is not a power of two or out of bounds
        int log_n = get_power_of_two(n);
        if (log_n < 0 || n > CAHEL_POLY_MOD_DEGREE_MAX ||
            n < CAHEL_POLY_MOD_DEGREE_MIN) {
            throw invalid_argument("poly_modulus_degree is invalid");
        }

        mul_tech_ = mul_tech;
        n_ = n;
        size_QP_ = modulus_QP.size();
        size_P_ = size_P;

        if (base_size == size_QP_)
            is_key_level_ = true;

        size_t size_QP = modulus_QP.size();
        size_t size_Q = size_QP - size_P;

        vector<Modulus> modulus_Q(size_Q);
        for (size_t i = 0; i < size_Q; i++)
            modulus_Q[i] = modulus_QP[i];
        base_Q_ = make_shared<RNSBase>(modulus_Q);

        vector<Modulus> modulus_P(size_P);
        for (size_t i = 0; i < size_P; i++)
            modulus_P[i] = modulus_QP[size_Q + i];

        if (base_size == size_QP) { // key level
            base_QlP_ = make_shared<RNSBase>(base);
            base_Ql_ = make_shared<RNSBase>(base_QlP_->drop(modulus_P));
        } else { // data level
            base_Ql_ = make_shared<RNSBase>(base);
            base_QlP_ = make_shared<RNSBase>(base_Ql_->extend(RNSBase(modulus_P)));
        }

        size_t size_Ql = base_Ql_->size();
        size_t size_QlP = size_Ql + size_P;

        // Compute base_[last]^(-1) mod base_[i] for i = 0..last-1
        // This is used by modulus switching and rescaling
        inv_q_last_mod_q_.resize(size_Ql - 1);
        uint64_t value_inv_q_last_mod_q;
        for (size_t i = 0; i < (size_Ql - 1); i++) {
            if (!try_invert_uint_mod((*base_Ql_)[size_Ql - 1].value(), (*base_Ql_)[i], value_inv_q_last_mod_q)) {
                throw logic_error("invalid rns bases in computing inv_q_last_mod_q");
            }
            inv_q_last_mod_q_[i].set(value_inv_q_last_mod_q, (*base_Ql_)[i]);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // hybrid key-switching
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if (size_P != 0) {
            size_t alpha = size_P;

            vector<uint64_t> values_P(size_P);
            for (size_t i = 0; i < size_P; i++)
                values_P[i] = modulus_P[i].value();

            // Compute big P
            vector<uint64_t> bigP(size_P, 0);
            multiply_many_uint64(values_P.data(), size_P, bigP.data());

            bigP_mod_q_.resize(size_Ql);
            bigPInv_mod_q_.resize(size_Ql);
            for (size_t i = 0; i < size_Ql; ++i) {
                auto base_qi = base_Ql_->base()[i];
                uint64_t tmp = modulo_uint(bigP.data(), size_P, base_qi);
                bigP_mod_q_[i].set(tmp, base_qi);
                if (!try_invert_uint_mod(tmp, base_qi, tmp))
                    throw std::logic_error("invalid rns bases in computing PInv mod q");
                bigPInv_mod_q_[i].set(tmp, base_qi);
            }

            // data level rns tool, create base converter from part Ql to compl part QlP
            if (base_size <= size_Q) {
                // create modulus_QlP
                vector<Modulus> modulus_QlP(size_QlP);
                for (size_t i = 0; i < size_Ql; i++)
                    modulus_QlP[i] = modulus_QP[i];
                for (size_t i = 0; i < size_P; i++)
                    modulus_QlP[size_Ql + i] = modulus_QP[size_Q + i];

                partQlHatInv_mod_Ql_concat_.resize(size_Ql);

                auto beta = static_cast<uint32_t>(ceil((static_cast<double>(size_Ql)) / static_cast<double>(alpha)));
                for (size_t beta_idx = 0; beta_idx < beta; beta_idx++) {
                    size_t startPartIdx = alpha * beta_idx;
                    size_t size_PartQl = (beta_idx == beta - 1) ? (size_Ql - alpha * (beta - 1)) : alpha;
                    size_t endPartIdx = startPartIdx + size_PartQl;

                    std::vector<Modulus> modulus_part_Ql{};
                    std::vector<Modulus> modulus_compl_part_QlP = modulus_QlP;

                    for (size_t j = startPartIdx; j < endPartIdx; ++j)
                        modulus_part_Ql.push_back(modulus_QlP[j]);
                    auto first = modulus_compl_part_QlP.cbegin() + startPartIdx;
                    auto last = modulus_compl_part_QlP.cbegin() + endPartIdx;
                    modulus_compl_part_QlP.erase(first, last);

                    auto base_part_Ql = RNSBase(modulus_part_Ql);
                    std::copy_n(base_part_Ql.QHatInvModq(), modulus_part_Ql.size(),
                                partQlHatInv_mod_Ql_concat_.begin() + startPartIdx);
                    auto base_compl_part_QlP = RNSBase(modulus_compl_part_QlP);

                    auto base_part_Ql_to_compl_part_QlP_conv = make_shared<BaseConverter>(
                            base_part_Ql, base_compl_part_QlP);
                    v_base_part_Ql_to_compl_part_QlP_conv_.push_back(base_part_Ql_to_compl_part_QlP_conv);
                }
            }

            // create base converter from P to Ql for mod down
            base_P_to_Ql_conv_ = make_shared<BaseConverter>(RNSBase(modulus_P), *base_Ql_);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // plain modulus related (BFV/BGV)
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        t_ = t;
        if (!t_.is_zero()) {
            // Compute q[last] mod t and q[last]^(-1) mod t
            if (!try_invert_uint_mod(base_Ql_->base()[size_Ql - 1].value(), t_, inv_q_last_mod_t_))
                throw logic_error("invalid rns bases");

            q_last_mod_t_ = barrett_reduce_64(base_Ql_->base()[size_Ql - 1].value(), t_);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // BGV only
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if (!t_.is_zero() && mul_tech == mul_tech_type::none) {
            // Set up BaseConvTool for q --> {t}
            base_q_to_t_conv_ = make_shared<BaseConverter>(*base_Ql_, RNSBase({t_}));

            if (size_P != 0) {
                vector<uint64_t> values_P(size_P);
                for (size_t i = 0; i < size_P; i++)
                    values_P[i] = modulus_P[i].value();

                // Compute big P
                vector<uint64_t> bigP(size_P, 0);
                multiply_many_uint64(values_P.data(), size_P, bigP.data());

                pjInv_mod_q_.resize(size_Ql * size_P);
                for (size_t i = 0; i < size_Ql; i++) {
                    for (size_t j = 0; j < size_P; j++) {
                        uint64_t pj = values_P[j];
                        uint64_t qi = base_Ql_->base()[i].value();
                        uint64_t pjInv_mod_qi_value;
                        if (!try_invert_uint_mod(pj, qi, pjInv_mod_qi_value))
                            throw std::logic_error("invalid rns bases when computing pjInv_mod_qi");
                        pjInv_mod_q_[i * size_P + j].set(pjInv_mod_qi_value, base_Ql_->base()[i]);
                    }
                }

                pjInv_mod_t_.resize(size_P);
                for (size_t j = 0; j < size_P; j++) {
                    uint64_t pjInv_mod_t_value;
                    if (!try_invert_uint_mod(modulus_P[j].value(), t_.value(), pjInv_mod_t_value))
                        throw std::logic_error("invalid rns bases when computing pjInv_mod_t");
                    pjInv_mod_t_[j].set(pjInv_mod_t_value, t_);
                }

                uint64_t bigP_mod_t_value = modulo_uint(bigP.data(), size_P, t_);
                uint64_t bigPInv_mod_t_value;
                if (!try_invert_uint_mod(bigP_mod_t_value, t_.value(), bigPInv_mod_t_value))
                    throw std::logic_error("invalid rns bases when computing pjInv_mod_t");
                bigPInv_mod_t_.set(bigPInv_mod_t_value, t_);

                // create base converter from P to t for mod down
                base_P_to_t_conv_ = make_shared<BaseConverter>(RNSBase(modulus_P), RNSBase({t_}));
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // BFV enc/add/sub
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if (mul_tech != mul_tech_type::none) {
            vector<uint64_t> values_Ql(size_Ql);
            for (size_t i = 0; i < size_Ql; i++)
                values_Ql[i] = base_Ql_->base()[i].value();

            // Compute big Ql
            vector<uint64_t> bigQl(size_Ql, 0);
            multiply_many_uint64(values_Ql.data(), size_Ql, bigQl.data());

            uint64_t bigQl_mod_t_value = modulo_uint(bigQl.data(), size_Ql, t_);
            negQl_mod_t_.set(t_.value() - bigQl_mod_t_value, t_);

            tInv_mod_q_.resize(size_Ql);
            for (size_t i = 0; i < size_Ql; i++) {
                uint64_t tInv_mod_qi_value;
                auto &qi = base_Ql_->base()[i];
                if (!try_invert_uint_mod(t_.value(), qi.value(), tInv_mod_qi_value))
                    throw std::logic_error("invalid rns bases when computing tInv_mod_qi");
                tInv_mod_q_[i].set(tInv_mod_qi_value, qi);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // BEHZ
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // BEHZ decrypt
        if (mul_tech == mul_tech_type::behz && base_size <= size_Q) {
            auto primes = get_primes(n_, CAHEL_INTERNAL_MOD_BIT_COUNT, 1);
            gamma_ = primes[0];

            // Set up t-gamma base if t_ is non-zero
            base_t_gamma_ = make_shared<RNSBase>(vector<Modulus>{t_, gamma_});

            uint64_t temp;

            // Compute gamma^(-1) mod t
            if (!try_invert_uint_mod(barrett_reduce_64(gamma_.value(), t_), t_, temp)) {
                throw logic_error("invalid rns bases");
            }
            inv_gamma_mod_t_.set(temp, t_);

            // Compute prod({t, gamma}) mod base_Ql
            prod_t_gamma_mod_q_.resize(size_Ql);
            for (size_t i = 0; i < size_Ql; i++) {
                prod_t_gamma_mod_q_[i].set(
                        multiply_uint_mod((*base_t_gamma_)[0].value(), (*base_t_gamma_)[1].value(),
                                          (*base_Ql_)[i]),
                        (*base_Ql_)[i]);
            }

            // Compute -prod(base_Ql)^(-1) mod {t, gamma}
            size_t base_t_gamma_size = 2;
            neg_inv_q_mod_t_gamma_.resize(base_t_gamma_size);
            for (size_t i = 0; i < base_t_gamma_size; i++) {
                auto operand = modulo_uint(base_Ql_->big_modulus(), size_Ql, (*base_t_gamma_)[i]);
                if (!try_invert_uint_mod(operand, (*base_t_gamma_)[i], neg_inv_q_mod_t_gamma_[i].operand)) {
                    throw logic_error("invalid rns bases");
                }
                neg_inv_q_mod_t_gamma_[i].set(
                        negate_uint_mod(neg_inv_q_mod_t_gamma_[i].operand, (*base_t_gamma_)[i]),
                        (*base_t_gamma_)[i]);
            }

            // Set up BaseConverter for base_Ql --> {t, gamma}
            base_q_to_t_gamma_conv_ = make_shared<BaseConverter>(*base_Ql_, *base_t_gamma_);
        }

        // BEHZ multiply
        if (mul_tech == mul_tech_type::behz && base_size == size_Q) {
            // In some cases we might need to increase the size of the base B by one, namely we require
            // K * n * t * base_Q^2 < base_Q * prod(B) * m_sk, where K takes into account cross terms when larger size ciphertexts
            // are used, and n is the "delta factor" for the ring. We reserve 32 bits for K * n. Here the coeff modulus
            // primes q_i are bounded to be SEAL_USER_MOD_BIT_COUNT_MAX (60) bits, and all primes in B and m_sk are
            // SEAL_INTERNAL_MOD_BIT_COUNT (61) bits.
            int total_coeff_bit_count = get_significant_bit_count_uint(base_Q_->big_modulus(), base_Q_->size());

            size_t base_B_size = size_Q;
            if (32 + t_.bit_count() + total_coeff_bit_count >=
                CAHEL_INTERNAL_MOD_BIT_COUNT * static_cast<int>(size_Q) + CAHEL_INTERNAL_MOD_BIT_COUNT) {
                base_B_size++;
            }

            // only generate gamma, m_sk, B at top data level
            // else only generate gamma
            // size_t get_primes_count = (base_size == size_Q) ? (base_B_size + 2) : 1;
            size_t get_primes_count = base_B_size + 1;

            // Sample primes for B and two more primes: m_sk and gamma
            auto baseconv_primes = get_primes(n_, CAHEL_INTERNAL_MOD_BIT_COUNT, get_primes_count);
            auto baseconv_primes_iter = baseconv_primes.cbegin();
            m_sk_ = *baseconv_primes_iter++;
            vector<Modulus> base_B_primes;
            copy_n(baseconv_primes_iter, base_B_size, back_inserter(base_B_primes));

            // Set m_tilde_ to a non-prime value
            m_tilde_ = uint64_t(1) << 32;

            // Populate the base arrays
            base_B_ = make_shared<RNSBase>(base_B_primes);
            base_Bsk_ = make_shared<RNSBase>(base_B_->extend(m_sk_));
            base_Bsk_m_tilde_ = make_shared<RNSBase>(base_Bsk_->extend(m_tilde_));

            // Generate the Bsk NTTTables; these are used for NTT after base extension to Bsk
            size_t base_Bsk_size = base_Bsk_->size();
            try {
                CreateNTTTables(
                        log_n, vector<Modulus>(base_Bsk_->base(), base_Bsk_->base() + base_Bsk_size),
                        base_Bsk_ntt_tables_);
            }
            catch (const logic_error &) {
                throw logic_error("invalid rns bases in Bsk");
            }

            // used in optimizing BEHZ fastbconv_m_tilde
            m_tilde_QHatInvModq_.resize(size_Q);
            for (size_t i = 0; i < size_Q; i++) {
                auto qi = base_Q_->base()[i];
                auto QHatInvModqi = base_Q_->QHatInvModq()[i];
                m_tilde_QHatInvModq_[i].set(multiply_uint_mod(m_tilde_.value(), QHatInvModqi, qi), qi);
            }

            tModBsk_.resize(base_Bsk_size);
            for (size_t i = 0; i < base_Bsk_->size(); i++) {
                tModBsk_[i].set(t.value(), (*base_Bsk_)[i]);
            }

            // Set up BaseConverter for base_Q --> Bsk
            base_q_to_Bsk_conv_ = make_shared<BaseConverter>(*base_Q_, *base_Bsk_);

            // Set up BaseConverter for base_Q --> {m_tilde}
            base_q_to_m_tilde_conv_ = make_shared<BaseConverter>(*base_Q_, RNSBase({m_tilde_}));

            // Set up BaseConverter for B --> base_Q
            base_B_to_q_conv_ = make_shared<BaseConverter>(*base_B_, *base_Q_);

            // Set up BaseConverter for B --> {m_sk}
            base_B_to_m_sk_conv_ = make_shared<BaseConverter>(*base_B_, RNSBase({m_sk_}));

            // Compute prod(B) mod base_Q
            prod_B_mod_q_.resize(size_Q);
            for (size_t i = 0; i < prod_B_mod_q_.size(); i++) {
                prod_B_mod_q_[i] = modulo_uint(base_B_->big_modulus(), base_B_size, (*base_Q_)[i]);
            }

            uint64_t temp;

            inv_prod_q_mod_Bsk_.resize(base_Bsk_size);
            for (size_t i = 0; i < base_Bsk_size; i++) {
                temp = modulo_uint(base_Q_->big_modulus(), size_Q, (*base_Bsk_)[i]);
                if (!try_invert_uint_mod(temp, (*base_Bsk_)[i], temp)) {
                    throw logic_error("invalid rns bases");
                }
                inv_prod_q_mod_Bsk_[i].set(temp, (*base_Bsk_)[i]);
            }

            // Compute prod(B)^(-1) mod m_sk
            temp = modulo_uint(base_B_->big_modulus(), base_B_size, m_sk_);
            if (!try_invert_uint_mod(temp, m_sk_, temp)) {
                throw logic_error("invalid rns bases");
            }
            inv_prod_B_mod_m_sk_.set(temp, m_sk_);

            // Compute m_tilde^(-1) mod Bsk
            inv_m_tilde_mod_Bsk_.resize(base_Bsk_size);
            for (size_t i = 0; i < base_Bsk_size; i++) {
                if (!try_invert_uint_mod(barrett_reduce_64(m_tilde_.value(), (*base_Bsk_)[i]), (*base_Bsk_)[i],
                                         temp)) {
                    throw logic_error("invalid rns bases");
                }
                inv_m_tilde_mod_Bsk_[i].set(temp, (*base_Bsk_)[i]);
            }

            // Compute prod(base_Q)^(-1) mod m_tilde
            temp = modulo_uint(base_Q_->big_modulus(), size_Q, m_tilde_);
            if (!try_invert_uint_mod(temp, m_tilde_, temp)) {
                throw logic_error("invalid rns bases");
            }
            neg_inv_prod_q_mod_m_tilde_.set(negate_uint_mod(temp, m_tilde_), m_tilde_);

            // Compute prod(base_Q) mod Bsk
            prod_q_mod_Bsk_.resize(base_Bsk_size);
            for (size_t i = 0; i < base_Bsk_size; i++) {
                prod_q_mod_Bsk_[i] = modulo_uint(base_Q_->big_modulus(), size_Q, (*base_Bsk_)[i]);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // HPS
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // HPS Decrypt Scale&Round
        if ((mul_tech == mul_tech_type::hps ||
             mul_tech == mul_tech_type::hps_overq ||
             mul_tech == mul_tech_type::hps_overq_leveled) && (base_size <= size_Q)) {

            vector<uint64_t> v_qi(size_Q);
            for (size_t i = 0; i < size_Q; i++) {
                v_qi[i] = modulus_Q[i].value();
            }

            size_t max_q_idx = max_element(v_qi.begin(), v_qi.end()) - v_qi.begin();
            min_q_idx_ = min_element(v_qi.begin(), v_qi.end()) - v_qi.begin();

            qMSB_ = get_significant_bit_count(v_qi[max_q_idx]);
            sizeQMSB_ = get_significant_bit_count_uint(&size_Ql, 1);
            tMSB_ = get_significant_bit_count_uint(t_.data(), 1);

            t_QHatInv_mod_q_div_q_mod_t_.resize(size_Ql);
            t_QHatInv_mod_q_div_q_frac_.resize(size_Ql);
            t_QHatInv_mod_q_B_div_q_mod_t_.resize(size_Ql);
            t_QHatInv_mod_q_B_div_q_frac_.resize(size_Ql);

            for (size_t i = 0; i < size_Ql; i++) {
                auto qi = base_Ql_->base()[i];
                auto value_t = t_.value();

                std::vector<uint64_t> big_t_QHatInv_mod_qi(2, 0);

                auto qiHatInv_mod_qi = base_Ql_->QHatInvModq()[i];

                multiply_uint(&value_t, 1,
                              qiHatInv_mod_qi.operand,
                              2, big_t_QHatInv_mod_qi.data());

                std::vector<uint64_t> padding_zero_qi(2, 0);
                padding_zero_qi[0] = qi.value();

                std::vector<uint64_t> big_t_QHatInv_mod_q_div_qi(2, 0);

                divide_uint_inplace(big_t_QHatInv_mod_qi.data(),
                                    padding_zero_qi.data(),
                                    2,
                                    big_t_QHatInv_mod_q_div_qi.data());

                uint64_t value_t_QHatInv_mod_q_div_q_mod_t = modulo_uint(big_t_QHatInv_mod_q_div_qi.data(),
                                                                         2,
                                                                         t_);

                t_QHatInv_mod_q_div_q_mod_t_[i].set(value_t_QHatInv_mod_q_div_q_mod_t, t_);

                uint64_t numerator = modulo_uint(big_t_QHatInv_mod_qi.data(),
                                                 2,
                                                 qi);
                uint64_t denominator = qi.value();
                t_QHatInv_mod_q_div_q_frac_[i] = static_cast<double>(numerator) / static_cast<double>(denominator);

                if (qMSB_ + sizeQMSB_ >= 52) {
                    size_t qMSBHf = qMSB_ >> 1;

                    std::vector<uint64_t> QHatInv_mod_qi_B(2, 0);
                    QHatInv_mod_qi_B[0] = qiHatInv_mod_qi.operand;
                    left_shift_uint128(QHatInv_mod_qi_B.data(), qMSBHf, QHatInv_mod_qi_B.data());
                    uint64_t QHatInv_B_mod_qi = modulo_uint(QHatInv_mod_qi_B.data(), 2, qi);

                    std::vector<uint64_t> t_QHatInv_B_mod_qi(2, 0);
                    multiply_uint(&value_t, 1,
                                  QHatInv_B_mod_qi,
                                  2, t_QHatInv_B_mod_qi.data());

                    std::vector<uint64_t> t_QHatInv_B_mod_qi_div_qi(2, 0);
                    divide_uint_inplace(t_QHatInv_B_mod_qi.data(),
                                        padding_zero_qi.data(),
                                        2,
                                        t_QHatInv_B_mod_qi_div_qi.data());

                    uint64_t value_t_QHatInv_mod_q_B_div_q_mod_t = modulo_uint(t_QHatInv_B_mod_qi_div_qi.data(),
                                                                               2,
                                                                               t_);

                    t_QHatInv_mod_q_B_div_q_mod_t_[i].set(value_t_QHatInv_mod_q_B_div_q_mod_t, t_);

                    numerator = modulo_uint(t_QHatInv_B_mod_qi.data(),
                                            2,
                                            qi);
                    t_QHatInv_mod_q_B_div_q_frac_[i] =
                            static_cast<double>(numerator) / static_cast<double>(denominator);
                }
            }
        }

        // HPS multiply
        // HPS or HPSOverQ don't need to pre-compute at levels other than first data level
        // HPSOverQLeveled doesn't need to pre-compute at the key level
        // otherwise, pre-computations are needed
        // note that if base size equals to Q size, it is the first data level
        if (mul_tech == mul_tech_type::hps && base_size == size_Q) {
            // Generate modulus R
            // for HPS, R is one more than Q
            size_t size_R = size_Q + 1;
            size_t size_QR = size_Q + size_R;

            // each prime in R is smaller than the smallest prime in Q
            auto modulus_R = get_primes_below(n_, modulus_Q[min_q_idx_].value(), size_R);
            base_Rl_ = make_shared<RNSBase>(modulus_R);
            base_QlRl_ = make_shared<RNSBase>(base_Q_->extend(*base_Rl_));

            // Generate QR NTT tables
            try {
                CreateNTTTables(log_n,
                                vector<Modulus>(base_QlRl_->base(), base_QlRl_->base() + size_QR),
                                base_QlRl_ntt_tables_);
            }
            catch (const logic_error &) {
                throw logic_error("invalid rns bases in base QR");
            }

            auto bigint_Q = base_Q_->big_modulus();
            auto bigint_R = base_Rl_->big_modulus();

            // Used for switching ciphertext from basis Q to R
            base_Ql_to_Rl_conv_ = make_shared<BaseConverter>(*base_Ql_, *base_Rl_);

            // Used for switching ciphertext from basis R to Q
            base_Rl_to_Ql_conv_ = make_shared<BaseConverter>(*base_Rl_, *base_Q_);

            // Used for t/Q scale&round in HPS method
            tRSHatInvModsDivsFrac_.resize(size_Q);
            tRSHatInvModsDivsModr_.resize(size_R * (size_Q + 1));

            // first compute tRSHatInvMods
            vector<vector<uint64_t>> tRSHatInvMods(size_QR);
            for (size_t i = 0; i < size_QR; i++) {
                // resize tRSHatInvModsi to size_R + 2 and initialize to 0
                tRSHatInvMods[i].resize(size_R + 2, 0);
                auto SHatInvModsi = base_QlRl_->QHatInvModq()[i];
                vector<uint64_t> tR(size_R + 1, 0);
                multiply_uint(bigint_R, size_R, t.value(), size_R + 1, tR.data());
                multiply_uint(tR.data(), size_R + 1, SHatInvModsi.operand, size_R + 2,
                              tRSHatInvMods[i].data());
            }

            // compute tRSHatInvModsDivsFrac
            for (size_t i = 0; i < size_Q; i++) {
                auto qi = (*base_Q_)[i];
                uint64_t tRSHatInvModsModqi = modulo_uint(tRSHatInvMods[i].data(), size_R + 2, qi);
                tRSHatInvModsDivsFrac_[i] = static_cast<double>(tRSHatInvModsModqi) /
                                            static_cast<double>(qi.value());
            }

            // compute tRSHatInvModsDivs
            vector<vector<uint64_t>> tRSHatInvModsDivs(size_QR);
            for (size_t i = 0; i < size_QR; i++) {
                // resize tRSHatInvModsDivsi to size_R + 2 and initialize to 0
                tRSHatInvModsDivs[i].resize(size_R + 2, 0);
                // align si with big integer tRSHatInvMods
                auto si = base_QlRl_->base()[i];
                vector<uint64_t> bigint_si(size_R + 2, 0);
                bigint_si[0] = si.value();
                // div si
                std::vector<uint64_t> temp_remainder(size_R + 2, 0);
                divide_uint(tRSHatInvMods[i].data(), bigint_si.data(), size_R + 2,
                            tRSHatInvModsDivs[i].data(), temp_remainder.data());
            }

            // compute tRSHatInvModsDivsModr
            for (size_t j = 0; j < size_R; j++) {
                auto &rj = modulus_R[j];
                for (size_t i = 0; i < size_Q; i++) {
                    // mod rj
                    uint64_t tRSHatInvModsDivqiModrj = modulo_uint(tRSHatInvModsDivs[i].data(), size_R + 2, rj);
                    tRSHatInvModsDivsModr_[j * (size_Q + 1) + i].set(tRSHatInvModsDivqiModrj, rj);
                }
                // mod rj
                uint64_t tRSHatInvModsDivrjModrj = modulo_uint(tRSHatInvModsDivs[size_Q + j].data(), size_R + 2,
                                                               rj);
                tRSHatInvModsDivsModr_[j * (size_Q + 1) + size_Q].set(tRSHatInvModsDivrjModrj, rj);
            }
        }

        if ((mul_tech == mul_tech_type::hps_overq && base_size == size_Q) ||
            (mul_tech == mul_tech_type::hps_overq_leveled && base_size <= size_Q)) {
            // Generate modulus Rl
            // for HPSOverQ and HPSOverQLeveled, Rl is the same size as Ql
            size_t size_Rl = size_Ql;
            size_t size_QlRl = size_Ql + size_Rl;

            // each prime in Rl is smaller than the smallest prime in Ql
            auto modulus_Rl = get_primes_below(n_, modulus_Q[min_q_idx_].value(), size_Rl);
            base_Rl_ = make_shared<RNSBase>(modulus_Rl);
            base_QlRl_ = make_shared<RNSBase>(base_Ql_->extend(*base_Rl_));

            // Generate QlRl NTT tables
            try {
                CreateNTTTables(log_n,
                                vector<Modulus>(base_QlRl_->base(), base_QlRl_->base() + size_QlRl),
                                base_QlRl_ntt_tables_);
            }
            catch (const logic_error &) {
                throw logic_error("invalid rns bases in base QlRl");
            }

            auto bigint_Ql = base_Ql_->big_modulus();
            auto bigint_Rl = base_Rl_->big_modulus();

            // Used for switching ciphertext from basis Q(Ql) to R(Rl)
            base_Ql_to_Rl_conv_ = make_shared<BaseConverter>(*base_Ql_, *base_Rl_);

            // Used for switching ciphertext from basis Rl to Ql
            base_Rl_to_Ql_conv_ = make_shared<BaseConverter>(*base_Rl_, *base_Ql_);

            // Used for t/Rl scale&round in overQ variants
            tQlSlHatInvModsDivsFrac_.resize(size_Rl);
            tQlSlHatInvModsDivsModq_.resize(size_Ql * (size_Rl + 1));

            // first compute tQlSlHatInvMods
            vector<vector<uint64_t>> tQlSlHatInvMods(size_QlRl);
            for (size_t i = 0; i < size_QlRl; i++) {
                // resize tQlSlHatInvModsi to size_Ql + 2 and initialize to 0
                tQlSlHatInvMods[i].resize(size_Ql + 2, 0);
                auto SHatInvModsi = base_QlRl_->QHatInvModq()[i];
                vector<uint64_t> tQl(size_Ql + 1, 0);
                multiply_uint(bigint_Ql, size_Ql, t.value(), size_Ql + 1, tQl.data());
                multiply_uint(tQl.data(), size_Ql + 1, SHatInvModsi.operand, size_Ql + 2,
                              tQlSlHatInvMods[i].data());
            }

            // compute tQlSlHatInvModsDivsFrac
            for (size_t j = 0; j < size_Rl; j++) {
                auto rj = base_Rl_->base()[j];
                uint64_t tQlSlHatInvModsModrj = modulo_uint(tQlSlHatInvMods[size_Ql + j].data(), size_Ql + 2, rj);
                tQlSlHatInvModsDivsFrac_[j] = static_cast<double>(tQlSlHatInvModsModrj) /
                                              static_cast<double>(rj.value());
            }

            // compute tQlSlHatInvModsDivs
            vector<vector<uint64_t>> tQlSlHatInvModsDivs(size_QlRl);
            for (size_t i = 0; i < size_QlRl; i++) {
                // resize tQlSlHatInvModsDivsi to size_Ql + 2 and initialize to 0
                tQlSlHatInvModsDivs[i].resize(size_Ql + 2, 0);
                // align si with big integer tQlSlHatInvMods
                auto si = base_QlRl_->base()[i];
                vector<uint64_t> bigint_si(size_Ql + 2, 0);
                bigint_si[0] = si.value();
                // div si
                std::vector<uint64_t> temp_remainder(size_Ql + 2, 0);
                divide_uint(tQlSlHatInvMods[i].data(), bigint_si.data(), size_Ql + 2,
                            tQlSlHatInvModsDivs[i].data(), temp_remainder.data());
            }

            // compute tQlSlHatInvModsDivsModq
            for (size_t i = 0; i < size_Ql; i++) {
                auto &qi = base_Ql_->base()[i];
                for (size_t j = 0; j < size_Rl; j++) {
                    // mod qi
                    uint64_t tQlSlHatInvModsDivrjModqi = modulo_uint(tQlSlHatInvModsDivs[size_Ql + j].data(),
                                                                     size_Ql + 2, qi);
                    tQlSlHatInvModsDivsModq_[i * (size_Rl + 1) + j].set(tQlSlHatInvModsDivrjModqi, qi);
                }
                // mod qi
                uint64_t tQlSlHatInvModsDivqiModqi = modulo_uint(tQlSlHatInvModsDivs[i].data(), size_Ql + 2, qi);
                tQlSlHatInvModsDivsModq_[i * (size_Rl + 1) + size_Rl].set(tQlSlHatInvModsDivqiModqi, qi);
            }

            // drop levels
            if (mul_tech == mul_tech_type::hps_overq_leveled && base_size < size_Q) {

                // Used for Ql/Q scale&round in overQLeveled variants
                if (size_Q - size_Ql < 1)
                    throw std::logic_error("Something is wrong, check rnstool.");
                size_t size_QlDrop = size_Q - size_Ql;

                vector<Modulus> modulus_QlDrop(size_QlDrop);
                for (size_t i = 0; i < size_QlDrop; i++)
                    modulus_QlDrop[i] = modulus_Q[size_Ql + i];
                base_QlDrop_ = make_shared<RNSBase>(modulus_QlDrop);

                // Used for switching ciphertext from basis Q to Rl
                base_Q_to_Rl_conv_ = make_shared<BaseConverter>(*base_Q_, *base_Rl_);

                // Used for switching ciphertext from basis Ql to QlDrop (Ql modup to Q)
                base_Ql_to_QlDrop_conv_ = make_shared<BaseConverter>(*base_Ql_, *base_QlDrop_);

                QlQHatInvModqDivqFrac_.resize(size_QlDrop);
                QlQHatInvModqDivqModq_.resize(size_Ql * (size_QlDrop + 1));

                // first compute QlQHatInvModq
                vector<vector<uint64_t>> QlQHatInvModq(size_Q);
                for (size_t i = 0; i < size_Q; i++) {
                    // resize QlQHatInvModq[i] to size_Ql + 1 and initialize to 0
                    QlQHatInvModq[i].resize(size_Ql + 1, 0);
                    multiply_uint(bigint_Ql, size_Ql,
                                  base_Q_->QHatInvModq()[i].operand,
                                  size_Ql + 1, QlQHatInvModq[i].data());
                }

                // compute QlQHatInvModqDivqFrac
                for (size_t j = 0; j < size_QlDrop; j++) {
                    auto rj = base_QlDrop_->base()[j];
                    uint64_t QlQHatInvModqModrj = modulo_uint(QlQHatInvModq[size_Ql + j].data(), size_Ql + 1, rj);
                    QlQHatInvModqDivqFrac_[j] = static_cast<double>(QlQHatInvModqModrj) /
                                                static_cast<double>(rj.value());
                }

                // compute QlQHatInvModqDivq
                vector<vector<uint64_t>> QlQHatInvModqDivq(size_Q);
                for (size_t i = 0; i < size_Q; i++) {
                    // resize QlQHatInvModqDivq[i] to size_Ql + 1 and initialize to 0
                    QlQHatInvModqDivq[i].resize(size_Ql + 1, 0);
                    // align qi with big integer QlQHatInvModq
                    auto qi = base_Q_->base()[i];
                    vector<uint64_t> bigint_qi(size_Ql + 1, 0);
                    bigint_qi[0] = qi.value();
                    // div qi
                    std::vector<uint64_t> temp_remainder(size_Ql + 1, 0);
                    divide_uint(QlQHatInvModq[i].data(), bigint_qi.data(), size_Ql + 1,
                                QlQHatInvModqDivq[i].data(), temp_remainder.data());
                }

                // compute QlQHatInvModqDivqModq
                for (size_t i = 0; i < size_Ql; i++) {
                    auto &qi = base_Ql_->base()[i];
                    for (size_t j = 0; j < size_QlDrop; j++) {
                        // mod qi
                        uint64_t QlQHatInvModqDivrjModqi = modulo_uint(QlQHatInvModqDivq[size_Ql + j].data(),
                                                                       size_Ql + 1, qi);
                        QlQHatInvModqDivqModq_[i * (size_QlDrop + 1) + j].set(QlQHatInvModqDivrjModqi, qi);
                    }
                    // mod qi
                    uint64_t QlQHatInvModqDivqiModqi = modulo_uint(QlQHatInvModqDivq[i].data(), size_Ql + 1, qi);
                    QlQHatInvModqDivqModq_[i * (size_QlDrop + 1) + size_QlDrop].set(QlQHatInvModqDivqiModqi, qi);
                }
            }
        }
    }
} // namespace cahel::util
