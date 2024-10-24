#include "context.h"
#include "util/uintarith.h"
#include "util/numth.h"
#include "util/polycore.h"
#include "util/uintarith.h"
#include "util/uintarithsmallmod.h"
#include "util/common.h"
#include <algorithm>
#include <stdexcept>
#include <utility>

using namespace std;
using namespace cahel::util;

using error_type = cahel::EncryptionParameterQualifiers::error_type;

namespace cahel {
    const char *EncryptionParameterQualifiers::parameter_error_name() const noexcept {
        switch (parameter_error) {
            case error_type::none:
                return "none";

            case error_type::success:
                return "success";

            case error_type::invalid_scheme:
                return "invalid_scheme";

            case error_type::invalid_coeff_modulus_size:
                return "invalid_coeff_modulus_size";

            case error_type::invalid_coeff_modulus_bit_count:
                return "invalid_coeff_modulus_bit_count";

            case error_type::invalid_coeff_modulus_no_ntt:
                return "invalid_coeff_modulus_no_ntt";

            case error_type::invalid_poly_modulus_degree:
                return "invalid_poly_modulus_degree";

            case error_type::invalid_poly_modulus_degree_non_power_of_two:
                return "invalid_poly_modulus_degree_non_power_of_two";

            case error_type::invalid_parameters_too_large:
                return "invalid_parameters_too_large";

            case error_type::invalid_parameters_insecure:
                return "invalid_parameters_insecure";

            case error_type::failed_creating_rns_base:
                return "failed_creating_rns_base";

            case error_type::invalid_plain_modulus_bit_count:
                return "invalid_plain_modulus_bit_count";

            case error_type::invalid_plain_modulus_coprimality:
                return "invalid_plain_modulus_coprimality";

            case error_type::invalid_plain_modulus_too_large:
                return "invalid_plain_modulus_too_large";

            case error_type::invalid_plain_modulus_nonzero:
                return "invalid_plain_modulus_nonzero";

            case error_type::failed_creating_rns_tool:
                return "failed_creating_rns_tool";

            default:
                return "invalid parameter_error";
        }
    }

    const char *EncryptionParameterQualifiers::parameter_error_message() const noexcept {
        switch (parameter_error) {
            case error_type::none:
                return "constructed but not yet validated";

            case error_type::success:
                return "valid";

            case error_type::invalid_scheme:
                return "scheme must be BFV or CKKS";

            case error_type::invalid_coeff_modulus_size:
                return "coeff_modulus's primes' count is not bounded by CAHEL_COEFF_MOD_COUNT_MIN(MAX)";

            case error_type::invalid_coeff_modulus_bit_count:
                return "coeff_modulus's primes' bit counts are not bounded by CAHEL_USER_MOD_BIT_COUNT_MIN(MAX)";

            case error_type::invalid_coeff_modulus_no_ntt:
                return "coeff_modulus's primes are not congruent to 1 modulo (2 * poly_modulus_degree)";

            case error_type::invalid_poly_modulus_degree:
                return "poly_modulus_degree is not bounded by CAHEL_POLY_MOD_DEGREE_MIN(MAX)";

            case error_type::invalid_poly_modulus_degree_non_power_of_two:
                return "poly_modulus_degree is not a power of two";

            case error_type::invalid_parameters_too_large:
                return "parameters are too large to fit in size_t type";

            case error_type::invalid_parameters_insecure:
                return "parameters are not compliant with HomomorphicEncryption.org security standard";

            case error_type::failed_creating_rns_base:
                return "RNSBase cannot be constructed";

            case error_type::invalid_plain_modulus_bit_count:
                return "plain_modulus's bit count is not bounded by CAHEL_PLAIN_MOD_BIT_COUNT_MIN(MAX)";

            case error_type::invalid_plain_modulus_coprimality:
                return "plain_modulus is not coprime to coeff_modulus";

            case error_type::invalid_plain_modulus_too_large:
                return "plain_modulus is not smaller than coeff_modulus";

            case error_type::invalid_plain_modulus_nonzero:
                return "plain_modulus is not zero";

            case error_type::failed_creating_rns_tool:
                return "RNSTool cannot be constructed";

            default:
                return "invalid parameter_error";
        }
    }

    CAHELContext::ContextData CAHELContext::validate(EncryptionParameters parms) {
        ContextData context_data(parms);
        context_data.qualifiers_.parameter_error = error_type::success;

        auto &key_modulus = parms.key_modulus();
        auto &coeff_modulus = parms.coeff_modulus();
        auto &plain_modulus = parms.plain_modulus();

        size_t special_modulus_size = parms.special_modulus_size();

        // we support 1 - 64 coeff modulus
        if (coeff_modulus.size() > CAHEL_COEFF_MOD_COUNT_MAX || coeff_modulus.size() < CAHEL_COEFF_MOD_COUNT_MIN) {
            context_data.qualifiers_.parameter_error = error_type::invalid_coeff_modulus_size;
            return context_data;
        }

        size_t coeff_modulus_size = coeff_modulus.size();
        for (size_t i = 0; i < coeff_modulus_size; i++) {
            // each coeff is at most 60 bits, at least 2 bits
            if (coeff_modulus[i].value() >> CAHEL_USER_MOD_BIT_COUNT_MAX ||
                !(coeff_modulus[i].value() >> (CAHEL_USER_MOD_BIT_COUNT_MIN - 1))) {
                context_data.qualifiers_.parameter_error = error_type::invalid_coeff_modulus_bit_count;
                return context_data;
            }
        }

        // Compute the product of all coeff modulus
        context_data.total_coeff_modulus_ = std::vector<uint64_t>(coeff_modulus_size);
        auto coeff_modulus_values = std::vector<uint64_t>(coeff_modulus_size);
        for (size_t i = 0; i < coeff_modulus_size; i++) {
            coeff_modulus_values[i] = coeff_modulus[i].value();
        }
        multiply_many_uint64(coeff_modulus_values.data(), coeff_modulus_size, context_data.total_coeff_modulus_.data());
        context_data.total_coeff_modulus_bit_count_ = get_significant_bit_count_uint(
                context_data.total_coeff_modulus_.data(), coeff_modulus_size);

        // polynomial modulus degree should be in 2 - 65536*2
        size_t poly_modulus_degree = parms.poly_modulus_degree();
        if (poly_modulus_degree < CAHEL_POLY_MOD_DEGREE_MIN || poly_modulus_degree > CAHEL_POLY_MOD_DEGREE_MAX)
            throw std::invalid_argument("invalid poly_modulus_degree");

        // coeff_count_power = log(N)
        int coeff_count_power = get_power_of_two(poly_modulus_degree);
        if (coeff_count_power < 0)
            throw std::invalid_argument("invalid poly_modulus_degree");

        context_data.qualifiers_.using_fft = true;

        context_data.qualifiers_.sec_level = sec_level_;

        // Check if the parameters are secure according to HomomorphicEncryption.org security standard
        if (context_data.total_coeff_modulus_bit_count_ > CoeffModulus::MaxBitCount(poly_modulus_degree, sec_level_)) {
            context_data.qualifiers_.sec_level = sec_level_type::none;
            if (sec_level_ != sec_level_type::none)
                throw std::invalid_argument("invalid parameters_insecure");
        }

        std::shared_ptr<RNSBase> coeff_modulus_base = std::make_shared<RNSBase>(RNSBase(coeff_modulus));

        // Create NTTTable and check whether the coeff modulus are set correctly
        context_data.qualifiers_.using_ntt = true;

        CreateNTTTables(coeff_count_power, coeff_modulus, context_data.small_ntt_tables_);

        if (parms.scheme() == scheme_type::bfv || parms.scheme() == scheme_type::bgv) {
            // Plain modulus must be at least 2 and at most 60 bits
            if (plain_modulus.value() >> CAHEL_PLAIN_MOD_BIT_COUNT_MAX ||
                !(plain_modulus.value() >> (CAHEL_PLAIN_MOD_BIT_COUNT_MIN - 1))) {
                context_data.qualifiers_.parameter_error = error_type::invalid_plain_modulus_bit_count;
                return context_data;
            }

            // plain_modulus should be coprime with each coeff modulus
            for (size_t i = 0; i < coeff_modulus_size; i++) {
                if (!are_coprime(coeff_modulus[i].value(), plain_modulus.value())) {
                    context_data.qualifiers_.parameter_error = error_type::invalid_plain_modulus_coprimality;
                    return context_data;
                }
            }
            // Check that plain_modulus is smaller than total coeff modulus
            if (!is_less_than_uint(
                    plain_modulus.data(), plain_modulus.uint64_count(), context_data.total_coeff_modulus_.data(),
                    coeff_modulus_size)) {
                // Parameters are not valid
                context_data.qualifiers_.parameter_error = error_type::invalid_plain_modulus_too_large;
                return context_data;
            }

            // Can we use batching? (NTT with plain_modulus)
            context_data.qualifiers_.using_batching = true;
            try {
                CreateNTTTables(coeff_count_power, {plain_modulus}, context_data.plain_ntt_tables_);
            }
            catch (const invalid_argument &) {
                context_data.qualifiers_.using_batching = false;
            }

            // Check for plain_lift, requring plain modulus smaller than all coeff modulus
            context_data.qualifiers_.using_fast_plain_lift = true;
            for (size_t i = 0; i < coeff_modulus_size; i++) {
                context_data.qualifiers_.using_fast_plain_lift &= (coeff_modulus[i].value() > plain_modulus.value());
            }

            // Calculate coeff_div_plain_modulus (BFV-"Delta") and the remainder upper_half_increment
            auto temp_coeff_div_plain_modulus = std::vector<uint64_t>(coeff_modulus_size);
            auto wide_plain_modulus = std::vector<uint64_t>(coeff_modulus_size);
            context_data.coeff_div_plain_modulus_.resize(coeff_modulus_size);
            context_data.plain_modulus_shoup_.resize(coeff_modulus_size);
            context_data.upper_half_increment_.resize(coeff_modulus_size);
            wide_plain_modulus[0] = plain_modulus.value();
            // temp_coeff_div_plain_modulus = total_coeff_modulus_ / wide_plain_modulus,
            // upper_half_increment_ is the remainder
            divide_uint(
                    context_data.total_coeff_modulus_.data(), wide_plain_modulus.data(), coeff_modulus_size,
                    temp_coeff_div_plain_modulus.data(), context_data.upper_half_increment_.data());

            // Store the non-RNS form of upper_half_increment for BFV encryption
            context_data.coeff_modulus_mod_plain_modulus_ = context_data.upper_half_increment_[0];

            // Decompose coeff_div_plain_modulus into RNS factors
            coeff_modulus_base->decompose(temp_coeff_div_plain_modulus.data());

            for (size_t i = 0; i < coeff_modulus_size; i++) {
                context_data.coeff_div_plain_modulus_[i].set(
                        temp_coeff_div_plain_modulus[i],
                        coeff_modulus_base->base()[i]);

                context_data.plain_modulus_shoup_[i].set(
                        plain_modulus.value(),
                        coeff_modulus_base->base()[i]);
            }

            // Decompose upper_half_increment into RNS factors
            coeff_modulus_base->decompose(context_data.upper_half_increment_.data());

            // Calculate (plain_modulus + 1) / 2.
            context_data.plain_upper_half_threshold_ = (plain_modulus.value() + 1) >> 1;

            // Calculate coeff_modulus - plain_modulus.
            context_data.plain_upper_half_increment_.resize(coeff_modulus_size);
            if (context_data.qualifiers_.using_fast_plain_lift) {
                // Calculate coeff_modulus[i] - plain_modulus if using_fast_plain_lift
                for (size_t i = 0; i < coeff_modulus_size; i++) {
                    context_data.plain_upper_half_increment_[i] = coeff_modulus[i].value() - plain_modulus.value();
                }
            } else {
                // compared with using_fast_plain_lift=true, here requires large number computation
                sub_uint(
                        context_data.total_coeff_modulus_.data(), wide_plain_modulus.data(), coeff_modulus_size,
                        context_data.plain_upper_half_increment_.data());
            }
        } else if (parms.scheme() == scheme_type::ckks) {
            // plain_modulus should be zero
            if (!plain_modulus.is_zero()) {
                context_data.qualifiers_.parameter_error = error_type::invalid_plain_modulus_nonzero;
                return context_data;
            }

            // When using CKKS batching (BatchEncoder) is always enabled
            context_data.qualifiers_.using_batching = true;

            // Cannot use fast_plain_lift for CKKS since the plaintext coefficients
            // can easily be larger than coefficient moduli
            context_data.qualifiers_.using_fast_plain_lift = false;

            // Calculate 2^64 / 2 (most negative plaintext coefficient value)
            context_data.plain_upper_half_threshold_ = uint64_t(1) << 63;

            // Calculate plain_upper_half_increment = 2^64 mod coeff_modulus for CKKS plaintexts
            context_data.plain_upper_half_increment_.resize(coeff_modulus_size);
            for (size_t i = 0; i < coeff_modulus_size; i++) {
                // tmp = (1 << 63) % coeff_modulus[i]
                uint64_t tmp = barrett_reduce_64(uint64_t(1) << 63, coeff_modulus[i]);
                // plain_upper_half_increment_[i] = tmp * (coeff_modulus[i] - 2) % coeff_modulus[i]
                context_data.plain_upper_half_increment_[i] =
                        multiply_uint_mod(tmp, sub_safe(coeff_modulus[i].value(), uint64_t(2)), coeff_modulus[i]);
            }

            // Compute the upper_half_threshold for this modulus.
            context_data.upper_half_threshold_.resize(coeff_modulus_size);
            // upper_half_threshold_ = (total_coeff_modulus_ + 1) /2
            increment_uint(
                    context_data.total_coeff_modulus_.data(), coeff_modulus_size,
                    context_data.upper_half_threshold_.data());
            right_shift_uint(
                    context_data.upper_half_threshold_.data(), 1, coeff_modulus_size,
                    context_data.upper_half_threshold_.data());
        } else {
            context_data.qualifiers_.parameter_error = error_type::invalid_scheme;
            return context_data;
        }

        // Create RNSTool
        // RNSTool's constructor may fail due to:
        //   (1) auxiliary base being too large
        //   (2) cannot find inverse of punctured products in auxiliary base
//        try {
        context_data.rns_tool_ = std::make_shared<RNSTool>(poly_modulus_degree,
                                                           special_modulus_size,
                                                           *coeff_modulus_base,
                                                           key_modulus,
                                                           plain_modulus,
                                                           mul_tech_);
//        }
//        catch (const exception &) {
//            // Parameters are not valid
//            context_data.qualifiers_.parameter_error = error_type::failed_creating_rns_tool;
//            return context_data;
//        }

        // Check whether the coefficient modulus consists of a set of primes that are in decreasing order
        context_data.qualifiers_.using_descending_modulus_chain = true;
        for (size_t i = 0; i < coeff_modulus_size - 1; i++) {
            context_data.qualifiers_.using_descending_modulus_chain &=
                    (coeff_modulus[i].value() > coeff_modulus[i + 1].value());
        }

        // Create GaloisTool
        // context_data.galois_tool_ = std::make_shared<GaloisTool>(coeff_count_power);

        // Done with validation and pre-computations
        return context_data;
    }

    CAHELContext::CAHELContext(
            EncryptionParameters parms, bool expand_mod_chain, sec_level_type sec_level)
            : sec_level_(sec_level) {
        // default to 0
        using_keyswitching_ = false;
        mul_tech_ = parms.mul_tech();

        auto &coeff_modulus = parms.coeff_modulus();
        size_t size_P = parms.special_modulus_size();
        size_t size_QP = coeff_modulus.size();

        if (size_QP == 1) {
            parms.set_special_modulus_size(0);
        }

        context_data_.push_back(validate(parms));
        if (size_QP == 1) {
            using_keyswitching_ = false;
            parms_ids_.push_back(context_data_[0].parms().parms_id());
            return;
        }
        if (!context_data_[0].qualifiers_.parameters_set()) {
            using_keyswitching_ = false;
            context_data_.pop_back();
            throw std::invalid_argument("parameter generate failed");
        }
        using_keyswitching_ = true;

        // Drop all special modulus at first data level
        for (size_t i = 0; i < size_P; i++) coeff_modulus.pop_back();

        for (size_t i = 1; expand_mod_chain && (i < size_QP); i++) {
            parms.update_parms_id();
            context_data_.push_back(validate(parms));
            if (!context_data_[i].qualifiers_.parameters_set()) {
                context_data_.pop_back();
                break;
            }
            // Drop one modulus after each data level
            coeff_modulus.pop_back();
        }
        current_parm_index_ = 0;
        first_parm_index_ = 0;
        if (context_data_.size() > 1)
            first_parm_index_ = 1;

        // set the parms_ids
        for (size_t idx = 0; idx < context_data_.size(); idx++) {
            context_data_[idx].chain_index_ = idx;
            parms_ids_.push_back(context_data_.at(idx).parms().parms_id());
        }
    }
} // namespace seal
