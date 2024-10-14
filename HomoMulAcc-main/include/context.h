#pragma once

#include "util/encryptionparams.h"
#include "util/modulus.h"
#include "util/ntt.h"
#include <memory>
#include <unordered_map>
#include <utility>
#include "util/rns.h"
#include "util/uintarithsmallmod.h"

namespace cahel {
    /** Record the following information
     * 1. paramter checking result
     * 2. whether FFT supports (must be true)
     * 3. whether NTT supports (must be true)
     * 4. whether batch supports (must be true)
     * 5. whether supports optimization for cipher * plain,
     *           requiring plaintext modulus is smaller than each prime in the coefficient modulus
     * 6. whether supports optimization for reducing the modular reduction,
     *           requring the coefficient modulus in decreasing order
     * 7. security level
     */
    class EncryptionParameterQualifiers {
    public:
        enum class error_type : int {
            // constructed but not yet validated
            none = -1,

            // valid
            success = 0,

            // scheme must be BFV or CKKS
            invalid_scheme = 1,

            // coeff_modulus's primes' count is not bounded by CAHEL_COEFF_MOD_COUNT_MIN(MAX)
            invalid_coeff_modulus_size = 2,

            // coeff_modulus's primes' bit counts are not bounded by CAHEL_USER_MOD_BIT_COUNT_MIN(MAX)
            invalid_coeff_modulus_bit_count = 3,

            // coeff_modulus's primes are not congruent to 1 modulo (2 * poly_modulus_degree)
            invalid_coeff_modulus_no_ntt = 4,

            // poly_modulus_degree is not bounded by CAHEL_POLY_MOD_DEGREE_MIN(MAX)
            invalid_poly_modulus_degree = 5,

            // poly_modulus_degree is not a power of two
            invalid_poly_modulus_degree_non_power_of_two = 6,

            // parameters are too large to fit in size_t type
            invalid_parameters_too_large = 7,

            // parameters are not compliant with HomomorphicEncryption.org security standard
            invalid_parameters_insecure = 8,

            // RNSBase cannot be constructed
            failed_creating_rns_base = 9,

            // plain_modulus's bit count is not bounded by CAHEL_PLAIN_MOD_BIT_COUNT_MIN(MAX)
            invalid_plain_modulus_bit_count = 10,

            // plain_modulus is not coprime to coeff_modulus
            invalid_plain_modulus_coprimality = 11,

            // plain_modulus is not smaller than coeff_modulus
            invalid_plain_modulus_too_large = 12,

            // plain_modulus is not zero
            invalid_plain_modulus_nonzero = 13,

            // RNSTool cannot be constructed
            failed_creating_rns_tool = 14,
        };

        /**
        The variable parameter_error is set to:
        - none, if parameters are not validated;
        - success, if parameters are considered valid;
        - other values, if parameters are validated and invalid.
        */
        error_type parameter_error;

        /**
        Returns the name of parameter_error.
        */
        [[nodiscard]] const char *parameter_error_name() const noexcept;

        /**
        Returns a comprehensive message that interprets parameter_error.
        */
        [[nodiscard]] const char *parameter_error_message() const noexcept;

        /**
        Tells whether parameter_error is error_type::success.
        */
        [[nodiscard]] inline bool parameters_set() const noexcept {
            return parameter_error == error_type::success;
        }

        /**
         * true, means supportting fast polynomials multiplication modulo the polynomial modulus,
         *         requring polynomial modulus is of the form X^N+1, where N is a power of two.
         */
        bool using_fft;

        /**
         * true, means supporting fast multiplications of polynomials modulo the polynomial modulus
         *         and coefficient modulus,
         *     requring coefficient modulus are congruent to 1 modulo 2N,
         *        where X^N+1 is the polynomial modulus and N is a power of two.
         */
        bool using_ntt;

        /**
         * true, means support batching, requring plaintext modulus is congruent to 1 modulo 2N,
         *     where X^N+1 is the polynomial modulus and N is a power of two.
         *     then, plaintext elements be viewed as 2-by-(N/2) matrices of integers modulo the plaintext modulus.
         */
        bool using_batching;

        /**
         * true, means supportting a faster ciphertext * plaintext and NTT transformation for plaintext.
         *     requring plaintext modulus is smaller than each prime in the coefficient modulus.
         */
        bool using_fast_plain_lift;

        /**
         * true, means certain modular reductions in base conversion can be omitted.
         *     requring primes in the coefficient modulus are in decreasing order.
         */
        bool using_descending_modulus_chain;

        /**
         * security level according to HomomorphicEncryption.org security standard.
         */
        sec_level_type sec_level;

    private:
        EncryptionParameterQualifiers()
                : parameter_error(error_type::none), using_fft(false), using_ntt(false), using_batching(false),
                  using_fast_plain_lift(false), using_descending_modulus_chain(false), sec_level(sec_level_type::none) {
        }

        friend class CAHELContext;
    };

    /**
     * perform validation and pre-computations for the given encryption parameters.
     * 1. validation result is stored in EncryptionParameterQualifiers,
     * 2. pre-computations are stored in the vector<ContextData>.
     */
    class CAHELContext {
    public:
        // stores pre-computation data for a given set of encryption parameters.
        class ContextData {
            friend class CAHELContext;

        public:
            ContextData() = delete;

            ContextData(const ContextData &copy) = delete;

            ContextData(ContextData &&move) = default;

            ContextData &operator=(ContextData &&move) = default;

            // Returns a const reference to the underlying encryption parameters.
            [[nodiscard]] inline auto &parms() const noexcept {
                return parms_;
            }

            // Returns a copy of EncryptionParameterQualifiers (validation result)
            [[nodiscard]] inline auto qualifiers() const noexcept {
                return qualifiers_;
            }

            /**
             * Returns a vector of uint64_t, which is a pre-computed product of all primes in the coefficient modulus.
             * The bit-length of this product is used to evaluate the security level (with the degree of polynomial modulus)
             */
            [[nodiscard]] inline auto &total_coeff_modulus() const noexcept {
                return total_coeff_modulus_;
            }

            // Returns the bit-length of the product of all primes in the coefficient modulus.
            [[nodiscard]] inline int total_coeff_modulus_bit_count() const noexcept {
                return total_coeff_modulus_bit_count_;
            }

            [[nodiscard]] inline const util::RNSTool *rns_tool() const noexcept {
                return rns_tool_.get();
            }

            // Returns std::vector<small_ntt_tables_>
            [[nodiscard]] inline auto &small_ntt_tables() const noexcept {
                return small_ntt_tables_;
            }

            // Returns std::vector<plain_ntt_tables_>
            [[nodiscard]] inline auto &plain_ntt_tables() const noexcept {
                return plain_ntt_tables_;
            }

            // Returns shared_ptr<BFV "Delta"> i.e. coefficient modulus divided by
            [[nodiscard]] inline auto coeff_div_plain_modulus() const noexcept {
                return coeff_div_plain_modulus_;
            }

            [[nodiscard]] inline auto plain_modulus_shoup() const noexcept {
                return plain_modulus_shoup_;
            }

            // Returns (plain_modulus + 1) / 2.
            [[nodiscard]] inline auto plain_upper_half_threshold() const noexcept {
                return plain_upper_half_threshold_;
            }

            /**
             * Return a vector<uint64_t>, the plaintext upper half increment,
             * i.e. coeff_modulus minus plain_modulus.
             * The upper half increment is represented as an integer for the full product coeff_modulus if using_fast_plain_lift is false;
             * and is otherwise represented modulo each of the coeff_modulus primes in order.
             */
            [[nodiscard]] inline auto &plain_upper_half_increment() const noexcept {
                return plain_upper_half_increment_;
            }

            /**
             * Return a vector<uint64_t>, the upper half threshold with respect to the total coefficient modulus.
             * This is needed in CKKS decryption.
             */
            [[nodiscard]] inline auto &upper_half_threshold() const noexcept {
                return upper_half_threshold_;
            }

            /**
             * Return a vector of uint64_t, which is r_t(q), used for the negative value.
             *   the upper half increment used for computing Delta*m and converting the coefficients to modulo coeff_modulus.
             * For example, t-1 in plaintext should change into q - Delta = Delta*t + r_t(q) - Delta = Delta*(t-1) + r_t(q)
             * so multiplying the message by Delta is not enough and requires also an addition of r_t(q).
             * This is precisely the upper_half_increment.
             * Note that this operation is "only done for negative message coefficients", i.e. those that exceed plain_upper_half_threshold.
             */
            [[nodiscard]] inline auto &upper_half_increment() const noexcept {
                return upper_half_increment_;
            }

            // Return the non-RNS form of upper_half_increment which is q (coeff) mod t (plain)
            [[nodiscard]] inline auto coeff_modulus_mod_plain_modulus() const noexcept -> std::uint64_t {
                return coeff_modulus_mod_plain_modulus_;
            }

            // Return the index (start from 0) for the parameters, when context chain is generated
            [[nodiscard]] inline std::size_t chain_index() const noexcept {
                return chain_index_;
            }

            /** Return the ntt tables copy flag
             * i.e., whether the ntt table (and inverse) has been copied to GPU
             */
            inline auto copy_ntttable() const noexcept {
                return copy_ntttable_;
            }

            /** Set the ntt tables copy flag
             * i.e., means this ntt table (and inverse) has been copied to GPU
             */
            inline auto set_copy_ntttable(bool flag = true) {
                copy_ntttable_ = flag;
            }

            /** Return the coeff modulus flag
             * i.e., whether the coeff modulus has been copied to GPU
             */
            inline auto copy_coeff_modulus() {
                return copy_coeff_modulus_;
            }

            /** Set the coeff modulus flag
             * i.e., means this coeff modulus has been copied to GPU
             */
            inline auto set_copy_coeff_modulus(bool flag = true) {
                copy_coeff_modulus_ = flag;
            }

            /** Return the root_powers_ and inverse_root_powers of small_NTTtable
             * "after the small_table has been computed"
             * @param[in] N Poly degree
             * @param[in] coeff_mod_size The number of coeff modulus
             * @param[out] root_powers The root_powers for all coeff
             * @param[out] inverse_root_powers The inverse root_powers for all coeff
             */
            inline void get_all_nttables(size_t N,
                                         size_t coeff_mod_size,
                                         std::vector<cahel::util::MultiplyUIntModOperand> &root_powers,
                                         std::vector<cahel::util::MultiplyUIntModOperand> &inverse_root_powers) const {
                auto size = small_ntt_tables_.size();
                if (size == 0)
                    throw std::invalid_argument("small ntt table has not been computed");
                if (size != coeff_mod_size ||
                    N != small_ntt_tables_[0].get_from_root_powers().size())
                    throw std::invalid_argument("small ntt table does not match the parameter");
                root_powers.resize(N * coeff_mod_size);
                inverse_root_powers.resize(N * coeff_mod_size);
                cahel::util::MultiplyUIntModOperand *root_ptr = root_powers.data();
                cahel::util::MultiplyUIntModOperand *inverse_root_ptr = inverse_root_powers.data();
                for (size_t i = 0; i < small_ntt_tables_.size(); i++) {
                    std::copy_n(root_ptr + i * N, N,
                                (cahel::util::MultiplyUIntModOperand *) (small_ntt_tables_[i].get_from_root_powers().data()));
                    std::copy_n(inverse_root_ptr + i * N, N,
                                (cahel::util::MultiplyUIntModOperand *) (small_ntt_tables_[i].get_from_inv_root_powers().data()));
                }
                return;
            }

        private:
            explicit ContextData(EncryptionParameters parms) : parms_(std::move(parms)) {}

            EncryptionParameters parms_;

            EncryptionParameterQualifiers qualifiers_;

            std::shared_ptr<util::RNSTool> rns_tool_;

            std::vector<util::NTTTables> small_ntt_tables_;

            std::vector<util::NTTTables> plain_ntt_tables_;

            // std::shared_ptr<util::GaloisTool> galois_tool_;

            std::vector<std::uint64_t> total_coeff_modulus_;

            int total_coeff_modulus_bit_count_ = 0;

            std::vector<util::MultiplyUIntModOperand> coeff_div_plain_modulus_;

            std::vector<util::MultiplyUIntModOperand> plain_modulus_shoup_;

            std::uint64_t plain_upper_half_threshold_ = 0;

            std::vector<std::uint64_t> plain_upper_half_increment_;

            std::vector<std::uint64_t> upper_half_threshold_;

            std::vector<std::uint64_t> upper_half_increment_;

            std::uint64_t coeff_modulus_mod_plain_modulus_ = 0;

            std::size_t chain_index_ = 0;

            // Whether the NTT Table(and inverse) has been copied to GPU
            bool copy_ntttable_ = false;
            // Whether the coeff modulus has been copied to GPU
            bool copy_coeff_modulus_ = false;
        };

        /**
        Creates an instance of CAHELContext and performs several pre-computations
        on the given EncryptionParameters.

        @param[in] parms The encryption parameters
        @param[in] expand_mod_chain Determines whether the modulus switching chain
        should be created
        @param[in] sec_level Determines whether a specific security level should be
        enforced according to HomomorphicEncryption.org security standard
        */
        explicit CAHELContext(
                EncryptionParameters parms, bool expand_mod_chain = true,
                sec_level_type sec_level = sec_level_type::tc128);

        /**
        Creates a new CAHELContext by copying a given one.

        @param[in] copy The CAHELContext to copy from
        */
        CAHELContext(const CAHELContext &copy) = default;

        /**
        Creates a new CAHELContext by moving a given one.

        @param[in] source The CAHELContext to move from
        */
        CAHELContext(CAHELContext &&source) = default;

        /**
        Copies a given CAHELContext to the current one.

        @param[in] assign The CAHELContext to copy from
        */
        CAHELContext &operator=(const CAHELContext &assign) = default;

        /**
        Moves a given CAHELContext to the current one.

        @param[in] assign The CAHELContext to move from
        */
        CAHELContext &operator=(CAHELContext &&assign) = default;

        /**
        Returns a parms_id_type corresponding to the set of encryption parameters
        that are used for keys.
        */
        [[nodiscard]] inline auto &key_parms_id() const {
            if (parms_ids_.size() >= 1) {
                return parms_ids_[0];
            } else {
                throw std::logic_error("the params_ids have not been computed");
            }
        }

        /**
        Returns a parms_id_type corresponding to the first encryption parameters
        that are used for data.
        */
        [[nodiscard]] inline auto &first_parms_id() const {
            if (parms_ids_.size() >= 2) {
                return parms_ids_[1];
            } else {
                throw std::logic_error("the params_ids have not been computed");
            }
        }

        /**
        Returns a parms_id_type corresponding to the last encryption parameters
        that are used for data.
        */
        [[nodiscard]] inline auto &last_parms_id() const {
            auto parms_ids_size = parms_ids_.size();
            if (parms_ids_size > 0) {
                return parms_ids_.at(parms_ids_size - 1);
            } else {
                throw std::logic_error("the params_ids have not been computed");
            }
        }

        [[nodiscard]] inline auto get_chain_index(parms_id_type parms_id) const {
            auto iter = find(parms_ids_.begin(), parms_ids_.end(), parms_id);
            if (iter != parms_ids_.end()) {
                auto parms_id_index = iter - parms_ids_.begin();
                return parms_id_index;
            } else {
                throw std::invalid_argument("parms_id is invalid!");
            }
        }

        /**
        Returns the ContextData corresponding to encryption parameters with a given
        parms_id. If parameters with the given parms_id are not found then the
        function returns nullptr.

        @param[in] parms_id The parms_id of the encryption parameters
        */
        [[nodiscard]] inline auto &get_context_data(parms_id_type parms_id) const {
            auto iter = find(parms_ids_.begin(), parms_ids_.end(), parms_id);
            if (iter != parms_ids_.end()) {
                auto parms_id_index = iter - parms_ids_.begin();
                return context_data_[parms_id_index];
            } else {
                throw std::invalid_argument("parms_id is invalid!");
            }
        }

        /**
         * Return the contextdata for the provided index,
         * we do not use the parm id for index for simple
         * The parm id is better for obtaining the corresponding context data for a paramter
        @param[in] index The index of context chain
        @param[out] ContextData Return Value
        */
        [[nodiscard]] inline auto &get_context_data(size_t index) const {
            if (index < context_data_.size()) {
                return context_data_[index];
            } else {
                throw std::invalid_argument("index is invalid!");
            }
        }

        /**
         * Returns the ContextData corresponding to encryption parameters that are
         * used for keys.
         */
        [[nodiscard]] inline auto &key_context_data() const {
            auto context_data_size = context_data_.size();
            if (context_data_size == 0) {
                throw std::invalid_argument("context_data is null!");
            }
            return context_data_[0];
        }

        [[nodiscard]] inline auto &first_context_data() const {
            auto context_data_size = context_data_.size();
            if (context_data_size == 0) {
                throw std::invalid_argument("context_data is null!");
            }
            return context_data_[static_cast<size_t>(1)];
        }

        [[nodiscard]] inline auto &last_context_data() const {
            auto context_data_size = context_data_.size();
            if (context_data_size == 0) {
                throw std::invalid_argument("context_data is null!");
            }
            return context_data_[context_data_size - 1];
        }

        [[nodiscard]] inline auto get_parms_id(size_t index) const {
            if (index < parms_ids_.size()) {
                return parms_ids_[index];
            } else {
                throw std::invalid_argument("index is invalid!!");
            }
        }

        [[nodiscard]] inline auto &get_next_parms_id(parms_id_type parms_id) const {
            auto iter = find(parms_ids_.begin(), parms_ids_.end(), parms_id);
            if (iter != parms_ids_.end()) {
                auto parms_id_index = iter - parms_ids_.begin();
                if (static_cast<size_t>(parms_id_index + 1) <= (parms_ids_.size() - 1))
                    return parms_ids_[parms_id_index + 1];
                else
                    throw std::invalid_argument("There exists no next parms_id!");
            } else {
                throw std::invalid_argument("parms_id is invalid!");
            }
        }

        /**
         * Return the contextdata for the provided index,
         * we do not use the parm id for index for simple
         * The parm id is better for obtaining the corresponding context data for a paramter
         @param[in] index The index of context chain
         @param[out] ContextData Return Value
        */
        [[nodiscard]] inline auto &get_context_data_rns_tool(size_t index) const {
            if (index < context_data_.size()) {
                return context_data_[index].rns_tool_;
            } else {
                throw std::invalid_argument("index is invalid!!!");
            }
        }

        // Returns whether the encryption parameters are valid.
        [[nodiscard]] inline auto parameters_set() const {
            if (context_data_.size() == 0)
                return false;
            else
                return context_data_[current_parm_index_].qualifiers_.parameters_set();
        }

        // Returns the name of encryption parameters' error.
        [[nodiscard]] inline const char *parameter_error_name() const {
            if (context_data_.size() == 0)
                return "CAHELContext is empty";
            else
                return context_data_[current_parm_index_].qualifiers_.parameter_error_name();
        }

        // Returns a comprehensive message that interprets encryption parameters' error.
        [[nodiscard]] inline const char *parameter_error_message() const {
            if (context_data_.size() == 0)
                return "CAHELContext is empty";
            else
                return context_data_[current_parm_index_].qualifiers_.parameter_error_message();
        }

        // Returns the current using parm index.
        [[nodiscard]] inline auto current_parm_index() const noexcept {
            return current_parm_index_;
        }

        // Returns the first parm index.
        [[nodiscard]] inline auto first_parm_index() const noexcept {
            return first_parm_index_;
        }

        // Returns the first parm index.
        [[nodiscard]] inline size_t previous_parm_index(size_t index) const {
            if (index >= context_data_.size())
                throw std::invalid_argument("index not valid");
            if (index < 1)
                return 0;
            else
                return index - 1;
        }

        // Returns the first parm index.
        [[nodiscard]] inline auto next_parm_index(size_t index) const {
            if (index >= (context_data_.size() - 1))
                throw std::invalid_argument("index not valid");
            else
                return index + 1;
        }

        // Returns the total number of parm index.
        [[nodiscard]] inline auto total_parm_size() const noexcept {
            return context_data_.size();
        }

        /**
         * true, when coefficient modulus parameter consists of at least two prime number factors
         *     then, supports keyswitching (which is required for relinearize, rotation, conjugation.)
         */
        [[nodiscard]] inline bool using_keyswitching() const noexcept {
            return using_keyswitching_;
        }

        [[nodiscard]] inline auto mul_tech() const noexcept {
            return mul_tech_;
        }

    private:
        ContextData validate(EncryptionParameters parms);

        // currently using contextData index
        size_t current_parm_index_ = 0;

        // first parms id, i.e., the first one used in Public key encryption
        // top data level
        size_t first_parm_index_ = 0;

        // notice: seal, maintains key_parms_id_, first_parms_id_, and last_parms_id_
        // in our lib, key_parms_id_ is always 0, first_parms_id_ is calculated as in seal (either 0 or 1),
        // while last_parms_id_ is context_data_.size().
        std::vector<parms_id_type> parms_ids_;

        std::vector<ContextData> context_data_;

        // security level, according HomomorphicEncryption.org
        sec_level_type sec_level_;

        // whether keyswitching supported by the encryption parameters?
        bool using_keyswitching_;

        mul_tech_type mul_tech_;
    };

    /** Check the provided scale is valid for provided config
    @param[in] scale The new scale
    @param[in] context The context data
    @param[in] parms The parms info
    @return boolean true for valid, false for invalid
    **/
    inline bool is_scale_within_bounds(double scale, const CAHELContext::ContextData &context_data,
                                       const EncryptionParameters &parms) {
        int scale_bit_count_bound = 0;
        switch (parms.scheme()) {
            case scheme_type::bfv:
            case scheme_type::bgv:
                scale_bit_count_bound = parms.plain_modulus().bit_count();
                break;
            case scheme_type::ckks:
                scale_bit_count_bound = context_data.total_coeff_modulus_bit_count();
                break;
            default:
                // Unsupported scheme; check will fail
                scale_bit_count_bound = -1;
                break;
        }
        return !(scale <= 0 || (static_cast<int>(log2(scale)) >= scale_bit_count_bound));
    }

} // namespace cahel
