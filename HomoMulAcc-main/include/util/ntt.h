#pragma once
#include "modulus.h"
#include "defines.h"
#include "uintarithsmallmod.h"
#include "uintcore.h"
#include <stdexcept>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <memory>

namespace cahel
{
    namespace util
    {
        class NTTTables
        {
        public:
            NTTTables(NTTTables &&source) = default;

            NTTTables(NTTTables &copy)
                : root_(copy.root_), coeff_count_power_(copy.coeff_count_power_),
                  coeff_count_(copy.coeff_count_), modulus_(copy.modulus_), inv_degree_modulo_(copy.inv_degree_modulo_)
            {
                root_powers_.resize(coeff_count_);
                inv_root_powers_.resize(coeff_count_);
                std::copy_n(copy.root_powers_.data(), coeff_count_, root_powers_.data());
                std::copy_n(copy.inv_root_powers_.data(), coeff_count_, inv_root_powers_.data());
            }

            NTTTables(int coeff_count_power, const Modulus &modulus);

            [[nodiscard]] inline std::uint64_t get_root() const
            {
                return root_;
            }

            [[nodiscard]] inline const auto &get_from_root_powers() const
            {
                return root_powers_;
            }

            [[nodiscard]] inline const auto &get_from_inv_root_powers() const
            {
                return inv_root_powers_;
            }

            [[nodiscard]] inline const auto &get_from_inv_root_powers_div2() const
            {
                return inv_root_powers_div2_;
            }

            [[nodiscard]] inline MultiplyUIntModOperand get_from_root_powers(std::size_t index) const
            {
                return root_powers_[index];
            }

            [[nodiscard]] inline MultiplyUIntModOperand get_from_inv_root_powers(std::size_t index) const
            {
                return inv_root_powers_[index];
            }

            [[nodiscard]] inline const MultiplyUIntModOperand &inv_degree_modulo() const
            {
                return inv_degree_modulo_;
            }

            [[nodiscard]] inline const Modulus &modulus() const
            {
                return modulus_;
            }

            [[nodiscard]] inline int coeff_count_power() const
            {
                return coeff_count_power_;
            }

            [[nodiscard]] inline std::size_t coeff_count() const
            {
                return coeff_count_;
            }

        private:
            NTTTables &operator=(const NTTTables &assign) = delete;

            NTTTables &operator=(NTTTables &&assign) = delete;

            std::uint64_t root_ = 0;

            std::uint64_t inv_root_ = 0;

            int coeff_count_power_ = 0;

            std::size_t coeff_count_ = 0;

            Modulus modulus_;

            // Inverse of coeff_count_ modulo modulus_.
            MultiplyUIntModOperand inv_degree_modulo_;

            // Holds 1~(n-1)-th powers of root_ in bit-reversed order, the 0-th power is left unset.
            std::vector<MultiplyUIntModOperand> root_powers_;

            // Holds 1~(n-1)-th powers of inv_root_ in scrambled order, the 0-th power is left unset.
            std::vector<MultiplyUIntModOperand> inv_root_powers_;

            // Holds 1~(n-1)-th powers of inv_root_ in scrambled order, the 0-th power is left unset.
            std::vector<MultiplyUIntModOperand> inv_root_powers_div2_;
        };

        /** Create NTTTable for each modulus
          * @param[in] coeff_count_power Log N, where N is the polynomial degree
          * @param[in] modulus The coeff modulus.
          * @param[inout] tables The generated NTTTable vector
        @throws std::invalid_argument if modulus is empty, modulus does not support NTT, coeff_count_power is invalid
        */
        void CreateNTTTables(
            int coeff_count_power, const std::vector<Modulus> &modulus, std::vector<NTTTables> &tables);

        void CreateNTTTables(
            int coeff_count_power, const Modulus &modulus, std::shared_ptr<NTTTables> &tables);
    } // namespace util
} // namespace cahel
