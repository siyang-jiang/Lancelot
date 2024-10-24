#include "util/ntt.h"
#include "util/uintarith.h"
#include "util/uintarithsmallmod.h"
#include <algorithm>

using namespace std;

namespace cahel
{
    namespace util
    {
        /** Create NTTTable for each modulus
          * @param[in] coeff_count_power Log N, where N is the polynomial degree
          * @param[in] modulus The coeff modulus.
          * @param[inout] tables The generated NTTTable vector
        @throws std::invalid_argument if modulus is empty, modulus does not support NTT, coeff_count_power is invalid
        */
        void CreateNTTTables(
            int coeff_count_power, const vector<Modulus> &modulus, std::vector<NTTTables> &tables)
        {
            if (!modulus.size())
            {
                throw invalid_argument("invalid modulus");
            }
            for (size_t i = 0; i < modulus.size(); i++)
            {
                tables.push_back(NTTTables(coeff_count_power, modulus[i]));
            }
            return;
        }

        NTTTables::NTTTables(int coeff_count_power, const Modulus &modulus)
        {
            coeff_count_power_ = coeff_count_power;
            coeff_count_ = size_t(1) << coeff_count_power_;
            modulus_ = modulus;
            // We defer parameter checking to try_minimal_primitive_root(...)
            if (!try_minimal_primitive_root(2 * coeff_count_, modulus_, root_))
            {
                throw invalid_argument("invalid modulus in try_minimal_primitive_root");
            }
            if (!try_invert_uint_mod(root_, modulus_, inv_root_))
            {
                throw invalid_argument("invalid modulus in try_invert_uint_mod");
            }
            // Populate tables with powers of root in specific orders.
            root_powers_.resize(coeff_count_);
            MultiplyUIntModOperand root;
            root.set(root_, modulus_);
            uint64_t power = root_;
            for (size_t i = 1; i < coeff_count_; i++)
            {
                root_powers_[reverse_bits(i, coeff_count_power_)].set(power, modulus_);
                power = multiply_uint_mod(power, root, modulus_);
            }
            root_powers_[0].set(static_cast<uint64_t>(1), modulus_);

            inv_root_powers_.resize(coeff_count_);
            inv_root_powers_div2_.resize(coeff_count_);
            root.set(inv_root_, modulus_);
            power = inv_root_;
            uint64_t power_div2 = div2_uint_mod(inv_root_, modulus_);
            for (size_t i = 1; i < coeff_count_; i++)
            {
                inv_root_powers_[reverse_bits(i, coeff_count_power_)].set(power, modulus_);
                inv_root_powers_div2_[reverse_bits(i, coeff_count_power_)].set(power_div2, modulus_);
                power = multiply_uint_mod(power, root, modulus_);
                power_div2 = multiply_uint_mod(power_div2, root, modulus_);
            }
            inv_root_powers_[0].set(static_cast<uint64_t>(1), modulus_);
            inv_root_powers_div2_[0].set(div2_uint_mod(inv_root_, modulus_), modulus_);

            // Compute n^(-1) modulo q.
            uint64_t degree_uint = static_cast<uint64_t>(coeff_count_);
            if (!try_invert_uint_mod(degree_uint, modulus_, inv_degree_modulo_.operand))
            {
                throw invalid_argument("invalid modulus in computing n^(-1) modulo q");
            }
            inv_degree_modulo_.set_quotient(modulus_);
        }
    } // namespace util
} // namespace cahel
