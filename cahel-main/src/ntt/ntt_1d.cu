#include "ntt.cuh"
#include "butterfly.cuh"

/** forward NTT transformation, with N (num of operands) up to 2048,
 * to ensure all operation completed in one block.
 * @param[inout] inout The value to operate and the returned result
 * @param[in] twiddles The pre-computated forward NTT table
 * @param[in] mod The coeff modulus value
 * @param[in] n The poly degreee
 * @param[in] logn The logarithm of n
 * @param[in] numOfGroups
 * @param[in] iter The current iteration in forward NTT transformation
 */
__global__ void inplace_fnwt_radix2(uint64_t *inout,
                                    const DMulUIntMod *twiddles,
                                    const DModulus *modulus,
                                    size_t coeff_mod_size,
                                    size_t start_mod_idx,
                                    size_t n) {
    extern __shared__ uint64_t buffer[];

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n / 2 * coeff_mod_size; // deal with 2 data per thread
         i += blockDim.x * gridDim.x) {
        size_t mod_idx = i / (n / 2) + start_mod_idx;
        size_t tid = i % (n / 2);

        // modulus
        const DModulus *modulus_table = modulus;
        uint64_t mod = modulus_table[mod_idx].value();
        uint64_t mod2 = mod << 1;

        size_t pairsInGroup;
        size_t k, j, glbIdx, bufIdx; // k = psi_step
        DMulUIntMod psi;
        uint64_t samples[2];

        for (size_t numOfGroups = 1; numOfGroups < n; numOfGroups <<= 1) {
            pairsInGroup = n / numOfGroups / 2;

            k = tid / pairsInGroup;
            j = tid % pairsInGroup;
            glbIdx = 2 * k * pairsInGroup + j;
            bufIdx = glbIdx % n;
            glbIdx += mod_idx * n;

            psi = twiddles[numOfGroups + k + n * mod_idx];

            if (numOfGroups == 1) {
                samples[0] = inout[glbIdx];
                samples[1] = inout[glbIdx + pairsInGroup];
            } else {
                samples[0] = buffer[bufIdx];
                samples[1] = buffer[bufIdx + pairsInGroup];
            }
            ct_butterfly(samples[0], samples[1], psi, mod);

            if (numOfGroups == n >> 1) {
                if (samples[0] >= mod2)
                    samples[0] -= mod2;
                if (samples[0] >= mod)
                    samples[0] -= mod;
                if (samples[1] >= mod2)
                    samples[1] -= mod2;
                if (samples[1] >= mod)
                    samples[1] -= mod;
                inout[glbIdx] = samples[0];
                inout[glbIdx + pairsInGroup] = samples[1];
            } else {
                buffer[bufIdx] = samples[0];
                buffer[bufIdx + pairsInGroup] = samples[1];
                __syncthreads();
            }
        }
    }
}

void nwt_1d_radix2_forward_inplace(uint64_t *inout,
                                   const DNTTTable &ntt_tables,
                                   size_t coeff_modulus_size,
                                   size_t start_modulus_idx) {

    size_t poly_degree = ntt_tables.n();
    const size_t per_block_memory = poly_degree * sizeof(uint64_t);

    inplace_fnwt_radix2<<<coeff_modulus_size, poly_degree / 2, per_block_memory>>>(
            inout,
            ntt_tables.twiddle(),
            ntt_tables.modulus(),
            coeff_modulus_size,
            start_modulus_idx,
            poly_degree);
}

/** backward NTT transformation, with N (num of operands) up to 2048,
 * to ensure all operation completed in one block.
 * @param[inout] inout The value to operate and the returned result
 * @param[in] inverse_twiddles The pre-computated backward NTT table
 * @param[in] mod The coeff modulus value
 * @param[in] n The poly degreee
 * @param[in] logn The logarithm of n
 * @param[in] numOfGroups
 * @param[in] iter The current iteration in backward NTT transformation
 */
__global__ void inplace_inwt_radix2(uint64_t *inout,
                                    const DMulUIntMod *itwiddles,
                                    const DModulus *modulus,
                                    size_t coeff_mod_size,
                                    size_t start_mod_idx,
                                    size_t n)
{
    extern __shared__ uint64_t buffer[];

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n / 2 * coeff_mod_size;
         i += blockDim.x * gridDim.x)
    {
        size_t mod_idx = i / (n / 2) + start_mod_idx;
        size_t tid = i % (n / 2);

        size_t pairsInGroup;
        size_t k, j, glbIdx, bufIdx;
        DMulUIntMod psi;
        uint64_t samples[2];

        const DModulus *modulus_table = modulus;
        uint64_t mod = modulus_table[mod_idx].value();

        for (size_t _numOfGroups = n / 2; _numOfGroups >= 1; _numOfGroups >>= 1)
        {
            pairsInGroup = n / _numOfGroups / 2;
            k = tid / pairsInGroup;
            j = tid % pairsInGroup;
            glbIdx = 2 * k * pairsInGroup + j;
            bufIdx = glbIdx % n;
            glbIdx += mod_idx * n;
            psi = itwiddles[_numOfGroups + k + mod_idx * n];
            if (_numOfGroups == n >> 1)
            {
                samples[0] = inout[glbIdx];
                samples[1] = inout[glbIdx + pairsInGroup];
            }
            else
            {
                samples[0] = buffer[bufIdx];
                samples[1] = buffer[bufIdx + pairsInGroup];
            }

            gs_butterfly(samples[0], samples[1], psi, mod);

            if (_numOfGroups == 1)
            {
                // final reduction
                if (samples[0] >= mod)
                    samples[0] -= mod;
                if (samples[1] >= mod)
                    samples[1] -= mod;
            }

            if (_numOfGroups == 1)
            {
                inout[glbIdx] = samples[0];
                inout[glbIdx + pairsInGroup] = samples[1];
            }
            else
            {
                buffer[bufIdx] = samples[0];
                buffer[bufIdx + pairsInGroup] = samples[1];
                __syncthreads();
            }
        }
    }
}

void nwt_1d_radix2_backward_inplace(uint64_t *inout,
                                    const DNTTTable &ntt_tables,
                                    size_t coeff_modulus_size,
                                    size_t start_modulus_idx)
{

    size_t poly_degree = ntt_tables.n();
    const size_t per_block_memory = poly_degree * sizeof(uint64_t);

    inplace_inwt_radix2<<<coeff_modulus_size, poly_degree / 2, per_block_memory>>>(
            inout,
            ntt_tables.itwiddle(),
            ntt_tables.modulus(),
            coeff_modulus_size,
            start_modulus_idx,
            poly_degree);
}
