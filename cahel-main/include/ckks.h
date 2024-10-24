#pragma once

#include "gpucontext.h"
#include "fft.h"
#include "ntt.cuh"
#include "plaintext.h"
#include "croots.h"
#include "rns.cuh"

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>

/**
    Provides functionality for encoding vectors of complex or real numbers into
    plaintext polynomials to be encrypted and computed on using the CKKS scheme.
    If the polynomial modulus degree is N, then CKKSEncoder converts vectors of
    N/2 complex numbers into plaintext elements. Homomorphic operations performed
    on such encrypted vectors are applied coefficient (slot-)wise, enabling
    powerful SIMD functionality for computations that are vectorizable. This
    functionality is often called "batching" in the homomorphic encryption
    literature.

    @par Mathematical Background
    Mathematically speaking, if the polynomial modulus is X^N+1, N is a power of
    two, the CKKSEncoder implements an approximation of the canonical embedding
    of the ring of integers Z[X]/(X^N+1) into C^(N/2), where C denotes the complex
    numbers. The Galois group of the extension is (Z/2NZ)* ~= Z/2Z x Z/(N/2)
    whose action on the primitive roots of unity modulo coeff_modulus is easy to
    describe. Since the batching slots correspond 1-to-1 to the primitive roots
    of unity, applying Galois automorphisms on the plaintext acts by permuting
    the slots. By applying generators of the two cyclic subgroups of the Galois
    group, we can effectively enable cyclic rotations and complex conjugations
    of the encrypted complex vectors.
    */

class CAHELGPUCKKSEncoder {
public:
    CAHELGPUCKKSEncoder() = default;

    /** Creates a CKKSEncoder instance initialized with the specified CAHELGPUContext.
     * @param[in] context The CAHELGPUContext
     * @throws std::invalid_argument if scheme is not scheme_type::CKKS
     */
    explicit CAHELGPUCKKSEncoder(const CAHELGPUContext &context);

    ~CAHELGPUCKKSEncoder() = default;

    /**
     * @brief Encodes a vector of complex numbers to specified chain index
     * @param context
     * @param values
     * @param chain_index
     * @param scale
     * @param destination
     */
    inline void encode(const CAHELGPUContext &context,
                       const std::vector<cuDoubleComplex> &values,
                       size_t chain_index,
                       double scale,
                       CAHELGPUPlaintext &destination) {
        encode_internal(context, values.data(), values.size(), chain_index, scale, destination);
    }

    /**
     * @brief Encodes a vector of complex numbers
     * @param context
     * @param values
     * @param scale
     * @param destination
     */
    inline void encode(const CAHELGPUContext &context,
                       const std::vector<cuDoubleComplex> &values,
                       double scale,
                       CAHELGPUPlaintext &destination) {
        encode_internal(context, values.data(), values.size(), first_chain_index_, scale, destination);
    }

    /**
     * @brief Encodes a vector of double numbers to specified chain index
     * @param context
     * @param values
     * @param chain_index
     * @param scale
     * @param destination
     */
    inline void encode(const CAHELGPUContext &context,
                       const std::vector<double> &values,
                       size_t chain_index,
                       double scale,
                       CAHELGPUPlaintext &destination) {
        encode_internal(context, values.data(), values.size(), chain_index, scale, destination);
    }

    /**
     * @brief Encodes a vector of double numbers
     * @param context
     * @param values
     * @param scale
     * @param destination
     */
    inline void encode(const CAHELGPUContext &context,
                       const std::vector<double> &values,
                       double scale,
                       CAHELGPUPlaintext &destination) {
        encode_internal(context, values.data(), values.size(), first_chain_index_, scale, destination);
    }

    /**
     * @brief Encodes a complex number
     * @param context
     * @param value
     * @param chain_index
     * @param scale
     * @param destination
     */
    inline void encode(const CAHELGPUContext &context,
                       cuDoubleComplex value,
                       size_t chain_index,
                       double scale,
                       CAHELGPUPlaintext &destination) {
        encode_internal(context, value, chain_index, scale, destination);
    }

    /**
     * @brief Encodes a double number
     * @param context
     * @param value
     * @param chain_index
     * @param scale
     * @param destination
     */
    inline void encode(const CAHELGPUContext &context,
                       double value,
                       size_t chain_index,
                       double scale,
                       CAHELGPUPlaintext &destination) {
        encode_internal(context, value, chain_index, scale, destination);
    }

    /**
     * @brief Encodes a int64_t number
     * @param context
     * @param value
     * @param chain_index
     * @param destination
     */
    inline void encode(const CAHELGPUContext &context,
                       std::int64_t value,
                       size_t chain_index,
                       CAHELGPUPlaintext &destination) {
        encode_internal(context, value, chain_index, destination);
    }

    /**
     * @brief Decodes to a vector of complex numbers
     * @param context
     * @param plain
     * @param destination
     */
    void decode(const CAHELGPUContext &context,
                const CAHELGPUPlaintext &plain,
                std::vector<cuDoubleComplex> &destination) {
        destination.resize(sparse_slots_);
        decode_internal(context, plain, destination.data());
    }

    /**
     * @brief Decodes to a vector of double numbers
     * @param context
     * @param plain
     * @param destination
     */
    void decode(const CAHELGPUContext &context,
                const CAHELGPUPlaintext &plain,
                std::vector<double> &destination) {
        destination.resize(sparse_slots_);
        decode_internal(context, plain, destination.data());
    }

    // for python wrapper

    auto encode(const CAHELGPUContext &context, const std::vector<double> &values, double scale) {
        CAHELGPUPlaintext destination = CAHELGPUPlaintext(context);
        encode_internal(context, values.data(), values.size(), first_chain_index_, scale, destination);
        return destination;
    }

    auto encode(const CAHELGPUContext &context, const std::vector<double> &values, size_t chain_index, double scale) {
        CAHELGPUPlaintext destination = CAHELGPUPlaintext(context);
        encode_internal(context, values.data(), values.size(), chain_index, scale, destination);
        return destination;
    }

    auto decode(const CAHELGPUContext &context, const CAHELGPUPlaintext &plain) {
        std::vector<double> destination(sparse_slots_);
        decode_internal(context, plain, destination.data());
        return destination;
    }

    [[nodiscard]] inline std::size_t slot_count() const noexcept {
        return slots_;
    }

    __host__ __device__ inline DCKKSEncoderInfo &gpu_ckks_msg_vec() {
        return gpu_ckks_msg_vec_;
    }

    inline void reset_sparse_slots() {
        sparse_slots_ = 0;
    }

private:
    /**
     * @brief Encodes a vector of complex numbers
     * @param context
     * @param values
     * @param values_size
     * @param chain_index
     * @param scale
     * @param destination
     */
    void encode_internal(const CAHELGPUContext &context,
                         const cuDoubleComplex *values, size_t values_size,
                         size_t chain_index, double scale,
                         CAHELGPUPlaintext &destination);

    /**
     * @brief Encodes a vector of double numbers
     * @param context
     * @param values
     * @param values_size
     * @param chain_index
     * @param scale
     * @param destination
     */
    inline void encode_internal(const CAHELGPUContext &context,
                                const double *values, size_t values_size,
                                size_t chain_index, double scale,
                                CAHELGPUPlaintext &destination) {
        std::vector<cuDoubleComplex> input;
        input.reserve(values_size);
        for (size_t i = 0; i < values_size; i++) {
            input.push_back(make_cuDoubleComplex(values[i], 0.0));
        }
        encode_internal(context, input.data(), values_size, chain_index, scale, destination);
        input.clear();
    }

    /**
     * @brief Encodes a complex number
     * @param context
     * @param value
     * @param chain_index
     * @param scale
     * @param destination
     */
    inline void encode_internal(const CAHELGPUContext &context,
                                cuDoubleComplex value,
                                size_t chain_index, double scale,
                                CAHELGPUPlaintext &destination) {
        std::vector<cuDoubleComplex> input(slots_, value);
        encode_internal(context, input.data(), slots_, chain_index, scale, destination);
        input.clear();
    }

    /**
     * @brief Encodes a double number
     * @param context
     * @param value
     * @param chain_index
     * @param scale
     * @param destination
     */
    void encode_internal(const CAHELGPUContext &context,
                         double value,
                         size_t chain_index, double scale,
                         CAHELGPUPlaintext &destination);

    /**
     * @brief Encodes a int64_t number (no scaling)
     * @param context
     * @param value
     * @param chain_index
     * @param destination
     */
    void encode_internal(const CAHELGPUContext &context,
                         int64_t value,
                         size_t chain_index,
                         CAHELGPUPlaintext &destination);

    /**
     * @brief Decodes to a vector of complex numbers
     * @param context
     * @param plain
     * @param destination
     */
    void decode_internal(const CAHELGPUContext &context,
                         const CAHELGPUPlaintext &plain,
                         cuDoubleComplex *destination);

    /**
     * @brief Decodes to a vector of double numbers
     * @param context
     * @param plain
     * @param destination
     */
    inline void decode_internal(const CAHELGPUContext &context,
                                const CAHELGPUPlaintext &plain,
                                double *destination) {
        std::vector<cuDoubleComplex> output;
        output.resize(sparse_slots_);
        decode_internal(context, plain, output.data());
        for (size_t i = 0; i < sparse_slots_; i++)
            destination[i] = output[i].x;
        output.clear();
    }

    std::uint32_t slots_{};
    std::uint32_t sparse_slots_ = 0;
    std::shared_ptr<cahel::util::ComplexRoots> complex_roots_;
    std::vector<cuDoubleComplex> root_powers_;
    std::vector<std::uint32_t> rotation_group_;
    DCKKSEncoderInfo gpu_ckks_msg_vec_;
    std::uint32_t first_chain_index_ = 1;
};
