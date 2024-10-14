#ifndef GPU_UINTMATH_H_
#define GPU_UINTMATH_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include "common.h"

typedef struct uint128_t {
    uint64_t hi;
    uint64_t lo;
    // TODO: implement uint128_t basic operations
//    __device__ uint128_t &operator+=(const uint128_t &op);
} uint128_t;

__forceinline__ __device__ void ld_two_uint64(uint64_t &x, uint64_t &y, const uint64_t *ptr) {
    asm("ld.global.v2.u64 {%0,%1}, [%2];" : "=l"(x), "=l"(y) : "l"(ptr));
}

__forceinline__ __device__ void st_two_uint64(uint64_t *ptr, const uint64_t &x, const uint64_t &y) {
    asm("st.cs.global.v2.u64 [%0], {%1, %2};" : :"l"(ptr), "l"(x), "l"(y));
}

/***************************
 *
 * others
 *
 ***************************/

__device__ inline uint64_t reverse_bits_uint32(uint32_t operand, uint32_t bit_count) {
    // Just return zero if bit_count is zero
    return (bit_count == 0) ? 0U
                            : uint32_t(__brev(operand)) >> (32U - bit_count);
}

/** Hamming weight
 * @param[in] value 8-bit input.
 * Return the hamming weight of operand.
 */
__forceinline__ __device__ int hamming_weight_uint8(const uint8_t &operand) {
    return __popc(static_cast<uint32_t>(operand));
}

/** compare two large unsigned integers
 * @param[in] operand1 The operand 1
 * @param[in] operand2 The operand 2
 * @param[in] uint64_count size of uint in uint64_t form
 * return 0 if operand1 = operand2, negative if operand1 < operand2, positive otherwise.
 */
__forceinline__ __device__ int compare_uint_uint(const uint64_t *operand1,
                                                 const uint64_t *operand2,
                                                 uint32_t uint64_count) {
    int result_ = 0;
    operand1 += uint64_count - 1;
    operand2 += uint64_count - 1;

    for (; (result_ == 0) && uint64_count--; operand1--, operand2--) {
        result_ = (*operand1 > *operand2) - (*operand1 < *operand2);
    }
    return result_;
}

/** compare if operand1 is greater than or equal to operand2
 * @param[in] operand1 The operand 1
 * @param[in] operand2 The operand 2
 * @param[in] uint64_count size of uint in uint64_t form
 * return true if operand1 >- operand2, false otherwise.
 */
__forceinline__ __device__ bool is_greater_than_or_equal_uint(const uint64_t *operand1, const uint64_t *operand2,
                                                              uint32_t uint64_count) {
    return compare_uint_uint(operand1, operand2, uint64_count) >= 0;
}

/** shift left a uint128 by shift_amount bits
 * @param[in] operand The operand.
 * @param[in] shift_amount the bits to be shifted.
 * @param[out] result = operand << shift_amount.
 */
__forceinline__ __device__ void shift_left_uint128(const uint128_t &operand,
                                                   const uint32_t &shift_amount,
                                                   uint128_t &result) {
    // Early return
    if (shift_amount >= cahel::util::bits_per_uint64_dev)
        result = {operand.lo, 0};
    else
        result = operand;

    // How many bits to shift in addition to word shift
    uint32_t bit_shift_amount = shift_amount & (cahel::util::bits_per_uint64_dev - 1);
    // Do we have a word shift
    if (bit_shift_amount > 0) {
        uint32_t neg_bit_shift_amount = cahel::util::bits_per_uint64_dev - bit_shift_amount;
        // Warning: if bit_shift_amount == 0 this is incorrect
        result.hi = (result.hi << bit_shift_amount) | (result.lo >> neg_bit_shift_amount);
        result.lo = result.lo << bit_shift_amount;
    }
}

/** shift right a uint128 by shift_amount bits
 * @param[in] operand The operand.
 * @param[in] shift_amount the bits to be shifted.
 * @param[out] result = operand >> shift_amount.
 */
__forceinline__ __device__ void shift_right_uint128(const uint128_t &operand,
                                                    const uint32_t &shift_amount,
                                                    uint128_t &result) {

    // Early return
    if (shift_amount >= cahel::util::bits_per_uint64_dev)
        result = {0, operand.hi};
    else
        result = operand;

    // How many bits to shift in addition to word shift
    uint32_t bit_shift_amount = shift_amount & (cahel::util::bits_per_uint64_dev - 1);
    // Do we have a word shift
    if (bit_shift_amount) {
        uint32_t neg_bit_shift_amount = cahel::util::bits_per_uint64_dev - bit_shift_amount;
        // Warning: if bit_shift_amount == 0 this is incorrect
        result.lo = (result.lo >> bit_shift_amount) | (result.hi << neg_bit_shift_amount);
        result.hi = result.hi >> bit_shift_amount;
    }
}

/***************************
 *
 * complex arithmetic
 *
 ***************************/

/** Multiplying a complex number by a real number.
 * @param[in] operand The operand
 * @param[in] scalar The scalar
 * return operand * scalar
 */
__forceinline__ __device__ cuDoubleComplex scalar_multiply_cuDoubleComplex(const cuDoubleComplex &operand,
                                                                           const double &scalar) {
    return make_cuDoubleComplex(operand.x * scalar, operand.y * scalar);
}

/***************************
 *
 * integer arithmetic
 *
 ***************************/

/** Add an uint128 integer to an uint64 integer, and assume there is no carry bit.
 * @param[in] operand1 The operand 1
 * @param[in] operand2 The operand 2
 * return operand1 + operand2
 */
__forceinline__ __device__ uint128_t add_uint128_uint64(const uint128_t &operand1, const uint64_t &operand2) {
    uint128_t result_;
    asm volatile("{\n\t"
                 "add.cc.u64   %0, %2, %4;\n\t"
                 "addc.u64  %1, %3,  0;\n\t"
                 "}"
            : "=l"(result_.lo), "=l"(result_.hi)
            : "l"(operand1.lo), "l"(operand1.hi), "l"(operand2));
    return result_;
}

/** unsigned 64-bit integer addition.
 * @param[in] operand1 The operand 1
 * @param[in] operand2 The operand 2
 * @param[out] result The result
 * return carry bit
 */
__forceinline__ __device__ uint8_t add_uint64_uint64(const uint64_t &operand1,
                                                     const uint64_t &operand2,
                                                     uint64_t &result) {
    uint32_t carry_;
    asm volatile("{\n\t"
                 "add.cc.u64  %0, %2, %3;\n\t"
                 "addc.u32    %1,  0,  0;\n\t"
                 "}"
            : "=l"(result), "=r"(carry_)
            : "l"(operand1), "l"(operand2));
    return static_cast<uint8_t>(carry_);
}

__forceinline__ __device__ uint8_t sub_uint64_uint64(const uint64_t &operand1,
                                                     const uint64_t &operand2,
                                                     uint64_t &result) {
    uint32_t borrow_;
    asm volatile("{\n\t"
                 "sub.cc.u64  %0, %2, %3;\n\t"
                 "subc.u32    %1,  0,  0;\n\t"
                 "}"
            : "=l"(result), "=r"(borrow_)
            : "l"(operand1), "l"(operand2));
    return static_cast<uint8_t>(borrow_) & 0x01;
}

/** unsigned 64-bit integer addition with array.
 * @param[in] operand1 The operand 1
 * @param[in] operand2 The operand 2
 * @param[in] carry The carry bit
 * @param[out] result The unsigned 64-bit sum
 * return the carry bit
 */
__forceinline__ __device__ uint8_t addc_uint64_uint64(const uint64_t &operand1,
                                                      const uint64_t &operand2,
                                                      const uint8_t &carry,
                                                      uint64_t &result) {
    uint32_t carry_ = 0;
    asm volatile("{\n\t"
                 "add.cc.u64  %0, %2, %3;\n\t"
                 "addc.u32    %1,  0,  0;\n\t"
                 "add.cc.u64  %0, %0, %4;\n\t"
                 "addc.u32    %1, %1,  0;\n\t"
                 "}"
            : "+l"(result), "+r"(carry_)
            : "l"(operand1), "l"(operand2), "l"(static_cast<uint64_t>(carry)));

    // if (blockIdx.x * blockDim.x + threadIdx.x == 3814)
    //     printf("**** %lu + %lu + %d = %lu, %d\n\n", operand1, operand2, carry, result, carry_);

    return static_cast<uint8_t>(carry_);
}

/** unsigned 128-bit integer addition.
 * @param[in] operand1 The operand 1
 * @param[in] operand2 The operand 2
 * @param[out] result The result
 * return carry bit
 */
__forceinline__ __device__ void add_uint128_uint128(const uint128_t &operand1,
                                                    const uint128_t &operand2,
                                                    uint128_t &result) {
    asm volatile("{\n\t"
                 "add.cc.u64     %0, %2, %4;\n\t"
                 "addc.u64    %1, %3, %5;\n\t"
                 "}"
            : "=l"(result.lo), "=l"(result.hi)
            : "l"(operand1.lo), "l"(operand1.hi), "l"(operand2.lo), "l"(operand2.hi));
}

/** unsigned 128-bit integer addition.
 * @param[in] operand1 The operand 1
 * @param[in] operand2 The operand 2
 * @param[out] result The result
 * return carry bit
 */
__forceinline__ __device__ uint8_t sub_uint128_uint128(const uint128_t &operand1,
                                                       const uint128_t &operand2,
                                                       uint128_t &result) {
    uint32_t borrow_;
    asm volatile("{\n\t"
                 "sub.cc.u64     %0, %3, %5;\n\t"
                 "subc.cc.u64    %1, %4, %6;\n\t"
                 "subc.u32       %2, 0, 0;\n\t"
                 "}"
            : "=l"(result.lo), "=l"(result.hi), "=r"(borrow_)
            : "l"(operand1.lo), "l"(operand1.hi), "l"(operand2.lo), "l"(operand2.hi));
    return static_cast<uint8_t>(borrow_) & 0x01; //__ffs(borrow);
}

/** addition of a multi-precision operand1 (has uint64_count uint64_t) and a uint64_t operand2, the result stored in res
 * @param[in] operand1 the multi-precision operand
 * @param[in] uint64_count the number of uint64_t for operand1
 * @param[in] operand2 the uint64_t
 * @param[out] result the multi-precision result
 */
__forceinline__ __device__ void add_uint_uint64(const uint64_t *operand1,
                                                uint32_t uint64_count,
                                                const uint64_t &operand2,
                                                uint64_t *result) {
    unsigned char carry = add_uint64_uint64(*operand1++, operand2, *result);
    result++;
    // Do the rest
    for (; --uint64_count; operand1++, result++) {
        uint64_t temp_result;
        carry = addc_uint64_uint64(*operand1, std::uint64_t(0), carry, temp_result);
        *result = temp_result;
    }
    return;
}

/** unsigned long integer addition.
 * @param[in] operand1 The operand 1
 * @param[in] operand2 The operand 2
 * @param[in] uint64_count size of the input
 * @param[out] result operand1 + operand2
 * return the carry bit
 */
__forceinline__ __device__ uint8_t add_uint_uint(const uint64_t *operand1,
                                                 const uint64_t *operand2,
                                                 uint32_t uint64_count,
                                                 uint64_t *result) {
    uint32_t carry_;

    asm volatile("add.cc.u64  %0, %1, %2;"
            : "=l"(*result++)
            : "l"(*operand1++), "l"(*operand2++));

    for (; --uint64_count; operand1++, operand2++, result++)
        asm volatile("addc.cc.u64  %0, %1, %2;"
                : "=l"(*result)
                : "l"(*operand1), "l"(*operand2));
    // asm("add.cc.u64  %0, %1, %2;"
    //     : "=l"(result[0])
    //     : "l"(operand1[0]), "l"(operand2[0]));
    // for (uint32_t i = 1; i < uint64_count; i++)
    //     asm("addc.cc.u64  %0, %1, %2;"
    //         : "=l"(result[i])
    //         : "l"(operand1[i]), "l"(operand2[i]));

    asm volatile("addc.u32  %0, 0, 0;"
            : "=r"(carry_));
    return static_cast<uint8_t>(carry_);
}

/** unsigned long integer subtraction.
 * @param[in] operand1 The operand 1
 * @param[in] operand2 The operand 2
 * @param[in] uint64_count size of the input
 * @param[out] result operand1 - operand2
 * return the borrow bit
 */
__forceinline__ __device__ uint8_t sub_uint_uint(const uint64_t *operand1,
                                                 const uint64_t *operand2,
                                                 uint32_t uint64_count,
                                                 uint64_t *result) {
    uint32_t borrow_;

    asm volatile("sub.cc.u64  %0, %1, %2;"
            : "=l"(*result++)
            : "l"(*operand1++), "l"(*operand2++));

    for (; --uint64_count; operand1++, operand2++, result++)
        asm volatile("subc.cc.u64  %0, %1, %2;"
                : "=l"(*result)
                : "l"(*operand1), "l"(*operand2));

    asm volatile("subc.u32  %0, 0, 0;"
            : "=r"(borrow_));
    return static_cast<uint8_t>(borrow_) & 0x01;
}

/**  a * b, return product is 128 bits.
 * schoolbook multiplication, copy from
 * https://forums.developer.nvidia.com/t/long-integer-multiplication-mul-wide-u64-and-mul-wide-u128/51520
 * Karatsuba multiplication is not suitable for PTX ISA
 * @param[in] operand1 The multiplier
 * @param[in] operand2 The multiplicand
 * return operand1 * operand2 in 128bits
 */
__forceinline__ __device__ uint128_t multiply_uint64_uint64(const uint64_t &operand1,
                                                            const uint64_t &operand2) {
    uint128_t result_;
    result_.lo = operand1 * operand2;
    result_.hi = __umul64hi(operand1, operand2);
    return result_;
}

/** multiply a long integer with an unsigned 64-bit integer.
 * @param[in] operand1 The operand 1
 * @param[in] uint64_count size of operand1
 * @param[in] operand2 The operand 2
 * @param[out] result operand1 * operand2
 */
__forceinline__ __device__ void multiply_uint_uint64(const uint64_t *operand1,
                                                     uint32_t uint64_count,
                                                     const uint64_t &operand2,
                                                     uint64_t *result) {
    uint128_t prod;
    uint64_t temp = 0;
    uint8_t carry = 0;

    prod = multiply_uint64_uint64(*operand1++, operand2);
    *result = prod.lo;
    // if (blockIdx.x * blockDim.x + threadIdx.x == 3814)
    //     printf("**** %lu * %lu = %lu, %lu\n\n", *(operand1 - 1), operand2, prod.hi, *result);
    result++;
    temp = prod.hi;

    for (; --uint64_count; operand1++, result++) {
        prod = multiply_uint64_uint64(*operand1, operand2);
        // if (blockIdx.x * blockDim.x + threadIdx.x == 3814)
        //     printf(">>>> %lu * %lu = %lu, %lu || %lu, %d\n", *operand1, operand2, prod.hi, prod.lo, temp, carry);
        carry = addc_uint64_uint64(prod.lo, temp, carry, *result);
        // if (blockIdx.x * blockDim.x + threadIdx.x == 3814)
        //     printf(">>>>>>>>>>>> result = %lu\n\n", *result);
        temp = prod.hi;
    }
}

__forceinline__ __device__ void
divide_uint128_uint64_generic(uint128_t &numerator, const uint64_t &denominator, uint128_t &quotient) {
    // Clear quotient. Set it to zero.
    quotient = {0, 0};

    // Determine significant bits in numerator and denominator.
    uint32_t numerator_bits;
    if (numerator.hi == 0)
        numerator_bits = cahel::util::bits_per_uint64_dev - __clzll(numerator.lo);
    else
        numerator_bits = cahel::util::bits_per_uint64_dev * 2 - __clzll(numerator.hi);
    uint32_t denominator_bits = cahel::util::bits_per_uint64_dev - __clzll(denominator);

    // if (threadIdx.x == 3)
    //     printf("%d, %d\n", numerator_bits, denominator_bits);

    // If numerator has fewer bits than denominator, then done.
    if (numerator_bits < denominator_bits) {
        return;
    }

    // Create temporary space to store mutable copy of denominator.
    uint128_t shifted_denominator = {0, denominator};

    // Create temporary space to store difference calculation.
    uint128_t difference = {0, 0};

    // Shift denominator to bring MSB in alignment with MSB of numerator.
    uint32_t denominator_shift = numerator_bits - denominator_bits;
    // if (threadIdx.x == 2)
    //     printf("===%lu, %lu\n", shifted_denominator.hi, shifted_denominator.lo);
    shift_left_uint128(shifted_denominator, denominator_shift, shifted_denominator);
    denominator_bits += denominator_shift;
    // if (threadIdx.x == 2)
    //     printf("===%lu, %lu\n", shifted_denominator.hi, shifted_denominator.lo);

    // Perform bit-wise division algorithm.
    uint32_t remaining_shifts = denominator_shift;

    while (numerator_bits == denominator_bits) {
        // NOTE: MSBs of numerator and denominator are aligned.

        // Even though MSB of numerator and denominator are aligned,
        // still possible numerator < shifted_denominator.
        // borrow = sub_uint128_uint128(numerator, shifted_denominator, difference);
        // if (threadIdx.x == 3)
        //     printf("%lu, %lu, %lu, %lu, %lu, %lu, %u\n", numerator.hi, numerator.lo, shifted_denominator.hi, shifted_denominator.lo, difference.hi, difference.lo, borrow);
        if (sub_uint128_uint128(numerator, shifted_denominator, difference)) {
            // numerator < shifted_denominator and MSBs are aligned,
            // so current quotient bit is zero and next one is definitely one.
            if (remaining_shifts == 0) {
                // No shifts remain and numerator < denominator so done.
                break;
            }
            // Effectively shift numerator left by 1 by instead adding
            // numerator to difference (to prevent overflow in numerator).
            add_uint128_uint128(difference, numerator, difference);

            // Adjust quotient and remaining shifts as a result of shifting numerator.
            quotient.hi = (quotient.hi << 1) | (quotient.lo >> (cahel::util::bits_per_uint64_dev - 1));
            quotient.lo <<= 1;
            remaining_shifts--;
        }
        // if (threadIdx.x == 3)
        //     printf("===%lu, %lu\n", quotient.hi, quotient.lo);
        //  Difference is the new numerator with denominator subtracted.

        // Determine amount to shift numerator to bring MSB in alignment
        // with denominator.
        if (difference.hi == 0)
            numerator_bits = cahel::util::bits_per_uint64_dev - __clzll(difference.lo);
        else
            numerator_bits = cahel::util::bits_per_uint64_dev * 2 - __clzll(difference.hi);

        // Clip the maximum shift to determine only the integer
        // (as opposed to fractional) bits.
        uint32_t numerator_shift = min(denominator_bits - numerator_bits, remaining_shifts);

        // Shift and update numerator.
        // This may be faster; first set to zero and then update if needed

        // Difference is zero so no need to shift, just set to zero.
        numerator = {0, 0};
        if (numerator_bits > 0) {
            shift_left_uint128(difference, numerator_shift, numerator);
            numerator_bits += numerator_shift;
        }

        // Update quotient to reflect subtraction.
        quotient.lo |= 1;

        // Adjust quotient and remaining shifts as a result of shifting numerator.
        shift_left_uint128(quotient, numerator_shift, quotient);
        remaining_shifts -= numerator_shift;
    }

    // Correct numerator (which is also the remainder) for shifting of
    // denominator, unless it is just zero.
    if (numerator_bits > 0) {
        shift_right_uint128(numerator, denominator_shift, numerator);
    }
}

/** addition with carry
 * @param[in] operand1
 * @param[in] operand2
 * @param[in] carry
 * @param[out] res
 */
__forceinline__ __device__ __host__ unsigned char
add_uint64_generic(uint64_t operand1, uint64_t operand2, unsigned char carry, uint64_t *result) {
    operand1 += operand2;
    *result = operand1 + carry;
    return (operand1 < operand2) || (~operand1 < carry);
}

__forceinline__ __host__ unsigned char add_uint_host(const std::uint64_t *operand1, std::size_t uint64_count,
                                                     std::uint64_t operand2, std::uint64_t *result) {
    if (!uint64_count) {
        throw std::invalid_argument("uint64_count");
    }
    if (!operand1) {
        throw std::invalid_argument("operand1");
    }
    if (!result) {
        throw std::invalid_argument("result");
    }
    // Unroll first iteration of loop. We assume uint64_count > 0.
    unsigned char carry = add_uint64_generic(*operand1++, operand2, 0, result++);

    // Do the rest
    for (; --uint64_count; operand1++, result++) {
        uint64_t temp_result;
        carry = add_uint64_generic(*operand1, std::uint64_t(0), carry, &temp_result);
        *result = temp_result;
    }
    return carry;
}

__forceinline__ __host__ void
divide_uint128_inplace_host(std::uint64_t *numerator, std::uint64_t denominator, std::uint64_t *result /*quotient*/) {
    unsigned __int128 n, q;
    n = (static_cast<unsigned __int128>(numerator[1]) << 64) | (static_cast<unsigned __int128>(numerator[0]));
    q = n / denominator;
    n -= q * denominator;
    numerator[0] = static_cast<std::uint64_t>(n);
    numerator[1] = 0;
    result[0] = static_cast<std::uint64_t>(q);
    result[1] = static_cast<std::uint64_t>(q >> 64);
}

#endif