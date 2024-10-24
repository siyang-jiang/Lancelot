#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "uintmath.cuh"
#include "gputype.h"
#include "common.h"

/***************************
 *
 * integer arithmetic
 *   |
 *   +--integer modular arithmetic
 *
 ***************************/

// negate
__forceinline__ __device__ uint64_t negate_uint64_mod(const uint64_t &operand,
                                                      const uint64_t &modulus)
{
    std::int64_t non_zero = (operand != 0);
    return (modulus - operand) & static_cast<std::uint64_t>(-non_zero);
}

/** 64-bit modulus addition
 */
__forceinline__ __device__ uint64_t add_uint64_uint64_mod(const uint64_t &operand1,
                                                          const uint64_t &operand2,
                                                          const uint64_t &modulus)
{
    uint64_t result_;
    asm volatile("add.u64  %0, %1, %2;"
                 : "=l"(result_)
                 : "l"(operand1), "l"(operand2));
    if (result_ >= modulus)
        result_ -= modulus;
    return result_;
}

/** 64-bit modulus subtraction
 */
__forceinline__ __device__ uint64_t sub_uint64_uint64_mod(const uint64_t &operand1,
                                                          const uint64_t &operand2,
                                                          const uint64_t &modulus)
{
    uint64_t result_ = operand1;
    if (operand1 < operand2)
        result_ += modulus;
    asm volatile("sub.u64  %0, %0, %1;"
                 : "+l"(result_)
                 : "l"(operand2));
    if (result_ >= modulus)
        result_ -= modulus;
    return result_;
}

/** unsigned long integer modular addition, all are long integers
 * @param[in] operand1 The operand 1
 * @param[in] operand2 The operand 2
 * @param[in] modulus The modulus
 * @param[in] uint64_count size of the input
 * @param[out] result operand1 + operand2 % modulus
 */
__forceinline__ __device__ void add_uint_uint_mod(const uint64_t *operand1,
                                                  const uint64_t *operand2,
                                                  const uint64_t *modulus,
                                                  uint32_t uint64_count, uint64_t *result)
{
    volatile uint8_t carry = add_uint_uint(operand1, operand2, uint64_count, result);
    // if (blockIdx.x * blockDim.x + threadIdx.x == 0)
    //     printf("carry = %d\n", carry);
    if (carry != 0 || is_greater_than_or_equal_uint(result, modulus, uint64_count))
    {
        sub_uint_uint(result, modulus, uint64_count, result);
    }
}

/** 128 bit multiplication, the returned value is 64bits
 * @param[in] operand1 The operand 1.
 * @param[in] operand2 The operand 2.
 * Return operand1 * operand2 >> 2^192
 */
__forceinline__ __device__ uint64_t barrett_multiply_and_shift_uint128(const uint128_t &operand1,
                                                                       const uint128_t &operand2)
{
    uint64_t p0 = __umul64hi(operand1.lo, operand2.lo);
    // !!!notice: volatile is necessary to avoid the incorrect compiler optimization!!!
    volatile uint128_t p1 = multiply_uint64_uint64(operand1.lo, operand2.hi);
    volatile uint128_t p2 = multiply_uint64_uint64(operand1.hi, operand2.lo);
    uint64_t p3 = operand1.hi * operand2.hi;
    asm volatile("add.cc.u64 %0, %0, %1;"
                 : "+l"(p1.lo)
                 : "l"(p0));
    asm volatile("addc.cc.u64 %0, %0, %1;"
                 : "+l"(p2.hi)
                 : "l"(p1.hi));
    asm volatile("add.cc.u64 %0, %0, %1;"
                 : "+l"(p2.lo)
                 : "l"(p1.lo));
    asm volatile("addc.cc.u64 %0, %0, %1;"
                 : "+l"(p3)
                 : "l"(p2.hi));
    return p3;
}

/** Reduce an 128-bit product into 64-bit modulus field using Barrett reduction
 * @param[in] product The input 128-bit product.
 * @param[in] modulus The modulus.
 * @param[in] barrett_mu The pre-computed value for mod, (2^128/modulus) in 128 bits.
 * Return prod % mod
 */
__forceinline__ __device__ uint64_t barrett_reduce_uint128_uint64(const uint128_t &product,
                                                                  const uint64_t &modulus,
                                                                  const uint64_t *barrett_mu)
{
    uint64_t result;

    uint64_t lo = product.lo;
    uint64_t hi = product.hi;

    uint64_t ratio0 = barrett_mu[0];
    uint64_t ratio1 = barrett_mu[1];

    uint64_t q = modulus;

    asm(
        "{\n\t"
        " .reg .u64 tmp;\n\t"
//        " .reg .pred p;\n\t"
        // Multiply input and const_ratio
        // Round 1
        " mul.hi.u64 tmp, %1, %3;\n\t"
        " mad.lo.cc.u64 tmp, %1, %4, tmp;\n\t"
        " madc.hi.u64 %0, %1, %4, 0;\n\t"
        // Round 2
        " mad.lo.cc.u64 tmp, %2, %3, tmp;\n\t"
        " madc.hi.u64 %0, %2, %3, %0;\n\t"
        // This is all we care about
        " mad.lo.u64 %0, %2, %4, %0;\n\t"
        // Barrett subtraction
        " mul.lo.u64 %0, %0, %5;\n\t"
        " sub.u64 %0, %1, %0;\n\t"
//        // cusbq
//        " sub.cc.s64 %0, %0, %5;\n\t"
//        " shr.s64 tmp, %0, 63;\n\t"
//        " and.b64 tmp, tmp, %5;\n\t"
//        " add.s64 %0, %0, tmp;\n\t"

//        " setp.hs.u64 p, %0, %5;\n\t"
//        " @p sub.u64 %0, %0, %5;\n\t"
        "}"
        : "=l"(result)
        : "l"(lo), "l"(hi), "l"(ratio0), "l"(ratio1), "l"(q));
    if (result >= q) result -= q;
    return result;
}

/** Barrett reduction for arbitrary 64-bit unsigned integer
 * @param[in] operand The operand.
 * @param[in] modulus The modulus.
 * @param[in] barrett_mu_hi 2^64/mod.
 * Return operand % modulus
 */
__forceinline__ __device__ uint64_t barrett_reduce_uint64_uint64(const uint64_t &operand,
                                                                 const uint64_t &modulus,
                                                                 const uint64_t &barrett_mu_hi)
{
    uint64_t s = __umul64hi(barrett_mu_hi, operand);
    uint64_t result_ = operand - s * modulus;
    if (result_ >= modulus)
        result_ -= modulus;
    return result_;
}

/** uint64 modular multiplication, result = operand1 * operand2 % mod
 * @param[in] operand1 The first operand (64 bits).
 * @param[in] operand2 The second operand (64 bits).
 * @param[in] modulus The modulus value (64 bits).
 * @param[in] barrett_mu 2^128/mod, (128 bits).
 *  res (64 bits).
 */
__forceinline__ __device__ uint64_t multiply_and_barrett_reduce_uint64(const uint64_t &operand1,
                                                                       const uint64_t &operand2,
                                                                       const uint64_t &modulus,
                                                                       const uint64_t *barrett_mu)
{
    uint64_t result;

    uint64_t q = modulus;

    uint64_t ratio0 = barrett_mu[0];
    uint64_t ratio1 = barrett_mu[1];
    asm(
        "{\n\t"
        " .reg .u64 tmp;\n\t"
        " .reg .u64 lo, hi;\n\t"
        // 128-bit multiply
        " mul.lo.u64 lo, %1, %2;\n\t"
        " mul.hi.u64 hi, %1, %2;\n\t"
        // Multiply input and const_ratio
        // Round 1
        " mul.hi.u64 tmp, lo, %3;\n\t"
        " mad.lo.cc.u64 tmp, lo, %4, tmp;\n\t"
        " madc.hi.u64 %0, lo, %4, 0;\n\t"
        // Round 2
        " mad.lo.cc.u64 tmp, hi, %3, tmp;\n\t"
        " madc.hi.u64 %0, hi, %3, %0;\n\t"
        // This is all we care about
        " mad.lo.u64 %0, hi, %4, %0;\n\t"
        // Barrett subtraction
        " mul.lo.u64 %0, %0, %5;\n\t"
        " sub.u64 %0, lo, %0;\n\t"
        "}"
        : "=l"(result)
        : "l"(operand1), "l"(operand2), "l"(ratio0), "l"(ratio1), "l"(q));
    if (result >= q) result -= q;
    return result;
}

/** Modular multiplication, result = operand1 * operand2 % mod
 * @param[in] operand1 The first operand (64 bits).
 * @param[in] operand2 Second operand (64-bit operand_ and 64-bit quotient_).
 * @param[in] mod The modulus value (64 bits).
 *  res (64 bits).
 */
[[nodiscard]] __inline__ __device__ uint64_t multiply_and_reduce_shoup(const uint64_t &operand1,
                                                         const DMulUIntMod &operand2,
                                                         const uint64_t &modulus)
{
    uint64_t hi = __umul64hi(operand1, operand2.quotient());
    uint64_t res = operand1 * operand2.operand() - hi * modulus;
    if (res >= modulus)
        res -= modulus;
    return res;
}

/** a = a * b % mod, Shoup's implementation
 */
[[nodiscard]] __inline__ __device__ uint64_t multiply_and_reduce_shoup_lazy(const uint64_t &operand1,
                                                              const DMulUIntMod &operand2,
                                                              const uint64_t &modulus)
{
    uint64_t hi = __umul64hi(operand1, operand2.quotient());
    return (uint64_t)operand1 * operand2.operand() - hi * modulus;
}

// calculate (op2 - op1) * op3 mod prime
__inline__ __device__ uint64_t sub_negate_const_mult(const uint64_t &op1,
                                                     const uint64_t &op2,
                                                     const DMulUIntMod &op3,
                                                     const uint64_t &prime) {
    uint64_t temp;
    if (op1 >= op2)
        temp = prime - op1 + op2;
    else {
        temp = op2 - op1;
    }
    return multiply_and_reduce_shoup(temp, op3, prime);
}
