// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// Check that double is 64 bits
static_assert(sizeof(double) == 8, "Require sizeof(double) == 8");

// Check that int is 32 bits
static_assert(sizeof(int) == 4, "Require sizeof(int) == 4");

// Check that unsigned long long is 64 bits
static_assert(sizeof(unsigned long long) == 8, "Require sizeof(unsigned long long) == 8");

// Bounds for bit-length of all coefficient moduli
#define CAHEL_MOD_BIT_COUNT_MAX 61
#define CAHEL_MOD_BIT_COUNT_MIN 2

// Bit-length of internally used coefficient moduli, e.g., auxiliary base in BFV
#define CAHEL_INTERNAL_MOD_BIT_COUNT 61

// Bounds for bit-length of user-defined coefficient moduli
#define CAHEL_USER_MOD_BIT_COUNT_MAX 60
#define CAHEL_USER_MOD_BIT_COUNT_MIN 2

// Bounds for bit-length of the plaintext modulus
#define CAHEL_PLAIN_MOD_BIT_COUNT_MAX CAHEL_USER_MOD_BIT_COUNT_MAX
#define CAHEL_PLAIN_MOD_BIT_COUNT_MIN CAHEL_USER_MOD_BIT_COUNT_MIN

// Bounds for number of coefficient moduli (no hard requirement)
#define CAHEL_COEFF_MOD_COUNT_MAX 64
#define CAHEL_COEFF_MOD_COUNT_MIN 1

// Bounds for polynomial modulus degree (no hard requirement)
#define CAHEL_POLY_MOD_DEGREE_MAX 131072
#define CAHEL_POLY_MOD_DEGREE_MIN 2

// Upper bound on the size of a ciphertext (no hard requirement)
#define CAHEL_CIPHERTEXT_SIZE_MAX 16
#define CAHEL_CIPHERTEXT_SIZE_MIN 2

// How many pairs of modular integers can we multiply and accumulate in a 128-bit data type
#if CAHEL_MOD_BIT_COUNT_MAX > 32
#define CAHEL_MULTIPLY_ACCUMULATE_MOD_MAX (1 << (128 - (CAHEL_MOD_BIT_COUNT_MAX << 1)))
#define CAHEL_MULTIPLY_ACCUMULATE_INTERNAL_MOD_MAX (1 << (128 - (JEDDAL_INTERNAL_MOD_BIT_COUNT_MAX << 1)))
#define CAHEL_MULTIPLY_ACCUMULATE_USER_MOD_MAX (1 << (128 - (CAHEL_USER_MOD_BIT_COUNT_MAX << 1)))
#else
#define CAHEL_MULTIPLY_ACCUMULATE_MOD_MAX SIZE_MAX
#define CAHEL_MULTIPLY_ACCUMULATE_INTERNAL_MOD_MAX SIZE_MAX
#define CAHEL_MULTIPLY_ACCUMULATE_USER_MOD_MAX SIZE_MAX
#endif

// Detect system
#define CAHEL_SYSTEM_OTHER 1
#define CAHEL_SYSTEM_WINDOWS 2
#define CAHEL_SYSTEM_UNIX_LIKE 3

#if defined(_WIN32)
#define CAHEL_SYSTEM SEAL_SYSTEM_WINDOWS
#elif defined(__linux__) || defined(__FreeBSD__) || defined(EMSCRIPTEN) || (defined(__APPLE__) && defined(__MACH__))
#define CAHEL_SYSTEM CAHEL_SYSTEM_UNIX_LIKE
#else
#define CAHEL_SYSTEM CAHEL_SYSTEM_OTHER
#error "Unsupported system"
#endif

// Detect compiler
#define CAHEL_COMPILER_MSVC 1
#define CAHEL_COMPILER_CLANG 2
#define CAHEL_COMPILER_GCC 3

#if defined(_MSC_VER)
#define CAHEL_COMPILER CAHEL_COMPILER_MSVC
#elif defined(__clang__)
#define CAHEL_COMPILER CAHEL_COMPILER_CLANG
#elif defined(__GNUC__) && !defined(__clang__)
#define CAHEL_COMPILER CAHEL_COMPILER_GCC
#else
#error "Unsupported compiler"
#endif

// CUDA support
#include <cuda_runtime_api.h>
#include <cuda.h>

// Allocate "size" bytes in memory and returns a seal_byte pointer
// If SEAL_USE_ALIGNED_ALLOC is defined, use _aligned_malloc and ::aligned_alloc (or std::malloc)
// Use `new seal_byte[size]` as fallback
#ifndef CAHEL_MALLOC
#define CAHEL_MALLOC(ptr, size)                                             \
    do                                                                       \
    {                                                                        \
        cudaMalloc((void **)&(ptr), size);                                   \
        cudaError_t status = cudaGetLastError();                             \
        if (status != cudaSuccess)                                           \
        {                                                                    \
            printf("CUDA error occurred: %s\n", cudaGetErrorString(status)); \
            throw std::logic_error("GPU malloc error!");                     \
        }                                                                    \
    } while (false)
#endif

// Deallocate a pointer in memory
// If SEAL_USE_ALIGNED_ALLOC is defined, use _aligned_free or std::free
// Use `delete [] ptr` as fallback
#ifndef CAHEL_FREE
#define CAHEL_FREE(ptr) (cudaFree(ptr))
#endif

#ifndef CAHEL_SUB_BORROW_UINT64
#define CAHEL_SUB_BORROW_UINT64(operand1, operand2, borrow, result) \
    sub_uint64_generic(operand1, operand2, borrow, result)
#endif

#ifndef CAHEL_DIVIDE_UINT128_UINT64
#define CAHEL_DIVIDE_UINT128_UINT64(numerator, denominator, result) \
    divide_uint128_uint64_inplace_generic(numerator, denominator, result);
#endif

#ifndef CAHEL_MULTIPLY_UINT64_HW64
#define CAHEL_MULTIPLY_UINT64_HW64(operand1, operand2, hw64) multiply_uint64_hw64_generic(operand1, operand2, hw64)
#endif

#ifndef CAHEL_MSB_INDEX_UINT64
#define CAHEL_MSB_INDEX_UINT64(result, value) get_msb_index_generic(result, value)
#endif

#define CAHEL_COND_SELECT(cond, if_true, if_false) (cond ? if_true : if_false)
