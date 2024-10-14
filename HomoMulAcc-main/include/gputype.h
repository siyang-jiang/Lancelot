#pragma once

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>
#include <stdint.h>
#include "uintmath.cuh"
#include "mempool.cuh"

/** Pre-computation for coeff modulus
 * value: the modulus value
 * const_ratio_: 2^128/value, in 128-bit
 */
typedef struct DModulus {
    uint64_t value_ = 0;
    uint64_t const_ratio_[2] = {0, 0}; // 0 corresponding low, 1 corresponding high

    DModulus() = default;

    DModulus(uint64_t value, uint64_t ratio0, uint64_t ratio1) : value_(value), const_ratio_{ratio0, ratio1} {
    }

    void set(const uint64_t value, const uint64_t const_ratio0, const uint64_t const_ratio1) {
        cudaMemcpy(&value_, &value, sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(&(const_ratio_[0]), &const_ratio0, sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(&(const_ratio_[1]), &const_ratio1, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    // Returns a const pointer to the value of the current Modulus.
    __device__ __host__ inline const uint64_t *data() const noexcept {
        return &value_;
    }

    __device__ __host__ inline uint64_t value() const {
        return value_;
    }

    __device__ __host__ inline auto &const_ratio() const {
        return const_ratio_;
    }
} DModulus;

/** Pre-computation for ntt and inverse ntt table values in shoup modulus reduction
 * operand: the value.
 * quotient: the inverse of the value mod the corresponding coeff modulus
 */
typedef struct DMulUIntMod {
    uint64_t operand_ = 0;
    uint64_t quotient_ = 0;

    DMulUIntMod() = default;

    DMulUIntMod(uint64_t operand, uint64_t quotient) : operand_(operand), quotient_(quotient) {
    }

    __device__ inline uint64_t operand() const {
        return operand_;
    }

    __device__ inline uint64_t quotient() const {
        return quotient_;
    }
} DMulUIntMod;

/** The GPU information for one RNS coeff
 * SID_: the cuda stream identifier
 * n_: poly degree
 * in_: input values
 * modulus_: the coeff modulus
 * twiddle: the forward NTT table
 * itwiddle: the inverse NTT table
 */

typedef struct DNTTTable {
    uint64_t n_;                    // vector length for this NWT
    uint64_t size_;                 // coeff_modulus_size
    cahel::util::Pointer<DModulus> modulus_;     // modulus for this NWT
    cahel::util::Pointer<DMulUIntMod> twiddle_;  // forward NTT table
    cahel::util::Pointer<DMulUIntMod> itwiddle_; // inverse NTT table

    DNTTTable() {
        n_ = 0;
        size_ = 0;
        modulus_ = cahel::util::Pointer<DModulus>();
        twiddle_ = cahel::util::Pointer<DMulUIntMod>();
        itwiddle_ = cahel::util::Pointer<DMulUIntMod>();
    }

    __host__ DNTTTable(DNTTTable &source) {
        n_ = source.n_;
        size_ = source.size_;
        modulus_.acquire(cahel::util::allocate<DModulus>(cahel::util::Global(), size_));
        twiddle_.acquire(cahel::util::allocate<DMulUIntMod>(cahel::util::Global(), n_ * size_));
        itwiddle_.acquire(cahel::util::allocate<DMulUIntMod>(cahel::util::Global(), n_ * size_));
        CUDA_CHECK(cudaMemcpy((void *) (modulus_.get()), source.modulus_.get(), size_ * sizeof(DModulus),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy((void *) (twiddle_.get()), source.twiddle_.get(), n_ * size_ * sizeof(DMulUIntMod),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy((void *) (itwiddle_.get()), source.itwiddle_.get(), n_ * size_ * sizeof(DMulUIntMod),
                              cudaMemcpyDeviceToDevice));
    }

    DNTTTable(DNTTTable &&source) noexcept {
        n_ = source.n_;
        size_ = source.size_;
        modulus_.acquire(source.modulus_);
        twiddle_.acquire(source.twiddle_);
        itwiddle_.acquire(source.itwiddle_);
    }

    __host__ DNTTTable &operator=(DNTTTable &source) {
        n_ = source.n_;
        size_ = source.size_;
        modulus_.acquire(cahel::util::allocate<DModulus>(cahel::util::Global(), size_));
        twiddle_.acquire(cahel::util::allocate<DMulUIntMod>(cahel::util::Global(), n_ * size_));
        itwiddle_.acquire(cahel::util::allocate<DMulUIntMod>(cahel::util::Global(), n_ * size_));
        CUDA_CHECK(cudaMemcpy((void *) (modulus_.get()), source.modulus_.get(), size_ * sizeof(DModulus),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy((void *) (twiddle_.get()), source.twiddle_.get(), n_ * size_ * sizeof(DMulUIntMod),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy((void *) (itwiddle_.get()), source.itwiddle_.get(), n_ * size_ * sizeof(DMulUIntMod),
                              cudaMemcpyDeviceToDevice));
        return *this;
    }

    __device__ __host__ DNTTTable &operator=(DNTTTable &&source) noexcept {
        n_ = source.n_;
        size_ = source.size_;
        modulus_.acquire(source.modulus_);
        twiddle_.acquire(source.twiddle_);
        itwiddle_.acquire(source.itwiddle_);
        return *this;
    }

    __device__ __host__ inline uint64_t n() const {
        return n_;
    }

    __device__ __host__ inline uint64_t size() const {
        return size_;
    }

    __device__ __host__ inline DModulus *modulus() const {
        return (DModulus *) (modulus_.get());
    }

    __device__ __host__ inline DMulUIntMod *twiddle() const {
        return (DMulUIntMod *) (twiddle_.get());
    }

    __device__ __host__ inline DMulUIntMod *itwiddle() const {
        return (DMulUIntMod *) (itwiddle_.get());
    }

    DNTTTable(uint64_t n, uint64_t size, DModulus *modulus, uint64_t *twiddle, uint64_t *itwiddle) : n_(n),
                                                                                                     size_(size) {
        modulus_.acquire(cahel::util::allocate<DModulus>(cahel::util::Global(), size));
        twiddle_.acquire(cahel::util::allocate<DMulUIntMod>(cahel::util::Global(), n * size));
        itwiddle_.acquire(cahel::util::allocate<DMulUIntMod>(cahel::util::Global(), n * size));
        CUDA_CHECK(cudaMemcpy((void *) (modulus_.get()), modulus, size * sizeof(DModulus), cudaMemcpyHostToDevice));
        CUDA_CHECK(
                cudaMemcpy((void *) (twiddle_.get()), twiddle, n * size * sizeof(DMulUIntMod), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void *) (itwiddle_.get()), itwiddle, n * size * sizeof(DMulUIntMod),
                              cudaMemcpyHostToDevice));
    }

    void init(uint64_t n, uint64_t size) {
        n_ = n;
        size_ = size;
        modulus_.acquire(cahel::util::allocate<DModulus>(cahel::util::Global(), size));
        twiddle_.acquire(cahel::util::allocate<DMulUIntMod>(cahel::util::Global(), n * size));
        itwiddle_.acquire(cahel::util::allocate<DMulUIntMod>(cahel::util::Global(), n * size));
    }

    void set(DModulus *modulus, uint64_t *twiddle, uint64_t *itwiddle, uint64_t index) const {
        CUDA_CHECK(cudaMemcpy((void *) (modulus_.get() + index), modulus, sizeof(DModulus), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void *) (twiddle_.get() + index * n_), twiddle, n_ * sizeof(DMulUIntMod),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void *) (itwiddle_.get() + index * n_), itwiddle, n_ * sizeof(DMulUIntMod),
                              cudaMemcpyHostToDevice));
    }

    ~DNTTTable() = default;
} DNTTTable;

//typedef struct DNTTTable {
//    uint64_t n_ = 0;                    // vector length for this NWT
//    uint64_t size_ = 0;                 // coeff_modulus_size
//
//    DModulus *modulus_ = nullptr;             // modulus for this NWT
//    DMulUIntMod *twiddle_ = nullptr;          // forward NTT table
//    DMulUIntMod *itwiddle_ = nullptr;         // inverse NTT table
//
//    DNTTTable() = default;
//    ~DNTTTable() = default;
//
//    DNTTTable(const DNTTTable &source) {
//        n_ = source.n_;
//        size_ = source.size_;
//        modulus_ = pool.allocate<DModulus>(size_);
//        twiddle_ = pool.allocate<DMulUIntMod>(n_ * size_);
//        itwiddle_ = pool.allocate<DMulUIntMod>(n_ * size_);
//
//        CUDA_CHECK(cudaMemcpy(modulus_, source.modulus_, size_ * sizeof(DModulus),
//                              cudaMemcpyDeviceToDevice));
//        CUDA_CHECK(cudaMemcpy(twiddle_, source.twiddle_, n_ * size_ * sizeof(DMulUIntMod),
//                              cudaMemcpyDeviceToDevice));
//        CUDA_CHECK(cudaMemcpy(itwiddle_, source.itwiddle_, n_ * size_ * sizeof(DMulUIntMod),
//                              cudaMemcpyDeviceToDevice));
//    }
//
//    DNTTTable &operator=(const DNTTTable &source) {
//        if (this == &source)
//            return *this;
//
//        n_ = source.n_;
//        size_ = source.size_;
//        modulus_ = pool.allocate<DModulus>(size_);
//        twiddle_ = pool.allocate<DMulUIntMod>(n_ * size_);
//        itwiddle_ = pool.allocate<DMulUIntMod>(n_ * size_);
//
//        CUDA_CHECK(cudaMemcpy(modulus_, source.modulus_, size_ * sizeof(DModulus),
//                              cudaMemcpyDeviceToDevice));
//        CUDA_CHECK(cudaMemcpy(twiddle_, source.twiddle_, n_ * size_ * sizeof(DMulUIntMod),
//                              cudaMemcpyDeviceToDevice));
//        CUDA_CHECK(cudaMemcpy(itwiddle_, source.itwiddle_, n_ * size_ * sizeof(DMulUIntMod),
//                              cudaMemcpyDeviceToDevice));
//        return *this;
//    }
//
//    DNTTTable(uint64_t n, uint64_t size, DModulus *modulus, uint64_t *twiddle, uint64_t *itwiddle) {
//        n_ = n;
//        size_ = size;
//        modulus_ = pool.allocate<DModulus>(size_);
//        twiddle_ = pool.allocate<DMulUIntMod>(n_ * size_);
//        itwiddle_ = pool.allocate<DMulUIntMod>(n_ * size_);
//
//        CUDA_CHECK(cudaMemcpy(modulus_, modulus, size_ * sizeof(DModulus),
//                              cudaMemcpyHostToDevice));
//        CUDA_CHECK(cudaMemcpy(twiddle_, twiddle, n_ * size_ * sizeof(DMulUIntMod),
//                              cudaMemcpyHostToDevice));
//        CUDA_CHECK(cudaMemcpy(itwiddle_, itwiddle, n_ * size_ * sizeof(DMulUIntMod),
//                              cudaMemcpyHostToDevice));
//    }
//
//    void init(uint64_t n, uint64_t size) {
//        n_ = n;
//        size_ = size;
//        modulus_ = pool.allocate<DModulus>(size_);
//        twiddle_ = pool.allocate<DMulUIntMod>(n_ * size_);
//        itwiddle_ = pool.allocate<DMulUIntMod>(n_ * size_);
//    }
//
//    void set(DModulus *modulus, uint64_t *twiddle, uint64_t *itwiddle, uint64_t index) const {
//        CUDA_CHECK(cudaMemcpy(modulus_ + index, modulus, sizeof(DModulus), cudaMemcpyHostToDevice));
//        CUDA_CHECK(cudaMemcpy(twiddle_ + index * n_, twiddle, n_ * sizeof(DMulUIntMod),
//                              cudaMemcpyHostToDevice));
//        CUDA_CHECK(cudaMemcpy(itwiddle_ + index * n_, itwiddle, n_ * sizeof(DMulUIntMod),
//                              cudaMemcpyHostToDevice));
//    }
//
//    [[nodiscard]] auto n() const {
//        return n_;
//    }
//
//    [[nodiscard]] auto size() const {
//        return size_;
//    }
//
//    [[nodiscard]] auto *modulus() const {
//        return modulus_;
//    }
//
//    [[nodiscard]] auto *twiddle() const {
//        return twiddle_;
//    }
//
//    [[nodiscard]] auto *itwiddle() const {
//        return itwiddle_;
//    }
//
//} DNTTTable;

typedef struct DCKKSEncoderInfo {
    cudaStream_t SID_;
    uint32_t m_; // order of the multiplicative group
    uint32_t sparse_slots_;
    cahel::util::Pointer<cuDoubleComplex> in_;      // input buffer, length must be n
    cahel::util::Pointer<cuDoubleComplex> twiddle_; // forward FFT table
    cahel::util::Pointer<uint32_t> mul_group_;

    DCKKSEncoderInfo() = default;

    DCKKSEncoderInfo &operator=(DCKKSEncoderInfo &&source) {
        SID_ = source.SID_;
        m_ = source.m_;
        sparse_slots_ = source.sparse_slots_;
        in_.acquire(source.in_);
        twiddle_.acquire(source.twiddle_);
        mul_group_.acquire(source.mul_group_);
        return *this;
    }

    DCKKSEncoderInfo(const size_t coeff_count) {
        m_ = coeff_count << 1;
        uint32_t slots = coeff_count >> 1; // n/2
        uint32_t slots_half = slots >> 1;
        /*
        CUDA_CHECK(cudaMallocManaged((void **)&(in_), slots * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMallocManaged((void **)&(twiddle_), m_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMallocManaged((void **)&(mul_group_), slots_half * sizeof(uint32_t)));
        */

        CUDA_CHECK(cudaStreamCreate(&SID_));
        in_.acquire(cahel::util::allocate<cuDoubleComplex>(cahel::util::Global(), slots));
        twiddle_.acquire(cahel::util::allocate<cuDoubleComplex>(cahel::util::Global(), m_));
        mul_group_.acquire(cahel::util::allocate<uint32_t>(cahel::util::Global(), slots_half));
    }

    ~DCKKSEncoderInfo() {
        // printf("debug ~~DeviceCKKSEncoderInfo\n");
        // TODO
        // CUDA_CHECK(cudaStreamDestroy(SID_));
    }

    __device__ __host__ inline cudaStream_t &SID() {
        return SID_;
    }

    __device__ __host__ inline uint32_t m() {
        return m_;
    }

    __device__ __host__ inline uint32_t sparse_slots() {
        return sparse_slots_;
    }

    __device__ __host__ inline cuDoubleComplex *in() {
        return (cuDoubleComplex *) (in_.get());
    }

    __device__ __host__ inline cuDoubleComplex *twiddle() {
        return (cuDoubleComplex *) (twiddle_.get());
    }

    __device__ __host__ inline uint32_t *mul_group() {
        return (uint32_t *) (mul_group_.get());
    }

    __device__ __host__ inline cuDoubleComplex *in() const {
        return (cuDoubleComplex *) (in_.get());
    }

    __device__ __host__ inline cuDoubleComplex *twiddle() const {
        return (cuDoubleComplex *) (twiddle_.get());
    }

    __device__ __host__ inline uint32_t *mul_group() const {
        return (uint32_t *) (mul_group_.get());
    }

    __device__ __host__ inline void set_sparse_slots(uint32_t sparse_slots) {
        sparse_slots_ = sparse_slots;
    }
} DCKKSEncoderInfo;

typedef uint64_t *uint64_ptr;
