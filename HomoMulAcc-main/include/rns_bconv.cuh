#pragma once

#include "rns.cuh"

struct uint128_t2 {
    uint128_t x;
    uint128_t y;
};

struct uint128_t4 {
    uint128_t x;
    uint128_t y;
    uint128_t z;
    uint128_t w;
};

struct double_t2 {
    double x;
    double y;
};

struct double_t4 {
    double x;
    double y;
    double z;
    double w;
};

typedef struct DBaseConverter {
    DRNSBase ibase_;
    DRNSBase obase_;
    cahel::util::Pointer<uint64_t> qiHat_mod_pj_;
    cahel::util::Pointer<uint64_t> alpha_Q_mod_pj_;
    cahel::util::Pointer<DMulUIntMod> negPQHatInvModq_;
    cahel::util::Pointer<uint64_t> QInvModp_;
    cahel::util::Pointer<DMulUIntMod> PModq_;

    DBaseConverter() {
        qiHat_mod_pj_ = cahel::util::Pointer<uint64_t>();
        alpha_Q_mod_pj_ = cahel::util::Pointer<uint64_t>();
        negPQHatInvModq_ = cahel::util::Pointer<DMulUIntMod>();
        QInvModp_ = cahel::util::Pointer<uint64_t>();
        PModq_ = cahel::util::Pointer<DMulUIntMod>();
    }

    DBaseConverter(DBaseConverter &source) {
        qiHat_mod_pj_.acquire(source.qiHat_mod_pj_);
        alpha_Q_mod_pj_.acquire(source.alpha_Q_mod_pj_);
        negPQHatInvModq_.acquire(source.negPQHatInvModq_);
        QInvModp_.acquire(source.QInvModp_);
        PModq_.acquire(source.PModq_);
    }

    DBaseConverter(DBaseConverter &&source) noexcept {
        qiHat_mod_pj_.acquire(source.qiHat_mod_pj_);
        alpha_Q_mod_pj_.acquire(source.alpha_Q_mod_pj_);
        negPQHatInvModq_.acquire(source.negPQHatInvModq_);
        QInvModp_.acquire(source.QInvModp_);
        PModq_.acquire(source.PModq_);
    }

    inline void init(cahel::util::BaseConverter &cpu_base_converter) {
        ibase_.init(cpu_base_converter.ibase());
        obase_.init(cpu_base_converter.obase());

        qiHat_mod_pj_.acquire(cahel::util::allocate<uint64_t>(cahel::util::Global(), obase_.size() * ibase_.size()));
        for (size_t idx = 0; idx < obase_.size(); idx++)
            CUDA_CHECK(cudaMemcpy(qiHat_mod_pj_.get() + idx * ibase_.size(),
                                  cpu_base_converter.QHatModp(idx),
                                  ibase_.size() * sizeof(std::uint64_t),
                                  cudaMemcpyHostToDevice));

        alpha_Q_mod_pj_.acquire(
                cahel::util::allocate<uint64_t>(cahel::util::Global(), (ibase_.size() + 1) * obase_.size()));
        for (size_t idx = 0; idx < ibase_.size() + 1; idx++)
            CUDA_CHECK(cudaMemcpy(alpha_Q_mod_pj_.get() + idx * obase_.size(),
                                  cpu_base_converter.alphaQModp(idx),
                                  obase_.size() * sizeof(std::uint64_t),
                                  cudaMemcpyHostToDevice));

        negPQHatInvModq_.acquire(cahel::util::allocate<DMulUIntMod>(cahel::util::Global(), ibase_.size()));
        CUDA_CHECK(cudaMemcpy(negPQHatInvModq_.get(), cpu_base_converter.negPQHatInvModq(),
                              ibase_.size() * sizeof(DMulUIntMod), cudaMemcpyHostToDevice));

        QInvModp_.acquire(cahel::util::allocate<uint64_t>(cahel::util::Global(), obase_.size() * ibase_.size()));
        for (size_t idx = 0; idx < obase_.size(); idx++)
            CUDA_CHECK(cudaMemcpy(QInvModp_.get() + idx * ibase_.size(), cpu_base_converter.QInvModp(idx),
                                  ibase_.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice));

        PModq_.acquire(cahel::util::allocate<DMulUIntMod>(cahel::util::Global(), ibase_.size()));
        CUDA_CHECK(cudaMemcpy(PModq_.get(), cpu_base_converter.PModq(),
                              ibase_.size() * sizeof(DMulUIntMod), cudaMemcpyHostToDevice));
    }

    void bConv_BEHZ(uint64_t *dst, const uint64_t *src, size_t n) const;

    void bConv_BEHZ_var1(uint64_t *dst, const uint64_t *src, size_t n) const;

    void bConv_HPS(uint64_t *dst, const uint64_t *src, size_t n) const;

    void exact_convert_array(uint64_t *dst, const uint64_t *src, const uint64_t poly_degree) const;

    ~DBaseConverter() = default;

    __host__ __device__ inline auto &ibase() const {
        return ibase_;
    }

    __host__ __device__ inline auto &obase() const {
        return obase_;
    }

    __host__ __device__ inline uint64_t *QHatModp() const {
        return qiHat_mod_pj_.get();
    }

    __host__ __device__ inline uint64_t *alpha_Q_mod_pj() const {
        return alpha_Q_mod_pj_.get();
    }

    __host__ __device__ inline DMulUIntMod *negPQHatInvModq() const {
        return negPQHatInvModq_.get();
    }

    __host__ __device__ inline uint64_t *QInvModp() const {
        return QInvModp_.get();
    }

    __host__ __device__ inline DMulUIntMod *PModq() const {
        return PModq_.get();
    }

} DBaseConverter;

__global__ void bconv_mult_kernel(uint64_t *dst,
                                  const uint64_t *src,
                                  const DMulUIntMod *scale,
                                  const DModulus *base, uint64_t base_size,
                                  uint64_t n);

__global__ void bconv_mult_unroll2_kernel(uint64_t *dst,
                                          const uint64_t *src,
                                          const DMulUIntMod *scale,
                                          const DModulus *base, uint64_t base_size,
                                          uint64_t n);

__global__ void bconv_mult_unroll4_kernel(uint64_t *dst,
                                          const uint64_t *src,
                                          const DMulUIntMod *scale,
                                          const DModulus *base, uint64_t base_size,
                                          uint64_t n);

__global__ void bconv_matmul_kernel(uint64_t *dst,
                                    const uint64_t *xi_qiHatInv_mod_qi,
                                    const uint64_t *QHatModp,
                                    const DModulus *ibase, uint64_t ibase_size,
                                    const DModulus *obase, uint64_t obase_size,
                                    uint64_t n);

__global__ void bconv_matmul_unroll2_kernel(uint64_t *dst,
                                            const uint64_t *xi_qiHatInv_mod_qi,
                                            const uint64_t *QHatModp,
                                            const DModulus *ibase, uint64_t ibase_size,
                                            const DModulus *obase, uint64_t obase_size,
                                            uint64_t n);

__global__ void bconv_matmul_unroll4_kernel(uint64_t *dst,
                                            const uint64_t *xi_qiHatInv_mod_qi,
                                            const uint64_t *QHatModp,
                                            const DModulus *ibase, uint64_t ibase_size,
                                            const DModulus *obase, uint64_t obase_size,
                                            uint64_t n);

__forceinline__ __device__ uint128_t base_convert_acc(const uint64_t *ptr,
                                                      const uint64_t *QHatModp, size_t out_prime_idx,
                                                      size_t degree,
                                                      size_t ibase_size,
                                                      size_t degree_idx) {
    uint128_t accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const uint64_t op2 = QHatModp[out_prime_idx * ibase_size + i];
        uint128_t out;

        uint64_t op1 = ptr[i * degree + degree_idx];
        out = multiply_uint64_uint64(op1, op2);
        add_uint128_uint128(out, accum, accum);
    }
    return accum;
}

__forceinline__ __device__ uint128_t2 base_convert_acc_unroll2(const uint64_t *ptr,
                                                               const uint64_t *QHatModp, size_t out_prime_idx,
                                                               size_t degree,
                                                               size_t ibase_size,
                                                               size_t degree_idx) {
    uint128_t2 accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const uint64_t op2 = QHatModp[out_prime_idx * ibase_size + i];
        uint128_t2 out{};

        uint64_t op1_x, op1_y;
        ld_two_uint64(op1_x, op1_y, ptr + i * degree + degree_idx);
        out.x = multiply_uint64_uint64(op1_x, op2);
        add_uint128_uint128(out.x, accum.x, accum.x);
        out.y = multiply_uint64_uint64(op1_y, op2);
        add_uint128_uint128(out.y, accum.y, accum.y);
    }
    return accum;
}

__forceinline__ __device__ uint128_t4 base_convert_acc_unroll4(const uint64_t *ptr,
                                                               const uint64_t *QHatModp, size_t out_prime_idx,
                                                               size_t degree,
                                                               size_t ibase_size,
                                                               size_t degree_idx) {
    uint128_t4 accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const uint64_t op2 = QHatModp[out_prime_idx * ibase_size + i];
        uint128_t4 out{};

        uint64_t op1_x, op1_y;
        ld_two_uint64(op1_x, op1_y, ptr + i * degree + degree_idx);
        out.x = multiply_uint64_uint64(op1_x, op2);
        add_uint128_uint128(out.x, accum.x, accum.x);
        out.y = multiply_uint64_uint64(op1_y, op2);
        add_uint128_uint128(out.y, accum.y, accum.y);

        uint64_t op1_z, op1_w;
        ld_two_uint64(op1_z, op1_w, ptr + i * degree + degree_idx + 2);
        out.z = multiply_uint64_uint64(op1_z, op2);
        add_uint128_uint128(out.z, accum.z, accum.z);
        out.w = multiply_uint64_uint64(op1_w, op2);
        add_uint128_uint128(out.w, accum.w, accum.w);
    }
    return accum;
}

__forceinline__ __device__ double_t base_convert_acc_frac(const uint64_t *ptr,
                                                          const double *qiInv,
                                                          size_t degree,
                                                          size_t ibase_size,
                                                          size_t degree_idx) {
    double_t accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const double op2 = qiInv[i];

        uint64_t op1 = ptr[i * degree + degree_idx];
        accum += static_cast<double>(op1) * op2;
    }
    return accum;
}

__forceinline__ __device__ double_t2 base_convert_acc_frac_unroll2(const uint64_t *ptr,
                                                                   const double *qiInv,
                                                                   size_t degree,
                                                                   size_t ibase_size,
                                                                   size_t degree_idx) {
    double_t2 accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const double op2 = qiInv[i];

        uint64_t op1_x, op1_y;
        ld_two_uint64(op1_x, op1_y, ptr + i * degree + degree_idx);
        accum.x += static_cast<double>(op1_x) * op2;
        accum.y += static_cast<double>(op1_y) * op2;
    }
    return accum;
}

__forceinline__ __device__ double_t4 base_convert_acc_frac_unroll4(const uint64_t *ptr,
                                                                   const double *qiInv,
                                                                   size_t degree,
                                                                   size_t ibase_size,
                                                                   size_t degree_idx) {
    double_t4 accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const double op2 = qiInv[i];

        uint64_t op1_x, op1_y;
        ld_two_uint64(op1_x, op1_y, ptr + i * degree + degree_idx);
        accum.x += static_cast<double>(op1_x) * op2;
        accum.y += static_cast<double>(op1_y) * op2;

        uint64_t op1_z, op1_w;
        ld_two_uint64(op1_z, op1_w, ptr + i * degree + degree_idx + 2);
        accum.z += static_cast<double>(op1_z) * op2;
        accum.w += static_cast<double>(op1_w) * op2;
    }
    return accum;
}

__global__ void add_to_ct_kernel(uint64_t *ct,
                                 const uint64_t *cx,
                                 const DModulus *modulus,
                                 size_t n,
                                 size_t size_Ql);
