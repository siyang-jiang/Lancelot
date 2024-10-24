#pragma once

#include "gpucontext.h"
#include "polymath.cuh"
#include "util/polycore.h"

typedef struct CAHELGPUCiphertext {
    cahel::parms_id_type parms_id_ = cahel::parms_id_zero;

    // The index this ciphertext corresponding
    std::size_t chain_index_ = 0;

    // The number of poly in ciphertext
    std::size_t size_ = 0;

    // The poly degree
    std::size_t poly_modulus_degree_ = 0;

    // The coeff prime number
    std::size_t coeff_modulus_size_ = 0;

    // The scale this ciphertext corresponding to
    double scale_ = 1.0;

    // The correction factor for BGV decryption
    std::uint64_t correction_factor_ = 1;

    // the degree of the scaling factor for the encrypted message
    size_t noiseScaleDeg_ = 1;

    bool is_ntt_form_ = true;

    bool is_asymmetric_ = false;

    // The data_
    cahel::util::Pointer<uint8_t> prng_seed_; // prng seed

    // multiple poly, include size_ * coeff_modulus_size_ * poly_modulus_degree_
    cahel::util::Pointer<uint64_t> data_;

    CAHELGPUCiphertext() {
        chain_index_ = 0;
        size_ = 0;
        poly_modulus_degree_ = 0;
        coeff_modulus_size_ = 0;
        scale_ = 0;
        correction_factor_ = 1;
        noiseScaleDeg_ = 1;
        is_ntt_form_ = true;
        is_asymmetric_ = false;
        prng_seed_.acquire(cahel::util::allocate<uint8_t>(cahel::util::Global(),
                                                          cahel::util::global_variables::prng_seed_byte_count));
    }

    explicit CAHELGPUCiphertext(const CAHELGPUContext &context) {
        prng_seed_.acquire(cahel::util::allocate<uint8_t>(cahel::util::Global(),
                                                          cahel::util::global_variables::prng_seed_byte_count));
    }

    // copy constructor
    CAHELGPUCiphertext(const CAHELGPUCiphertext &copy) {
        parms_id_ = copy.parms_id_;
        chain_index_ = copy.chain_index_;
        size_ = copy.size_;
        poly_modulus_degree_ = copy.poly_modulus_degree_;
        coeff_modulus_size_ = copy.coeff_modulus_size_;
        scale_ = copy.scale_;
        correction_factor_ = copy.correction_factor_;
        noiseScaleDeg_ = copy.noiseScaleDeg_;
        is_ntt_form_ = copy.is_ntt_form_;
        is_asymmetric_ = copy.is_asymmetric_;
        if (copy.prng_seed() != nullptr) {
            prng_seed_.acquire(cahel::util::allocate<uint8_t>(cahel::util::Global(),
                                                              cahel::util::global_variables::prng_seed_byte_count));
            CUDA_CHECK(cudaMemcpy(prng_seed(), copy.prng_seed(),
                                  cahel::util::global_variables::prng_seed_byte_count * sizeof(uint8_t),
                                  cudaMemcpyDeviceToDevice));
        }
        if (copy.data() != nullptr) {
            data_.acquire(cahel::util::allocate<uint64_t>(cahel::util::Global(),
                                                          size_ * coeff_modulus_size_ * poly_modulus_degree_));
            CUDA_CHECK(cudaMemcpy(data(), copy.data(),
                                  size_ * coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                                  cudaMemcpyDeviceToDevice));
        }
    }

    // copy assignment
    CAHELGPUCiphertext &operator=(const CAHELGPUCiphertext &copy) {
        if (this != (CAHELGPUCiphertext *) &copy) {
            parms_id_ = copy.parms_id_;
            chain_index_ = copy.chain_index_;
            size_ = copy.size_;
            poly_modulus_degree_ = copy.poly_modulus_degree_;
            coeff_modulus_size_ = copy.coeff_modulus_size_;
            scale_ = copy.scale_;
            correction_factor_ = copy.correction_factor_;
            noiseScaleDeg_ = copy.noiseScaleDeg_;
            is_ntt_form_ = copy.is_ntt_form_;
            is_asymmetric_ = copy.is_asymmetric_;
            if (copy.prng_seed() != nullptr) {
                prng_seed_.acquire(cahel::util::allocate<uint8_t>(cahel::util::Global(),
                                                                  cahel::util::global_variables::prng_seed_byte_count));
                CUDA_CHECK(cudaMemcpy(prng_seed(), copy.prng_seed(),
                                      cahel::util::global_variables::prng_seed_byte_count * sizeof(uint8_t),
                                      cudaMemcpyDeviceToDevice));
            }
            if (copy.data() != nullptr) {
                data_.acquire(cahel::util::allocate<uint64_t>(cahel::util::Global(),
                                                              size_ * coeff_modulus_size_ * poly_modulus_degree_ *
                                                              sizeof(uint64_t)));
                CUDA_CHECK(cudaMemcpy(data(), copy.data(),
                                      size_ * coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                                      cudaMemcpyDeviceToDevice));
            }
        }
        return *this;
    }

    // move constructor
    CAHELGPUCiphertext(CAHELGPUCiphertext &&move) noexcept {
        parms_id_ = move.parms_id_;
        chain_index_ = move.chain_index_;
        size_ = move.size_;
        poly_modulus_degree_ = move.poly_modulus_degree_;
        coeff_modulus_size_ = move.coeff_modulus_size_;
        scale_ = move.scale_;
        correction_factor_ = move.correction_factor_;
        noiseScaleDeg_ = move.correction_factor_;
        is_ntt_form_ = move.is_ntt_form_;
        is_asymmetric_ = move.is_asymmetric_;
        prng_seed_.acquire(move.prng_seed_);
        data_.acquire(move.data_);
    }

    // move assignment
    CAHELGPUCiphertext &operator=(CAHELGPUCiphertext &&move) {
        parms_id_ = move.parms_id_;
        chain_index_ = move.chain_index_;
        size_ = move.size_;
        poly_modulus_degree_ = move.poly_modulus_degree_;
        coeff_modulus_size_ = move.coeff_modulus_size_;
        scale_ = move.scale_;
        correction_factor_ = move.correction_factor_;
        noiseScaleDeg_ = move.correction_factor_;
        is_ntt_form_ = move.is_ntt_form_;
        is_asymmetric_ = move.is_asymmetric_;
        prng_seed_.acquire(move.prng_seed_);
        data_.acquire(move.data_);
        return *this;
    }
    
    inline void free() {
        parms_id_ = cahel::parms_id_zero;
        chain_index_ = 0;
        size_ = 0;
        poly_modulus_degree_ = 0;
        coeff_modulus_size_ = 0;
        scale_ = 0;
        correction_factor_ = 0;
        noiseScaleDeg_ = 0;
    }

    ~CAHELGPUCiphertext() {
        free();
    }

    [[nodiscard]] std::string to_string() const {
        if (is_ntt_form())
            throw std::invalid_argument("cannot convert NTT transformed plaintext to string");
        return cahel::util::poly_to_hex_string(data(), size_ * coeff_modulus_size_ * poly_modulus_degree_, 1);
    }

    inline void save(std::ostream &stream) {
        auto old_except_mask = stream.exceptions();
        try {
            // Throw exceptions on std::ios_base::badbit and std::ios_base::failbit
            stream.exceptions(std::ios_base::badbit | std::ios_base::failbit);

            stream.write(reinterpret_cast<const char *>(&parms_id_), sizeof(cahel::parms_id_type));
            stream.write(reinterpret_cast<const char *>(&is_ntt_form_), sizeof(bool));
            auto chain64 = static_cast<uint64_t>(chain_index_);
            stream.write(reinterpret_cast<const char *>(&chain64), sizeof(uint64_t));
            auto size64 = static_cast<uint64_t>(size_);
            stream.write(reinterpret_cast<const char *>(&size64), sizeof(uint64_t));
            auto poly_modulus_degree64 = static_cast<uint64_t>(poly_modulus_degree_);
            stream.write(reinterpret_cast<const char *>(&poly_modulus_degree64), sizeof(uint64_t));
            auto coeff_modulus_size64 = static_cast<uint64_t>(coeff_modulus_size_);
            stream.write(reinterpret_cast<const char *>(&coeff_modulus_size64), sizeof(uint64_t));
            stream.write(reinterpret_cast<const char *>(&scale_), sizeof(double));

            uint64_t data_size = size_ * coeff_modulus_size_ * poly_modulus_degree_;
            stream.write(reinterpret_cast<const char *>(&data_size), sizeof(std::uint64_t));

            std::vector<uint64_t> temp_data;
            temp_data.resize(data_size);
            CUDA_CHECK(cudaMemcpy(temp_data.data(), data_.get(),
                                  size_ * coeff_modulus_size_ * poly_modulus_degree_ * sizeof(uint64_t),
                                  cudaMemcpyDeviceToHost));
            stream.write(
                    reinterpret_cast<const char *>(temp_data.data()),
                    static_cast<std::streamsize>(cahel::util::mul_safe(data_size, sizeof(uint64_t))));

            std::vector<uint8_t> temp_seed;
            temp_seed.resize(cahel::util::global_variables::prng_seed_byte_count);
            CUDA_CHECK(cudaMemcpy(temp_seed.data(), prng_seed_.get(),
                                  cahel::util::global_variables::prng_seed_byte_count * sizeof(uint8_t),
                                  cudaMemcpyDeviceToHost));
            stream.write(
                    reinterpret_cast<const char *>(temp_seed.data()),
                    static_cast<std::streamsize>(cahel::util::mul_safe(
                            cahel::util::global_variables::prng_seed_byte_count, sizeof(uint8_t))));
        }
        catch (const std::ios_base::failure &) {
            stream.exceptions(old_except_mask);
            throw std::runtime_error("I/O error");
        }
        catch (...) {
            stream.exceptions(old_except_mask);
            throw;
        }
        stream.exceptions(old_except_mask);
    }

    inline void load(const CAHELGPUContext &context, std::istream &stream) {
        CAHELGPUCiphertext new_data(context);
        auto old_except_mask = stream.exceptions();
        try {
            // Throw exceptions on std::ios_base::badbit and std::ios_base::failbit
            stream.exceptions(std::ios_base::badbit | std::ios_base::failbit);

            cahel::parms_id_type parms_id{};
            stream.read(reinterpret_cast<char *>(&parms_id), sizeof(cahel::parms_id_type));
            bool is_ntt_form_byte = 0;
            stream.read(reinterpret_cast<char *>(&is_ntt_form_byte), sizeof(bool));
            uint64_t chain64 = 0;
            stream.read(reinterpret_cast<char *>(&chain64), sizeof(uint64_t));
            uint64_t size64 = 0;
            stream.read(reinterpret_cast<char *>(&size64), sizeof(uint64_t));
            uint64_t poly_modulus_degree64 = 0;
            stream.read(reinterpret_cast<char *>(&poly_modulus_degree64), sizeof(uint64_t));
            uint64_t coeff_modulus_size64 = 0;
            stream.read(reinterpret_cast<char *>(&coeff_modulus_size64), sizeof(uint64_t));
            double scale = 0;
            stream.read(reinterpret_cast<char *>(&scale), sizeof(double));

            // Set values already at this point for the metadata validity check
            new_data.parms_id_ = parms_id;
            new_data.is_ntt_form_ = is_ntt_form_byte;
            new_data.chain_index_ = static_cast<size_t>(chain64);
            new_data.size_ = static_cast<size_t>(size64);
            new_data.poly_modulus_degree_ = static_cast<size_t>(poly_modulus_degree64);
            new_data.coeff_modulus_size_ = static_cast<size_t>(coeff_modulus_size64);
            new_data.scale_ = scale;
            auto total_uint64_count =
                    cahel::util::mul_safe(new_data.size_, new_data.poly_modulus_degree_, new_data.coeff_modulus_size_);

            uint64_t data_size = 0;
            stream.read(reinterpret_cast<char *>(&data_size), sizeof(std::uint64_t));
            std::vector<uint64_t> temp_data;
            temp_data.resize(total_uint64_count);
            stream.read(
                    reinterpret_cast<char *>(temp_data.data()),
                    static_cast<std::streamsize>(cahel::util::mul_safe(total_uint64_count * sizeof(uint64_t))));

            new_data.data_.acquire(
                    cahel::util::allocate<uint64_t>(cahel::util::Global(), total_uint64_count * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemcpy(new_data.data_.get(), temp_data.data(), total_uint64_count * sizeof(uint64_t),
                                  cudaMemcpyHostToDevice));

            std::vector<uint8_t> temp_seed;
            temp_seed.resize(cahel::util::global_variables::prng_seed_byte_count);
            stream.read(
                    reinterpret_cast<char *>(temp_seed.data()),
                    static_cast<std::streamsize>(cahel::util::mul_safe(
                            cahel::util::global_variables::prng_seed_byte_count, sizeof(uint8_t))));
            CUDA_CHECK(cudaMemcpy(new_data.prng_seed_.get(), temp_seed.data(),
                                  cahel::util::global_variables::prng_seed_byte_count * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice));
        }
        catch (const std::ios_base::failure &) {
            stream.exceptions(old_except_mask);
            throw std::runtime_error("I/O error");
        }
        catch (...) {
            stream.exceptions(old_except_mask);
            throw;
        }
        stream.exceptions(old_except_mask);

        *this = std::move(new_data);
    }

    /** return the number of ploy in ciphertext
     */
    [[nodiscard]] __host__ __device__ __forceinline__ std::size_t size() const noexcept {
        return size_;
    }

    /* Reset and malloc the ciphertext
     * @notice: when size is larger, the previous data is copied.
     */
    void resize(const CAHELGPUContext &context, size_t chain_index, size_t size) {
        auto cahel_context = context.cpu_context_;
        auto &context_data = (cahel::CAHELContext::ContextData &) (cahel_context->get_context_data(chain_index));
        auto &parms = (cahel::EncryptionParameters &) (context_data.parms());
        auto &coeff_modulus = parms.coeff_modulus();
        auto coeff_modulus_size = coeff_modulus.size();
        auto poly_modulus_degree = parms.poly_modulus_degree();

        size_t old_size = size_ * coeff_modulus_size_ * poly_modulus_degree_;
        size_t new_size = size * coeff_modulus_size * poly_modulus_degree;

        cahel::util::Pointer<uint64_t> prev_data;

        if (new_size > old_size) {
            prev_data.acquire(data_);
            data_.acquire(cahel::util::allocate<uint64_t>(cahel::util::Global(),
                                                          size * poly_modulus_degree * coeff_modulus_size));
            CUDA_CHECK(cudaMemcpy(data_.get(), prev_data.get(), old_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        } else if (new_size > 0 && new_size < old_size) {
            prev_data.acquire(data_);
            data_.acquire(cahel::util::allocate<uint64_t>(cahel::util::Global(),
                                                          size * poly_modulus_degree * coeff_modulus_size));
            CUDA_CHECK(cudaMemcpy(data_.get(), prev_data.get(), new_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        }

        size_ = size;
        chain_index_ = chain_index;
        poly_modulus_degree_ = poly_modulus_degree;
        coeff_modulus_size_ = coeff_modulus_size;
        parms_id_ = cahel_context->get_parms_id(chain_index);
    }

    void resize(size_t size, size_t coeff_modulus_size, size_t poly_modulus_degree) {
        size_t old_size = size_ * coeff_modulus_size_ * poly_modulus_degree_;
        size_t new_size = size * coeff_modulus_size * poly_modulus_degree;
        cahel::util::Pointer<uint64_t> prev_data;
        prev_data.acquire(data_);
        data_.acquire(cahel::util::allocate<uint64_t>(cahel::util::Global(),
                                                      size * coeff_modulus_size * poly_modulus_degree));

        if (new_size > old_size)
            CUDA_CHECK(cudaMemcpy(data_.get(), prev_data.get(), old_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        if (new_size > 0 && new_size <= old_size)
            CUDA_CHECK(cudaMemcpy(data_.get(), prev_data.get(), new_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

        size_ = size;
        coeff_modulus_size_ = coeff_modulus_size;
        poly_modulus_degree_ = poly_modulus_degree;
    }

    [[nodiscard]] bool is_transparent() const {
        uint64_t *c1 = data() + poly_modulus_degree_ * coeff_modulus_size_;
        // CUDA_CHECK(cudaMemset(c1, 0, poly_modulus_degree_ * coeff_modulus_size_ * sizeof(uint64_t)));
        return zero_coeff_count_poly(c1, coeff_modulus_size_, poly_modulus_degree_) ==
               poly_modulus_degree_ * coeff_modulus_size_;
    }

    /**
    * Get the degree of the scaling factor for the encrypted message.
    */
    [[nodiscard]] size_t GetNoiseScaleDeg() const {
        return noiseScaleDeg_;
    }

    /**
    * Set the degree of the scaling factor for the encrypted message.
    */
    void SetNoiseScaleDeg(size_t noiseScaleDeg) {
        noiseScaleDeg_ = noiseScaleDeg;
    }

    /**
    Returns whether the ciphertext is in NTT form.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ bool is_ntt_form() const noexcept {
        return is_ntt_form_;
    }

    /**
    Returns whether the ciphertext is in NTT form.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ bool &is_ntt_form() noexcept {
        return is_ntt_form_;
    }

    /**
    Returns whether the ciphertext is encrypted using asymmetric encryption.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ bool &is_asymmetric() noexcept {
        return is_asymmetric_;
    }

    /**
    Returns a reference to parms_id.
    @see EncryptionParameters for more information about parms_id.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ auto &parms_id() noexcept {
        return parms_id_;
    }

    /**
    Returns a reference to parms_id.
    @see EncryptionParameters for more information about parms_id.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ auto &parms_id() const noexcept {
        return parms_id_;
    }

    /**
    Returns a const reference to chain_index.
    @see EncryptionParameters for more information about chain_index.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ auto &chain_index() noexcept {
        return chain_index_;
    }

    /**
    Returns a const reference to chain_index.
    @see EncryptionParameters for more information about chain_index.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ auto &chain_index() const noexcept {
        return chain_index_;
    }

    /**
    Returns a reference to the scale. This is only needed when using the
    CKKS encryption scheme. The user should have little or no reason to ever
    change the scale by hand.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ auto &scale() noexcept {
        return scale_;
    }

    /**
    Returns a constant reference to the scale. This is only needed when
    using the CKKS encryption scheme.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ auto &scale() const noexcept {
        return scale_;
    }

    void set_scale(double scale) {
        scale_ = scale;
    }

    /**
Returns a reference to the correction factor. This is only needed when using the BGV encryption scheme. The user
should have little or no reason to ever change the scale by hand.
*/
    [[nodiscard]] __host__ __device__ __forceinline__ std::uint64_t &correction_factor() noexcept {
        return correction_factor_;
    }

    /**
    Returns a constant reference to the correction factor. This is only needed when using the BGV encryption scheme.
    */
    [[nodiscard]] __host__ __device__ __forceinline__ const std::uint64_t &correction_factor() const noexcept {
        return correction_factor_;
    }

    __host__ __device__ __forceinline__ uint8_t *prng_seed() const {
        return (uint8_t *) (prng_seed_.get());
    }

    __host__ __device__ __forceinline__ uint64_t *data() const {
        return (uint64_t *) (data_.get());
    }
} CAHELGPUCiphertext;
