#pragma once

#include "context.h"

#include "plaintext.h"
#include "ciphertext.h"
#include "ntt.cuh"
#include "polymath.cuh"
#include "scalingvariant.cuh"
#include "prng.cuh"
#include "gpucontext.h"

#include <cuda_runtime_api.h>
#include <cuda.h>

typedef struct CAHELGPUPublicKey {

    CAHELGPUCiphertext pk_;

    CAHELGPUPublicKey() = default;

    explicit CAHELGPUPublicKey(const CAHELGPUContext &context) {}

    // copy constructor
    CAHELGPUPublicKey(const CAHELGPUPublicKey &copy) {
        pk_ = copy.pk_;
    }

    // move constructor
    CAHELGPUPublicKey(CAHELGPUPublicKey &&source) noexcept {
        pk_ = std::move(source.pk_);
    }

    // copy assignment
    CAHELGPUPublicKey &operator=(const CAHELGPUPublicKey &assign) {
        if (this != (CAHELGPUPublicKey *) &assign) {
            pk_ = assign.pk_;
        }
        return *this;
    }

    // move assignment
    CAHELGPUPublicKey &operator=(CAHELGPUPublicKey &&assign) noexcept {
        if (this != (CAHELGPUPublicKey *) &assign) {
            pk_ = std::move(assign.pk_);
        }

        return *this;
    }

    ~CAHELGPUPublicKey() = default;

    void save(std::ostream &stream) {
        pk_.save(stream);
    }

    void load(const CAHELGPUContext &context, std::istream &stream) {
        pk_.load(context, stream);
    }

    /**
    Returns a const reference to parms_id.
    */
    [[nodiscard]] inline auto &parms_id() const noexcept {
        return pk_.parms_id();
    }

    /** Encrypt zero using the public key, internal function, no modulus switch here.
     * @param[in] context CAHELGPUContext
     * @param[inout] cipher The generated ciphertext
     * @param[in] chain_index The id of the corresponding context data
     * @param[in] is_ntt_from Whether the ciphertext should be in NTT form
     */
    void encrypt_zero_asymmetric(const CAHELGPUContext &context, CAHELGPUCiphertext &cipher, size_t chain_index,
                                 bool is_ntt_from) const;

    /** Encrypt zero using the public key, and perform the model switch is neccessary
     * @brief pk [pk0, pk1], ternary variable u, cbd (gauss) noise e0, e1, return [pk0*u+e0, pk1*u+e1]
     * @param[in] context CAHELGPUContext
     * @param[inout] cipher The generated ciphertext
     * @param[in] chain_index The id of the corresponding context data
     * @param[in] save_seed Save random seed in ciphertext
     */
    void
    encrypt_zero_asymmetric_internal(const CAHELGPUContext &context, CAHELGPUCiphertext &cipher, size_t chain_index,
                                     bool save_seed) const;

    /** asymmetric encryption.
     * @brief: asymmetric encryption requires modulus switching.
     * @param[in] context CAHELGPUContext
     * @param[in] plain The data to be encrypted
     * @param[out] cipher The generated ciphertext
     * @param[in] save_seed Save random seed in ciphertext
     */
    void encrypt_asymmetric(const CAHELGPUContext &context, const CAHELGPUPlaintext &plain, CAHELGPUCiphertext &cipher,
                            bool save_seed = false);

    // for python wrapper

    inline CAHELGPUCiphertext encrypt_asymmetric(const CAHELGPUContext &context,
                                                 const CAHELGPUPlaintext &plain,
                                                 bool save_seed) {
        CAHELGPUCiphertext cipher(context);
        encrypt_asymmetric(context, plain, cipher, save_seed);
        return cipher;
    }

} CAHELGPUPublicKey;

/** CAHELGPURelinKey contains the relinear key in RNS and NTT form
 * gen_flag denotes whether the secret key has been generated.
 */
typedef struct CAHELGPURelinKey {
    cahel::parms_id_type parms_id_ = cahel::parms_id_zero;

    std::vector<CAHELGPUPublicKey> public_keys_;

    cahel::util::Pointer<uint64_t *> public_keys_ptr_;

    size_t pk_num_ = 0;

    bool gen_flag_ = false;

    CAHELGPURelinKey() = default;

    /**
     * Malloc for pk_
     * The pk_ stores the relinear keys  in rns and NTT form
     * gen_flag is set to false, until the gen_secretkey has been invoked
     */
    explicit CAHELGPURelinKey(const CAHELGPUContext &context) {
        auto cahel_context = context.cpu_context_;
        auto first_index = cahel_context->first_parm_index();
        auto &context_data = cahel_context->get_context_data(first_index);
        auto &parms = context_data.parms();
        auto &key_modulus = parms.key_modulus();

        size_t size_P = parms.special_modulus_size();
        size_t size_QP = key_modulus.size();
        size_t size_Q = size_QP - size_P;
        if (size_P != 0) {
            size_t dnum = size_Q / size_P;
            pk_num_ = dnum;
        }
        gen_flag_ = false;
    }

    // copy constructor
    CAHELGPURelinKey(const CAHELGPURelinKey &copy) {
        for (size_t idx{0}; idx < copy.pk_num_; idx++) {
            public_keys_.push_back(copy.public_keys_[idx]);
        }
        if (copy.public_keys_ptr_.get() != nullptr) {
            public_keys_ptr_.acquire(cahel::util::allocate<uint64_t *>(cahel::util::Global(), pk_num_));
            CUDA_CHECK(cudaMemcpy(public_keys_ptr_.get(), copy.public_keys_ptr_.get(),
                                  pk_num_ * sizeof(uint64_t *), cudaMemcpyDeviceToDevice));
        }
        parms_id_ = copy.parms_id_;
        pk_num_ = copy.pk_num_;
        gen_flag_ = copy.gen_flag_;
    }

    // move constructor
    CAHELGPURelinKey(CAHELGPURelinKey &&source) noexcept {
        pk_num_ = source.pk_num_;
        public_keys_.resize(pk_num_);
        for (size_t idx{0}; idx < pk_num_; idx++) {
            public_keys_[idx] = std::move(source.public_keys_[idx]);
        }
        public_keys_ptr_.acquire(source.public_keys_ptr_);
        parms_id_ = source.parms_id_;
        gen_flag_ = source.gen_flag_;
    }

    // copy assignment
    CAHELGPURelinKey &operator=(const CAHELGPURelinKey &assign) {
        if (this != (CAHELGPURelinKey *) &assign) {
            for (size_t idx{0}; idx < assign.pk_num_; idx++) {
                public_keys_.push_back(assign.public_keys_[idx]);
            }
            if (assign.public_keys_ptr_.get() != nullptr) {
                public_keys_ptr_.acquire(cahel::util::allocate<uint64_t *>(cahel::util::Global(), pk_num_));
                CUDA_CHECK(cudaMemcpy(public_keys_ptr_.get(), assign.public_keys_ptr_.get(),
                                      pk_num_ * sizeof(uint64_t *), cudaMemcpyDeviceToDevice));
            }
            parms_id_ = assign.parms_id_;
            pk_num_ = assign.pk_num_;
            gen_flag_ = assign.gen_flag_;
        }

        return *this;
    }

    // move assignment
    CAHELGPURelinKey &operator=(CAHELGPURelinKey &&assign) noexcept {
        if (this != (CAHELGPURelinKey *) &assign) {
            pk_num_ = assign.pk_num_;
            public_keys_.resize(pk_num_);
            for (size_t idx{0}; idx < pk_num_; idx++) {
                public_keys_[idx] = std::move(assign.public_keys_[idx]);
            }
            public_keys_ptr_.acquire(assign.public_keys_ptr_);
            parms_id_ = assign.parms_id_;
            gen_flag_ = assign.gen_flag_;
        }

        return *this;
    }

    ~CAHELGPURelinKey() = default;

    void save(std::ostream &stream) {
        auto old_except_mask = stream.exceptions();
        try {
            // Throw exceptions on std::ios_base::badbit and std::ios_base::failbit
            stream.exceptions(std::ios_base::badbit | std::ios_base::failbit);
            stream.write(reinterpret_cast<const char *>(&parms_id_), sizeof(cahel::parms_id_type));
            stream.write(reinterpret_cast<const char *>(&gen_flag_), sizeof(bool));
            stream.write(reinterpret_cast<const char *>(&pk_num_), sizeof(size_t));
            for (size_t i = 0; i < pk_num_; i++) {
                public_keys_[i].save(stream);
            }
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

    void load(const CAHELGPUContext &context, std::istream &stream) {
        std::vector<CAHELGPUPublicKey> new_data;

        auto old_except_mask = stream.exceptions();
        try {
            // Throw exceptions on std::ios_base::badbit and std::ios_base::failbit
            stream.exceptions(std::ios_base::badbit | std::ios_base::failbit);

            cahel::parms_id_type parms_id{};
            stream.read(reinterpret_cast<char *>(&parms_id), sizeof(cahel::parms_id_type));
            parms_id_ = parms_id;
            stream.read(reinterpret_cast<char *>(&gen_flag_), sizeof(bool));
            stream.read(reinterpret_cast<char *>(&pk_num_), sizeof(size_t));
            new_data.resize(pk_num_);
            for (size_t i = 0; i < pk_num_; i++) {
                new_data[i].load(context, stream);
            }
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

        swap(public_keys_, new_data);
    }

    /**
    Returns a reference to parms_id.

    @see EncryptionParameters for more information about parms_id.
    */
    [[nodiscard]] inline auto &parms_id() noexcept {
        return parms_id_;
    }

    /**
    Returns a reference to parms_id.

    @see EncryptionParameters for more information about parms_id.
    */
    [[nodiscard]] inline auto &parms_id() const noexcept {
        return parms_id_;
    }

} CAHELGPURelinKey;

/** CAHELGPUGaloisKey stores Galois keys.
 * gen_flag denotes whether the Galois key has been generated.
 */
typedef struct CAHELGPUGaloisKey {
    cahel::parms_id_type parms_id_ = cahel::parms_id_zero;
    std::vector<CAHELGPURelinKey> relin_keys_;
    size_t relin_key_num_ = 0;
    bool gen_flag_ = false;

    /**
     * Malloc for pk_
     * The pk_ stores the relinear keys  in rns and NTT form
     * gen_flag is set to false, until the gen_secretkey has been invoked
     */
    explicit CAHELGPUGaloisKey(const CAHELGPUContext &context) {}

    // copy constructor
    CAHELGPUGaloisKey(const CAHELGPUGaloisKey &copy) {
        for (size_t idx{0}; idx < copy.relin_key_num_; idx++) {
            relin_keys_.push_back(copy.relin_keys_[idx]);
        }
        parms_id_ = copy.parms_id_;
        relin_key_num_ = copy.relin_key_num_;
        gen_flag_ = copy.gen_flag_;
    }

    // move constructor
    CAHELGPUGaloisKey(CAHELGPUGaloisKey &&source) noexcept {
        relin_key_num_ = source.relin_key_num_;
        relin_keys_.resize(relin_key_num_);
        for (size_t idx{0}; idx < relin_key_num_; idx++) {
            relin_keys_[idx] = std::move(source.relin_keys_[idx]);
        }
        parms_id_ = source.parms_id_;
        gen_flag_ = source.gen_flag_;
    }

    // copy assignment
    CAHELGPUGaloisKey &operator=(const CAHELGPUGaloisKey &assign) {
        if (this != (CAHELGPUGaloisKey *) &assign) {
            for (size_t idx{0}; idx < assign.relin_key_num_; idx++) {
                relin_keys_.push_back(assign.relin_keys_[idx]);
            }
            parms_id_ = assign.parms_id_;
            relin_key_num_ = assign.relin_key_num_;
            gen_flag_ = assign.gen_flag_;
        }

        return *this;
    }

    // move assignment
    CAHELGPUGaloisKey &operator=(CAHELGPUGaloisKey &&assign) noexcept {
        if (this != (CAHELGPUGaloisKey *) &assign) {
            relin_key_num_ = assign.relin_key_num_;
            relin_keys_.resize(relin_key_num_);
            for (size_t idx{0}; idx < relin_key_num_; idx++) {
                relin_keys_[idx] = std::move(assign.relin_keys_[idx]);
            }
            parms_id_ = assign.parms_id_;
            gen_flag_ = assign.gen_flag_;
        }

        return *this;
    }

    ~CAHELGPUGaloisKey() = default;

    void save(std::ostream &stream) {
        auto old_except_mask = stream.exceptions();
        try {
            // Throw exceptions on std::ios_base::badbit and std::ios_base::failbit
            stream.exceptions(std::ios_base::badbit | std::ios_base::failbit);
            stream.write(reinterpret_cast<const char *>(&parms_id_), sizeof(cahel::parms_id_type));
            stream.write(reinterpret_cast<const char *>(&gen_flag_), sizeof(bool));
            stream.write(reinterpret_cast<const char *>(&relin_key_num_), sizeof(size_t));
            for (size_t i = 0; i < relin_key_num_; i++) {
                relin_keys_[i].save(stream);
            }
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

    void load(const CAHELGPUContext &context, std::istream &stream) {
        std::vector<CAHELGPURelinKey> new_data;

        auto old_except_mask = stream.exceptions();
        try {
            // Throw exceptions on std::ios_base::badbit and std::ios_base::failbit
            stream.exceptions(std::ios_base::badbit | std::ios_base::failbit);

            cahel::parms_id_type parms_id{};
            stream.read(reinterpret_cast<char *>(&parms_id), sizeof(cahel::parms_id_type));
            parms_id_ = parms_id;
            stream.read(reinterpret_cast<char *>(&gen_flag_), sizeof(bool));
            stream.read(reinterpret_cast<char *>(&relin_key_num_), sizeof(size_t));
            new_data.resize(relin_key_num_);
            for (size_t i = 0; i < relin_key_num_; i++) {
                new_data[i].load(context, stream);
            }
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

        swap(relin_keys_, new_data);
    }

    /**
    Returns a reference to parms_id.

    @see EncryptionParameters for more information about parms_id.
    */
    [[nodiscard]] inline auto &parms_id() noexcept {
        return parms_id_;
    }

    /**
    Returns a reference to parms_id.

    @see EncryptionParameters for more information about parms_id.
    */
    [[nodiscard]] inline auto &parms_id() const noexcept {
        return parms_id_;
    }

} CAHELGPUGaloisKey;

/** CAHELGPUSecretKey contains the secret key in RNS and NTT form
 * gen_flag denotes whether the secret key has been generated.
 */
typedef struct CAHELGPUSecretKey {
    // std::vector<Pointer<uint64_t>> data_rns_;
    cahel::util::Pointer<uint64_t> data_rns_;
    uint64_t chain_index_ = 0;
    bool gen_flag_ = false;
    uint64_t sk_max_power_ = 0;          // the max power of secret key
    cahel::util::Pointer<uint64_t> secret_key_array_; // the powers of secret key
    uint64_t poly_modulus_degree_ = 0;
    uint64_t coeff_modulus_size_ = 0;

    /** Malloc for data_rns_
     * The data_rns_ stores the secret key in rns and NTT form
     * gen_flag is set to false, until the gen_secretkey has been invoked
     */
    explicit CAHELGPUSecretKey(const cahel::EncryptionParameters &parms) {
        poly_modulus_degree_ = parms.poly_modulus_degree();
        coeff_modulus_size_ = parms.coeff_modulus().size();
        // data_rns_.resize(coeff_modulus_size);
        // for (size_t i = 0; i < coeff_modulus_size; i++)
        //     data_rns_[i].acquire(allocate<uint64_t>(Global(), poly_modulus_degree_));
        data_rns_.acquire(
                cahel::util::allocate<uint64_t>(cahel::util::Global(), poly_modulus_degree_ * coeff_modulus_size_));
        gen_flag_ = false;
    }

    // copy constructor
    CAHELGPUSecretKey(const CAHELGPUSecretKey &copy) {
        chain_index_ = copy.chain_index_;
        gen_flag_ = copy.gen_flag_;
        sk_max_power_ = copy.sk_max_power_;
        poly_modulus_degree_ = copy.poly_modulus_degree_;
        coeff_modulus_size_ = copy.coeff_modulus_size_;
        // auto data_rns_size = copy.data_rns_.size();
        // data_rns_.resize(data_rns_size);
        // for (size_t i = 0; i < data_rns_size; i++)
        // {
        //     data_rns_[i].acquire(allocate<uint64_t>(Global(), poly_modulus_degree_));
        //     CUDA_CHECK(cudaMemcpy(data_rns_[i].get(), copy.data_rns_[i].get(), poly_modulus_degree_ * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        // }

        data_rns_.acquire(
                cahel::util::allocate<uint64_t>(cahel::util::Global(), poly_modulus_degree_ * coeff_modulus_size_));
        CUDA_CHECK(cudaMemcpy(data_rns_.get(), copy.data_rns_.get(),
                              poly_modulus_degree_ * coeff_modulus_size_ * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

        auto secret_key_array_size = sk_max_power_ * poly_modulus_degree_ * coeff_modulus_size_;
        secret_key_array_.acquire(cahel::util::allocate<uint64_t>(cahel::util::Global(), secret_key_array_size));
        CUDA_CHECK(cudaMemcpy(secret_key_array_.get(), copy.secret_key_array_.get(),
                              secret_key_array_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
    }

    // move constructor
    CAHELGPUSecretKey(CAHELGPUSecretKey &&copy) noexcept {
        chain_index_ = copy.chain_index_;
        gen_flag_ = copy.gen_flag_;
        sk_max_power_ = copy.sk_max_power_;
        poly_modulus_degree_ = copy.poly_modulus_degree_;
        coeff_modulus_size_ = copy.coeff_modulus_size_;
        // data_rns_.resize(data_rns_size);
        // for (size_t i = 0; i < data_rns_size; i++)
        // {
        //     data_rns_[i].acquire(data_rns_[i]);
        // }
        data_rns_.acquire(copy.data_rns_);
        secret_key_array_.acquire(copy.secret_key_array_);
    }

    // copy assignment
    CAHELGPUSecretKey &operator=(const CAHELGPUSecretKey &source) {
        if (this != (CAHELGPUSecretKey *) &source) {
            chain_index_ = source.chain_index_;
            gen_flag_ = source.gen_flag_;
            sk_max_power_ = source.sk_max_power_;
            poly_modulus_degree_ = source.poly_modulus_degree_;
            coeff_modulus_size_ = source.coeff_modulus_size_;
            // data_rns_.resize(data_rns_size);
            // for (size_t i = 0; i < data_rns_size; i++)
            // {
            //     data_rns_[i].acquire(allocate<uint64_t>(Global(), poly_modulus_degree_));
            //     CUDA_CHECK(cudaMemcpy(data_rns_[i].get(), source.data_rns_[i].get(), poly_modulus_degree_ * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
            // }
            data_rns_.acquire(
                    cahel::util::allocate<uint64_t>(cahel::util::Global(), poly_modulus_degree_ * coeff_modulus_size_));
            CUDA_CHECK(cudaMemcpy(data_rns_.get(), source.data_rns_.get(),
                                  poly_modulus_degree_ * coeff_modulus_size_ * sizeof(uint64_t),
                                  cudaMemcpyDeviceToDevice));

            auto secret_key_array_size = sk_max_power_ * poly_modulus_degree_ * coeff_modulus_size_;
            secret_key_array_.acquire(cahel::util::allocate<uint64_t>(cahel::util::Global(), secret_key_array_size));
            CUDA_CHECK(cudaMemcpy(secret_key_array_.get(), source.secret_key_array_.get(),
                                  secret_key_array_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        }

        return *this;
    }

    // move assignment
    CAHELGPUSecretKey &operator=(CAHELGPUSecretKey &&source) noexcept {
        if (this != (CAHELGPUSecretKey *) &source) {
            chain_index_ = source.chain_index_;
            gen_flag_ = source.gen_flag_;
            sk_max_power_ = source.sk_max_power_;
            poly_modulus_degree_ = source.poly_modulus_degree_;
            // auto data_rns_size = source.data_rns_.size();
            coeff_modulus_size_ = source.coeff_modulus_size_;
            // data_rns_.resize(data_rns_size);
            // for (size_t i = 0; i < data_rns_size; i++)
            //     data_rns_[i].acquire(source.data_rns_[i]);
            data_rns_.acquire(source.data_rns_);
            secret_key_array_.acquire(source.secret_key_array_);
        }

        return *this;
    }

    ~CAHELGPUSecretKey() {
        gen_flag_ = false;
    }

    /** Generate the secret key for the specified param
     * As one invocation of salsa20 will generate 64 bytes, for key generation, one byte could generate one ternary
     * Therefore, we only need random number of length poly_degree/64 bytes.
     * @param[in] context CAHELGPUContext
     */
    void gen_secretkey(const CAHELGPUContext &context);

    /** Generate the powers of secret key
     * @param[in] context CAHELGPUContext
     * @param[in] max_power the mox power of secret key
     * @param[out] secret_key_array
     */
    void compute_secret_key_array(const CAHELGPUContext &context, size_t max_power);

    /** Generate one public key for this secret key
     * Return CAHELGPUPublicKey
     * @param[in] context CAHELGPUContext
     * @param[inout] pk The generated public key
     * @param[in] save_seed Save random seed in public key struct or not
     @throws std::invalid_argument if secret key or public keyhas not been inited
     */
    void gen_publickey(const CAHELGPUContext &context, CAHELGPUPublicKey &pk, bool save_seed = true) const;

    /** Generate one public key for this secret key
     * Return CAHELGPUPublicKey
     * @param[in] context CAHELGPUContext
     * @param[inout] relin_key The generated relinear key
     * @param[in] save_seed Save random seed in relinear key struct or not
     * @throws std::invalid_argument if secret key or relinear key has not been inited
     */
    void generate_one_kswitch_key(const CAHELGPUContext &context, uint64_t *new_key, CAHELGPURelinKey &relin_key,
                                  bool save_seed = false) const;

    void gen_relinkey(const CAHELGPUContext &context, CAHELGPURelinKey &relin_key, bool save_seed = false);

    /** Generates Galois keys and stores the result in destination. Every time
     * this function is called, new Galois keys will be generated.
     * @param[in] context CAHELGPUContext
     * @param[out] galois_key The generated galois key
     */
    void
    create_galois_keys(const CAHELGPUContext &context, CAHELGPUGaloisKey &galois_key, bool save_seed = false) const;

    /** Encrypt zero using the secret key, the ciphertext is in NTT form
     * @param[in] context CAHELGPUContext
     * @param[inout] cipher The generated ciphertext
     * @param[in] chain_index The index of the context data
     * @param[in] is_ntt_form Whether the ciphertext needs to be in NTT form
     * @param[in] save_seed Save random seed in ciphertext
     */
    void encrypt_zero_symmetric(const CAHELGPUContext &context, CAHELGPUCiphertext &cipher, size_t chain_index,
                                bool is_ntt_form, bool save_seed) const;

    /** Symmetric encryption, the plaintext and ciphertext are in NTT form
     * @param[in] context CAHELGPUContext
     * @param[in] plain The data to be encrypted
     * @param[out] cipher The generated ciphertext
     * @param[in] save_seed Save random seed in ciphertext
     */
    void encrypt_symmetric(const CAHELGPUContext &context, const CAHELGPUPlaintext &plain, CAHELGPUCiphertext &cipher,
                           bool save_seed) const;

    /** decryption
     * @param[in] context CAHELGPUContext
     * @param[in] cipher The ciphertext to be decrypted
     * @param[out] plain The plaintext
     */
    void decrypt(const CAHELGPUContext &context, const CAHELGPUCiphertext &cipher, CAHELGPUPlaintext &plain);

    void
    bfv_decrypt(const CAHELGPUContext &context, const CAHELGPUCiphertext &encrypted, CAHELGPUPlaintext &destination);

    void
    ckks_decrypt(const CAHELGPUContext &context, const CAHELGPUCiphertext &encrypted, CAHELGPUPlaintext &destination);

    void
    bgv_decrypt(const CAHELGPUContext &context, const CAHELGPUCiphertext &encrypted, CAHELGPUPlaintext &destination);

    // for python wrapper

    [[nodiscard]] inline CAHELGPUCiphertext encrypt_symmetric(const CAHELGPUContext &context,
                                                              const CAHELGPUPlaintext &plain,
                                                              bool save_seed) const {
        CAHELGPUCiphertext cipher(context);
        encrypt_symmetric(context, plain, cipher, save_seed);
        return cipher;
    }

    [[nodiscard]] inline CAHELGPUPlaintext decrypt(const CAHELGPUContext &context,
                                                   const CAHELGPUCiphertext &cipher) {
        CAHELGPUPlaintext plain(context);
        decrypt(context, cipher, plain);
        return plain;
    }

    /**
    Computes the invariant noise budget (in bits) of a ciphertext. The
    invariant noise budget measures the amount of room there is for the noise
    to grow while ensuring correct decryptions. This function works only with
    the BFV scheme.
    * @param[in] context CAHELGPUContext
    * @param[in] cipher The ciphertext to be decrypted
    */
    [[nodiscard]] int invariant_noise_budget(const CAHELGPUContext &context, const CAHELGPUCiphertext &cipher);

    __host__ __device__ inline uint64_t *secret_key_array() const {
        return (uint64_t *) (secret_key_array_.get());
    }

} CAHELGPUSecretKey;
