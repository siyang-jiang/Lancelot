#pragma once

#include <memory>

#include "context.h"
#include "util/encryptionparams.h"
#include "util/modulus.h"

#include "util/galois.h"
#include "galois.cuh"
#include "gputype.h"
#include "rns.cuh"
#include "mempool.cuh"

typedef struct CAHELGPUContext {
    std::vector<cudaStream_t> sid_vec_;

    DNTTTable gpu_rns_tables_;

    DNTTTable gpu_plain_tables_;

    cahel::util::Pointer<uint64_t> in_;

    cahel::util::Pointer<DMulUIntMod> coeff_div_plain_;     // stores all the values for all possible modulus switch, auto choose the corresponding start pos
    cahel::util::Pointer<DMulUIntMod> plain_modulus_shoup_; // shoup pre-computations of (t mod qi)

    cahel::util::Pointer<uint64_t> plain_upper_half_increment_;

    cahel::util::Pointer<uint8_t> prng_seed_;
    // prng seed
    std::size_t coeff_mod_size_ = 0;                     // corresponding to the key param index, i.e., all coeff prime exists.
    std::size_t poly_degree_ = 0;                        // unchanged
    std::shared_ptr<cahel::CAHELContext> cpu_context_; // the corresponding context
    std::vector<DRNSTool> gpu_rns_tool_vec_;                  // changed during modulus switch
    std::vector<DCKKSEncoderInfo> gpu_ckks_msg_vec_;
    std::shared_ptr<CAHELGPUGaloisTool> key_galois_tool_;

    /**
     * Creates an instance of CAHELGPUContext, performs several pre-computations
      on the given EncryptionParameters, and malloc the memory for further computation.

      @param[in] parms The encryption parameters
      @param[in] expand_mod_chain Determines whether the modulus switching chain
          should be created
      @param[in] sec_level Determines whether a specific security level should be
          enforced according to HomomorphicEncryption.org security standard
     */

    CAHELGPUContext(const CAHELGPUContext &) = delete;

    void operator=(const CAHELGPUContext &) = delete;

    explicit CAHELGPUContext(const cahel::EncryptionParameters &parms,
                             bool expand_mod_chain = true,
                             cahel::sec_level_type sec_level = cahel::sec_level_type::tc128) {

        cpu_context_ = std::make_shared<cahel::CAHELContext>(parms, expand_mod_chain, sec_level);
        poly_degree_ = parms.poly_modulus_degree();

        // Create CUDA streams
        sid_vec_.resize(cahel::util::sid_count);
        for (auto &i: sid_vec_) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&i, cudaStreamNonBlocking));
        }

        auto &coeff_modulus_cpu = parms.coeff_modulus();
        coeff_mod_size_ = coeff_modulus_cpu.size();
        auto &small_ntt_tables = cpu_context_->get_context_data(0).small_ntt_tables();
        gpu_rns_tables().init(poly_degree_, coeff_mod_size_);
        for (size_t i = 0; i < coeff_mod_size_; i++) {
            DModulus temp = DModulus(coeff_modulus_cpu[i].value(),
                                     coeff_modulus_cpu[i].const_ratio()[0],
                                     coeff_modulus_cpu[i].const_ratio()[1]);
            gpu_rns_tables().set(&temp,
                                 (uint64_t *) (small_ntt_tables[i].get_from_root_powers().data()),
                                 (uint64_t *) (small_ntt_tables[i].get_from_inv_root_powers_div2().data()),
                                 i);
        }

        prng_seed_.acquire(
                cahel::util::allocate<uint8_t>(cahel::util::Global(),
                                               cahel::util::global_variables::prng_seed_byte_count));

        in_.acquire(cahel::util::allocate<uint64_t>(cahel::util::Global(), coeff_mod_size_ * poly_degree_));

        // Construct gpu rns tool from cpu rns tool
        auto gpu_rns_tool_vec_size = cpu_context_->total_parm_size();
        gpu_rns_tool_vec_.resize(gpu_rns_tool_vec_size);
        for (size_t i = 0; i < gpu_rns_tool_vec_size; i++) {
            gpu_rns_tool_vec()[i].init(*(cpu_context_->get_context_data_rns_tool(i)));
        }

        if (parms.scheme() == cahel::scheme_type::bfv || parms.scheme() == cahel::scheme_type::bgv) {
            auto &plain_ntt_tables = cpu_context_->get_context_data(0).plain_ntt_tables();
            auto &plain_modulus_cpu = parms.plain_modulus();
            gpu_plain_tables().init(poly_degree_, 1);
            DModulus temp = DModulus(plain_modulus_cpu.value(),
                                     plain_modulus_cpu.const_ratio()[0],
                                     plain_modulus_cpu.const_ratio()[1]);
            gpu_plain_tables().set(&temp,
                                   (uint64_t *) (plain_ntt_tables[0].get_from_root_powers().data()),
                                   (uint64_t *) (plain_ntt_tables[0].get_from_inv_root_powers_div2().data()),
                                   0);

            plain_modulus_shoup_.acquire(cahel::util::allocate<DMulUIntMod>(cahel::util::Global(), coeff_mod_size_));
            CUDA_CHECK(cudaMemcpy((void *) (plain_modulus_shoup_.get()),
                                  cpu_context_->get_context_data(0).plain_modulus_shoup().data(),
                                  coeff_mod_size_ * sizeof(DMulUIntMod),
                                  cudaMemcpyHostToDevice));
        }

        if (parms.scheme() == cahel::scheme_type::bfv) {
            auto coeff_div_plain_size =
                    (coeff_mod_size_ * 2 - cpu_context_->total_parm_size() + 1) * cpu_context_->total_parm_size() / 2;
            coeff_div_plain_.acquire(cahel::util::allocate<DMulUIntMod>(cahel::util::Global(), coeff_div_plain_size));
            auto cdp_pos = 0;
            for (size_t i = 0; i < cpu_context_->total_parm_size(); i++) {
                auto size = cpu_context_->get_context_data(i).coeff_div_plain_modulus().size();
                // force to memcpy, as the type is different but the values are consistent
                CUDA_CHECK(cudaMemcpy((void *) (coeff_div_plain_.get() + cdp_pos),
                                      (void *) (cpu_context_->get_context_data(i).coeff_div_plain_modulus().data()),
                                      size * sizeof(DMulUIntMod), cudaMemcpyHostToDevice));
                cdp_pos += size;
            }

            plain_upper_half_increment_.acquire(
                    cahel::util::allocate<uint64_t>(cahel::util::Global(), coeff_mod_size_));
            CUDA_CHECK(cudaMemcpy((void *) (plain_upper_half_increment_.get()),
                                  cpu_context_->get_context_data(0).plain_upper_half_increment().data(),
                                  coeff_mod_size_ * sizeof(uint64_t), cudaMemcpyHostToDevice));
        }

        int log_n = cahel::util::get_power_of_two(poly_degree_);
        bool is_bfv = (parms.scheme() == cahel::scheme_type::bfv);
        key_galois_tool_ = std::make_shared<CAHELGPUGaloisTool>(parms.galois_elts(), log_n, is_bfv);
    }

    ~CAHELGPUContext() = default;
//        cahel::util::global_variables::global_memory_pool->Release();

    void copy_data_to_device(uint64_t *in) const {
        cudaMemcpy(in_.get(), in, sizeof(uint64_t) * coeff_mod_size_ * poly_degree_, cudaMemcpyHostToDevice);
    }

    [[nodiscard]] inline size_t get_current_index() const {
        return cpu_context_->current_parm_index();
    }

    [[nodiscard]] inline size_t get_first_index() const {
        return cpu_context_->first_parm_index();
    }

    [[nodiscard]] inline size_t get_previous_index(size_t index) const {
        return cpu_context_->previous_parm_index(index);
    }

    [[nodiscard]] inline size_t get_next_index(size_t index) const {
        return cpu_context_->next_parm_index(index);
    }

    /**
     * Return the coeff_div_plain corresponding to the index
     */
    [[nodiscard]] inline DMulUIntMod *get_coeff_div_plain(size_t index) const {
        if (index > cpu_context_->total_parm_size())
            throw std::invalid_argument("index invalid");
        size_t pos = (coeff_mod_size_ * 2 + 1 - index) * index / 2;
        return coeff_div_plain() + pos;
    }

    [[nodiscard]] inline const char *parameter_error_message() const {
        if (cpu_context_ != nullptr)
            return cpu_context_->parameter_error_message();
        else
            return "CAHELContext is empty";
    }

    [[nodiscard]] inline const std::vector<cudaStream_t> &sid_vec() const {
        return sid_vec_;
    }

    [[nodiscard]] inline const DNTTTable &gpu_plain_tables() const noexcept {
        return gpu_plain_tables_;
    }

    inline DNTTTable &gpu_plain_tables() {
        return gpu_plain_tables_;
    }

    [[nodiscard]] inline const DNTTTable &gpu_rns_tables() const noexcept {
        return gpu_rns_tables_;
    }

    inline DNTTTable &gpu_rns_tables() {
        return gpu_rns_tables_;
    }

    [[nodiscard]] inline uint64_t *in() const {
        return in_.get();
    }

    [[nodiscard]] inline DMulUIntMod *coeff_div_plain() const {
        return coeff_div_plain_.get();
    }

    [[nodiscard]] inline DMulUIntMod *plain_modulus_shoup() const {
        return plain_modulus_shoup_.get();
    }

    [[nodiscard]] inline uint64_t *plain_upper_half_increment() const {
        return plain_upper_half_increment_.get();
    }

    [[nodiscard]] inline uint8_t *prng_seed() const {
        return prng_seed_.get();
    }

    [[nodiscard]] inline DRNSTool *gpu_rns_tool_vec() const {
        return (DRNSTool *) (gpu_rns_tool_vec_.data());
    }

    [[nodiscard]] inline DCKKSEncoderInfo *gpu_ckks_msg_vec() const {
        return (DCKKSEncoderInfo *) (gpu_ckks_msg_vec_.data());
    }
} CAHELGPUContext;
