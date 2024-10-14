//
// Created by byte on 2021/12/31.
//
#include "polymath.cuh"
#include "scalingvariant.cuh"

using namespace std;
using namespace cahel;
using namespace cahel::util;

// Multiply plain by scalar coeff_div_plaintext and reposition if in upper-half.
// Result gets added into the c_0 term of ciphertext (c_0,c_1).
void multiply_add_plain_with_scaling_variant(const CAHELGPUContext &context, const CAHELGPUPlaintext &plain,
                                             size_t chain_index, CAHELGPUCiphertext &cipher)
{
    auto cahel_context = context.cpu_context_;
    auto &context_data = (CAHELContext::ContextData &)(cahel_context->get_context_data(chain_index));
    auto &parms = (EncryptionParameters &)(context_data.parms());
    auto &rns_tool = context.gpu_rns_tool_vec_[chain_index];
    auto poly_degree = parms.poly_modulus_degree(); // = N
    auto &coeff_modulus = parms.coeff_modulus();    // coeff modulus
    auto coeff_mod_size = coeff_modulus.size();
    uint64_t t = parms.plain_modulus().value();

    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    bfv_add_timesQ_overt_kernel<<<gridDimGlb, blockDimGlb>>>(
            cipher.data(), plain.data(),
            rns_tool.negQl_mod_t_,
            rns_tool.tInv_mod_q_.get(),
            context.gpu_rns_tables().modulus(),
            t, poly_degree, coeff_mod_size);

    cipher.chain_index() = chain_index;
    cipher.poly_modulus_degree_ = poly_degree;
    cipher.coeff_modulus_size_ = coeff_mod_size;
}

void multiply_sub_plain_with_scaling_variant(const CAHELGPUContext &context, const CAHELGPUPlaintext &plain,
                                             size_t chain_index, CAHELGPUCiphertext &cipher)
{
    auto cahel_context = context.cpu_context_;
    auto &context_data = (CAHELContext::ContextData &)(cahel_context->get_context_data(chain_index));
    auto &parms = (EncryptionParameters &)(context_data.parms());
    auto &rns_tool = context.gpu_rns_tool_vec_[chain_index];
    auto poly_degree = parms.poly_modulus_degree(); // = N
    auto &coeff_modulus = parms.coeff_modulus();    // coeff modulus
    auto coeff_mod_size = coeff_modulus.size();
    uint64_t t = parms.plain_modulus().value();

    uint64_t gridDimGlb = poly_degree * coeff_mod_size / blockDimGlb.x;
    bfv_sub_timesQ_overt_kernel<<<gridDimGlb, blockDimGlb>>>(
            cipher.data(), plain.data(),
            rns_tool.negQl_mod_t_,
            rns_tool.tInv_mod_q_.get(),
            context.gpu_rns_tables().modulus(),
            t, poly_degree, coeff_mod_size);

    cipher.chain_index() = chain_index;
    cipher.poly_modulus_degree_ = poly_degree;
    cipher.coeff_modulus_size_ = coeff_mod_size;
}