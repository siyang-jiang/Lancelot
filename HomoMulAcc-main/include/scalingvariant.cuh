//
// Created by byte on 2021/12/31.
//

#ifndef _GPU_SCALINGVARIANT_H
#define _GPU_SCALINGVARIANT_H

#include "ciphertext.h"
#include "gpucontext.h"
#include "plaintext.h"

/** For BFV cipher + ceil(m*q/t);
 */
void multiply_add_plain_with_scaling_variant(const CAHELGPUContext &context,
                                             const CAHELGPUPlaintext &plain,
                                             size_t chain_index,
                                             CAHELGPUCiphertext &cipher);

void multiply_sub_plain_with_scaling_variant(const CAHELGPUContext &context,
                                             const CAHELGPUPlaintext &plain,
                                             size_t chain_index,
                                             CAHELGPUCiphertext &cipher);
#endif //_GPU_SCALINGVARIANT_H
