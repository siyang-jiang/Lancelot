#ifndef __GPU_FFT_H_
#define __GPU_FFT_H_
#include "gpucontext.h"
#include "uintmath.cuh"
#include <cuda_runtime_api.h>
#include <cuda.h>

void special_fft_forward(DCKKSEncoderInfo *gpu_ckks_msg_vec_,
                         uint32_t msg_vec_size);

void special_fft_backward(DCKKSEncoderInfo *gpu_ckks_msg_vec_,
                          uint32_t msg_vec_size,
                          double scalar = 0.0);
#endif