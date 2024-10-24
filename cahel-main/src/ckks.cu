#include "ckks.h"

using namespace std;
using namespace cahel;
using namespace cahel::util;

__global__ void bit_reverse_and_zero_padding(cuDoubleComplex *dst, cuDoubleComplex *src, uint64_t in_size, uint32_t slots, uint32_t logn)
{
    for (uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < slots;
         tid += blockDim.x * gridDim.x)
    {
        if (tid < uint32_t(in_size))
        {
            dst[reverse_bits_uint32(tid, logn)] = src[tid];
        }
        else
        {
            dst[reverse_bits_uint32(tid, logn)] = (cuDoubleComplex){0.0, 0.0};
        }
    }
}

__global__ void bit_reverse(cuDoubleComplex *dst, cuDoubleComplex *src, uint32_t slots, uint32_t logn)
{
    for (uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < slots;
         tid += blockDim.x * gridDim.x)
    {
        dst[reverse_bits_uint32(tid, logn)] = src[tid];
    }
}

CAHELGPUCKKSEncoder::CAHELGPUCKKSEncoder(const CAHELGPUContext &context) // : context_(context)
{
    auto cpu_context_ptr = context.cpu_context_;
    auto &context_data = (CAHELContext::ContextData &)(cpu_context_ptr->get_context_data(first_chain_index_));
    auto &parms = (EncryptionParameters &)(context_data.parms());
    auto &coeff_modulus = parms.coeff_modulus();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    std::size_t coeff_count = parms.poly_modulus_degree();

#ifdef CAHEL_DEBUG_CKKS_ENCODER
    cout
        << "modulus====================" << endl;
    for (auto twr = 0; twr < coeff_modulus_size; twr++)
        cout << coeff_modulus[twr].value() << endl;
    cout << endl;
#endif

    if (parms.scheme() != scheme_type::ckks)
    {
        throw std::invalid_argument("unsupported scheme");
    }
    uint32_t logn = get_power_of_two(coeff_count);
    slots_ = coeff_count >> 1; // n/2
    uint32_t m = coeff_count << 1;
    uint32_t slots_half = slots_ >> 1;
    gpu_ckks_msg_vec_ = DCKKSEncoderInfo(coeff_count);

    // We need m powers of the primitive 2n-th root, m = 2n
    root_powers_.reserve(m);
    rotation_group_.reserve(slots_half);

    uint32_t gen = 5;
    uint32_t pos = 1; // Position in normal bit order
    for (size_t i = 0; i < slots_half; i++)
    {
        // Set the bit-reversed locations
        rotation_group_[i] = pos;

        // Next primitive root
        pos *= gen; // 5^i mod m
        pos &= (m - 1);
    }

    // Powers of the primitive 2n-th root have 4-fold symmetry
    if (m >= 8)
    {
        complex_roots_ = make_shared<util::ComplexRoots>(util::ComplexRoots(static_cast<size_t>(m)));
        for (size_t i = 0; i < m; i++)
        {
            root_powers_[i] = complex_roots_->get_root(i);
        }
    }
    else if (m == 4)
    {
        root_powers_[0] = {1, 0};
        root_powers_[1] = {0, 1};
        root_powers_[2] = {-1, 0};
        root_powers_[3] = {0, -1};
    }

    CUDA_CHECK(cudaMemcpy(gpu_ckks_msg_vec_.twiddle(), root_powers_.data(), m * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_ckks_msg_vec_.mul_group(), rotation_group_.data(), slots_half * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // CUDA_CHECK(cudaStreamAttachMemAsync(NULL, gpu_ckks_msg_vec_.twiddle(), 0, cudaMemAttachGlobal));
    // CUDA_CHECK(cudaStreamAttachMemAsync(NULL, gpu_ckks_msg_vec_.mul_group(), 0, cudaMemAttachGlobal));

    // Create gpu_plain_rns
    // auto &small_ntt_tables = cpu_context_ptr->get_context_data(0).small_ntt_tables();
}

void CAHELGPUCKKSEncoder::encode_internal(const CAHELGPUContext &context, const cuDoubleComplex *values, size_t values_size, size_t chain_index, double scale, CAHELGPUPlaintext &destination)
{
    auto cpu_context_ptr = context.cpu_context_;
    auto &context_data = (CAHELContext::ContextData &)(cpu_context_ptr->get_context_data(chain_index));
    auto &parms = (EncryptionParameters &)(context_data.parms());
    auto &coeff_modulus = parms.coeff_modulus();
    auto &rns_tool = context.gpu_rns_tool_vec()[chain_index];
    std::size_t coeff_modulus_size = coeff_modulus.size();
    std::size_t coeff_count = parms.poly_modulus_degree();

    if (!values && values_size > 0)
    {
        throw std::invalid_argument("values cannot be null");
    }
    if (values_size > slots_)
    {
        throw std::invalid_argument("values_size is too large");
    }

    // CUDA_CHECK(cudaMallocManaged((void **)&(destination.data_), coeff_count * coeff_modulus_size * sizeof(uint64_t)));

    // Check that scale is positive and not too large
    if (scale <= 0 || (static_cast<int>(log2(scale)) + 1 >= context_data.total_coeff_modulus_bit_count()))
    {
        throw std::invalid_argument("scale out of bounds");
    }

    if (sparse_slots_ == 0)
    {
        uint32_t log_sparse_slots = ceil(log2(values_size));
        sparse_slots_ = 1 << log_sparse_slots;
    }
    else
    {
        if (values_size > sparse_slots_)
        {
            throw std::invalid_argument("values_size exceeds previous message length");
        }
    }
    // size_t log_sparse_slots = ceil(log2(slots_));
    // sparse_slots_ = slots_;
    if (sparse_slots_ < 2)
    {
        throw std::invalid_argument("single value encoding is not available");
    }

    gpu_ckks_msg_vec_.set_sparse_slots(sparse_slots_);
    CUDA_CHECK(cudaMemset(gpu_ckks_msg_vec_.in(), 0, slots_ * sizeof(cuDoubleComplex)));
    Pointer<cuDoubleComplex> temp;
    temp.acquire(allocate<cuDoubleComplex>(Global(), values_size));
    CUDA_CHECK(cudaMemset(temp.get(), 0, values_size * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(temp.get(), values, sizeof(cuDoubleComplex) * values_size, cudaMemcpyHostToDevice));

    uint32_t log_sparse_n = log2(sparse_slots_);
    uint64_t gridDimGlb = ceil(sparse_slots_ / blockDimGlb.x);
    bit_reverse_and_zero_padding<<<gridDimGlb, blockDimGlb>>>(gpu_ckks_msg_vec_.in(), temp.get(), values_size, sparse_slots_, log_sparse_n);

    double fix = scale / static_cast<double>(sparse_slots_);

#ifdef CAHEL_DEBUG_CKKS_ENCODER
    printf("\n\n %lf, %d, \n\n", fix, sparse_slots_);
#endif

    special_fft_backward(&gpu_ckks_msg_vec_, (uint32_t)1, fix);
    // we calculate max_coeff_bit_count at cpu side
    // CUDA_CHECK(cudaStreamAttachMemAsync(NULL, gpu_ckks_msg_vec_.in(), 0, cudaMemAttachGlobal));

    // TODO to opt this
    vector<cuDoubleComplex> temp2(sparse_slots_);
    CUDA_CHECK(cudaMemcpy(temp2.data(), gpu_ckks_msg_vec_.in(), sparse_slots_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    double max_coeff = 0;
    for (std::size_t i = 0; i < sparse_slots_; i++)
    {
        max_coeff = std::max(max_coeff, std::fabs(temp2[i].x));
    }
    for (std::size_t i = 0; i < sparse_slots_; i++)
    {
        max_coeff = std::max(max_coeff, std::fabs(temp2[i].y));
    }
    // Verify that the values are not too large to fit in coeff_modulus
    // Note that we have an extra + 1 for the sign bit
    // Don't compute logarithmis of numbers less than 1
    int max_coeff_bit_count = static_cast<int>(std::ceil(std::log2(std::max(max_coeff, 1.0)))) + 1;

    if (max_coeff_bit_count >= context_data.total_coeff_modulus_bit_count())
    {
        throw std::invalid_argument("encoded values are too large");
    }
    // Resize destination to appropriate size
    // Need to first set parms_id to zero, otherwise resize
    // will throw an exception.
    destination.chain_index() = 0;
    // destination.resize(util::mul_safe(coeff_count, coeff_modulus_size));
    destination.resize(coeff_modulus_size, coeff_count);

    // we can in fact find all coeff_modulus in DNTTTable structure....
    rns_tool.base_Ql_.decompose_array(destination.data(), gpu_ckks_msg_vec_.in(), sparse_slots_ << 1, (uint32_t)slots_ / sparse_slots_, max_coeff_bit_count);
    // CUDA_CHECK(cudaStreamAttachMemAsync(NULL, destination.data(), 0, cudaMemAttachGlobal));

    nwt_2d_radix8_forward_inplace(destination.data(), context.gpu_rns_tables(), coeff_modulus_size, 0);

    destination.chain_index() = chain_index;
    destination.scale() = scale;
    destination.parms_id() = cpu_context_ptr->get_parms_id(chain_index);
}

void CAHELGPUCKKSEncoder::encode_internal(const CAHELGPUContext &context, double value, size_t chain_index, double scale, CAHELGPUPlaintext &destination)
{
    auto cpu_context_ptr = context.cpu_context_;
    auto &context_data = (CAHELContext::ContextData &)(cpu_context_ptr->get_context_data(chain_index));
    auto &parms = (EncryptionParameters &)(context_data.parms());
    auto &coeff_modulus = parms.coeff_modulus();
    auto &rns_tool = context.gpu_rns_tool_vec()[chain_index];
    std::size_t coeff_modulus_size = coeff_modulus.size();
    std::size_t coeff_count = parms.poly_modulus_degree();

    // CUDA_CHECK(cudaMallocManaged((void **)&(destination.data_), coeff_count * coeff_modulus_size * sizeof(uint64_t)));

    // Check that scale is positive and not too large
    if (scale <= 0 || (static_cast<int>(log2(scale)) + 1 >= context_data.total_coeff_modulus_bit_count()))
    {
        throw std::invalid_argument("scale out of bounds");
    }

    if (sparse_slots_ == 0)
    {
        sparse_slots_ = slots_;
    }

    // Compute the scaled value
    value *= scale;

    int coeff_bit_count = static_cast<int>(log2(fabs(value))) + 2;
    if (coeff_bit_count >= context_data.total_coeff_modulus_bit_count())
    {
        throw invalid_argument("encoded value is too large");
    }

    // Resize destination to appropriate size
    // Need to first set parms_id to zero, otherwise resize
    // will throw an exception.
    destination.chain_index() = 0;
    destination.resize(coeff_modulus_size, coeff_count);

    // decompose and fill
    rns_tool.base_Ql_.decompose(destination.data(), value, coeff_count, coeff_bit_count);

    destination.chain_index() = chain_index;
    destination.scale() = scale;
    destination.parms_id() = cpu_context_ptr->get_parms_id(chain_index);
}

void CAHELGPUCKKSEncoder::encode_internal(const CAHELGPUContext &context, int64_t value, size_t chain_index, CAHELGPUPlaintext &destination)
{
    auto cpu_context_ptr = context.cpu_context_;
    auto &context_data = (CAHELContext::ContextData &)(cpu_context_ptr->get_context_data(chain_index));
    auto &parms = (EncryptionParameters &)(context_data.parms());
    auto &coeff_modulus = parms.coeff_modulus();
    auto &rns_tool = context.gpu_rns_tool_vec()[chain_index];
    std::size_t coeff_modulus_size = coeff_modulus.size();
    std::size_t coeff_count = parms.poly_modulus_degree();

    if (sparse_slots_ == 0)
    {
        sparse_slots_ = slots_;
    }

    int coeff_bit_count = static_cast<int>(log2(fabs(value))) + 2;
    if (coeff_bit_count >= context_data.total_coeff_modulus_bit_count())
    {
        throw invalid_argument("encoded value is too large");
    }

    // Resize destination to appropriate size
    // Need to first set parms_id to zero, otherwise resize
    // will throw an exception.
    destination.chain_index() = 0;
    destination.resize(coeff_modulus_size, coeff_count);

    // decompose and fill
    rns_tool.base_Ql_.decompose(destination.data(), value, coeff_count, coeff_bit_count);

    destination.chain_index() = chain_index;
    destination.scale() = 1.0;
    destination.parms_id() = cpu_context_ptr->get_parms_id(chain_index);
}

void CAHELGPUCKKSEncoder::decode_internal(const CAHELGPUContext &context, const CAHELGPUPlaintext &plain, cuDoubleComplex *destination)
{
    if (!plain.is_ntt_form())
    {
        throw std::invalid_argument("plain is not in NTT form");
    }
    if (!destination)
    {
        throw std::invalid_argument("destination cannot be null");
    }

    auto cpu_context_ptr = context.cpu_context_;
    auto &context_data = (CAHELContext::ContextData &)(cpu_context_ptr->get_context_data(plain.chain_index_));
    auto &parms = (EncryptionParameters &)(context_data.parms());
    auto &coeff_modulus = parms.coeff_modulus();
    auto &rns_tool = context.gpu_rns_tool_vec()[plain.chain_index_];
    std::size_t coeff_modulus_size = coeff_modulus.size();
    std::size_t coeff_count = parms.poly_modulus_degree();
    std::size_t rns_poly_uint64_count = coeff_count * coeff_modulus_size;

    // cout << endl << "chain_index = " << plain.chain_index_ << endl;

    // CUDA_CHECK(cudaMallocManaged((void **)&(plain.data_), rns_poly_uint64_count * sizeof(uint64_t)));

    if (plain.scale() <= 0 ||
        (static_cast<int>(log2(plain.scale())) >= context_data.total_coeff_modulus_bit_count()))
    {
        throw std::invalid_argument("scale out of bounds");
    }

    auto decryption_modulus = context_data.total_coeff_modulus();
    auto upper_half_threshold = context_data.upper_half_threshold();
    int logn = util::get_power_of_two(coeff_count);
    Pointer<uint64_t> gpu_upper_half_threshold;
    gpu_upper_half_threshold.acquire(allocate<uint64_t>(Global(), upper_half_threshold.size()));
    CUDA_CHECK(cudaMemcpy(gpu_upper_half_threshold.get(), upper_half_threshold.data(), upper_half_threshold.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    gpu_ckks_msg_vec_.set_sparse_slots(sparse_slots_);
    CUDA_CHECK(cudaMemset(gpu_ckks_msg_vec_.in(), 0, slots_ * sizeof(cuDoubleComplex)));

    // Quick sanity check
    if ((logn < 0) || (coeff_count < CAHEL_POLY_MOD_DEGREE_MIN) || (coeff_count > CAHEL_POLY_MOD_DEGREE_MAX))
    {
        throw std::logic_error("invalid parameters");
    }

#ifdef CAHEL_DEBUG_CKKS_DECODER
    uint64_t *debug = (uint64_t *)malloc(coeff_modulus_size * coeff_count * sizeof(uint64_t));
    CUDA_CHECK(cudaMemcpy(debug, plain.data(), coeff_modulus_size * coeff_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    cout << endl;
    cout << endl;
    for (std::size_t i = 0; i < 8; i++)
    {
        if (i == 0)
        {
            for (std::size_t j = 0; j < coeff_modulus_size; j++)
                cout << context.gpu_rns_vec()[j].modulus().value() << ",  ";
            cout << endl;
            cout << "======== plain input =============" << endl;
        }

        for (std::size_t j = 0; j < coeff_modulus_size; j++)
        {

            cout << debug[j * coeff_count + i] << ",  ";
        }
        cout << endl;
    }

    cout << "====================================" << endl;
#endif

    double inv_scale = double(1.0) / plain.scale();
    // Create mutable copy of input
    Pointer<uint64_t> plain_copy;
    plain_copy.acquire(allocate<uint64_t>(Global(), rns_poly_uint64_count));
    CUDA_CHECK(cudaMemcpy(plain_copy.get(), plain.data(), rns_poly_uint64_count * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

#ifdef CAHEL_DEBUG_CKKS_DECODER
    CUDA_CHECK(cudaMemcpy(debug, plain_copy.get(), coeff_modulus_size * coeff_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    cout << endl;
    cout << endl;
    for (std::size_t i = 0; i < 8; i++)
    {
        if (i == 0)
        {
            for (std::size_t j = 0; j < coeff_modulus_size; j++)
                cout << context.gpu_rns_vec()[j].modulus().value() << ",  ";
            cout << endl;
            cout << "======== plain copy =============" << endl;
        }

        for (std::size_t j = 0; j < coeff_modulus_size; j++)
        {

            cout << debug[j * coeff_count + i] << ",  ";
        }
        cout << endl;
    }
    cout << "====================================" << endl;
#endif

    nwt_2d_radix8_backward_inplace(plain_copy.get(), context.gpu_rns_tables(), coeff_modulus_size, 0);

#ifdef CAHEL_DEBUG_CKKS_DECODER
    CUDA_CHECK(cudaMemcpy(debug, plain_copy.get(), coeff_modulus_size * coeff_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    cout << endl;
    cout << endl;
    for (std::size_t i = 0; i < 8; i++)
    {
        if (i == 0)
        {
            for (std::size_t j = 0; j < coeff_modulus_size; j++)
                cout << context.gpu_rns_vec()[j].modulus().value() << ",  ";
            cout << endl;
            cout << "================== plain after intt =======================" << endl;
        }

        for (std::size_t j = 0; j < coeff_modulus_size; j++)
        {

            cout << debug[j * coeff_count + i] << ",  ";
        }
        cout << endl;
    }
    cout << "====================================" << endl;
#endif

    // CRT-compose the polynomial
    rns_tool.base_Ql_.compose_array(gpu_ckks_msg_vec().in(), plain_copy.get(), gpu_upper_half_threshold.get(), inv_scale, coeff_count, sparse_slots_ << 1, slots_ / sparse_slots_);

#ifdef CAHEL_DEBUG_CKKS_DECODER
    cuDoubleComplex *debug_cplx = (cuDoubleComplex *)malloc(coeff_count * sizeof(cuDoubleComplex));
    CUDA_CHECK(cudaMemcpy(debug_cplx, gpu_ckks_msg_vec().in(), coeff_count * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    cout << endl
         << "compose_array result" << endl;
    cout << gpu_ckks_msg_vec().m() << endl;
    cout << gpu_ckks_msg_vec().sparse_slots() << endl;
    for (size_t i = 0; i < 8; i++)
    {
        printf("%lu: %lf + i %lf\n", i, debug_cplx[i].x, debug_cplx[i].y);
    }
    cout
        << "====================================" << endl;
    cout << endl
         << endl;

#endif

    special_fft_forward(&gpu_ckks_msg_vec_, (uint32_t)1);
    // CUDA_CHECK(cudaStreamAttachMemAsync(NULL, gpu_ckks_msg_vec_.in(), 0, cudaMemAttachGlobal));

    // finally, bit-reverse and output
    Pointer<cuDoubleComplex> out;
    out.acquire(allocate<cuDoubleComplex>(Global(), sparse_slots_));
    uint32_t log_sparse_n = log2(sparse_slots_);
    uint64_t gridDimGlb = ceil(sparse_slots_ / blockDimGlb.x);
    bit_reverse<<<gridDimGlb, blockDimGlb>>>(out.get(), gpu_ckks_msg_vec_.in(), sparse_slots_, log_sparse_n);
    CUDA_CHECK(cudaMemcpy(destination, out.get(), sparse_slots_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

#ifdef CAHEL_DEBUG_CKKS_DECODER
    cout << endl;
    for (size_t i = 0; i < 8; i++)
    {
        printf("%lu: %lf + i %lf\n", i, destination[i].x, destination[i].y);
    }
    cout << "......" << endl;
    cout
        << "====================================" << endl;
    cout << endl
         << endl;
#endif
}