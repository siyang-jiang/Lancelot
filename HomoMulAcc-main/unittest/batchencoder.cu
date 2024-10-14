#include "cahel.h"

#include <gtest/gtest.h>
using namespace std;
using namespace cahel;
namespace CAHELTest
{
    TEST(BatchEncoderTest, Constructor)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        ASSERT_NO_THROW(CAHELGPUBatchEncoder batchEncoder(context));
    }
    TEST(BatchEncoderTest, Encode)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUBatchEncoder batchEncoder(context);
        size_t slot_count = batchEncoder.slots_;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            pod_matrix[idx] = idx;
        }
        CAHELGPUPlaintext plain_matrix(context);
        ASSERT_NO_THROW(batchEncoder.encode(pod_matrix, plain_matrix));
    }
    TEST(BatchEncoderTest, Decode)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUBatchEncoder batchEncoder(context);
        size_t slot_count = batchEncoder.slots_;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            pod_matrix[idx] = idx;
        }
        CAHELGPUPlaintext plain_matrix(context);
        batchEncoder.encode(pod_matrix, plain_matrix);
        std::vector<int64_t> res;
        ASSERT_NO_THROW(batchEncoder.decode(plain_matrix, res));
        ASSERT_EQ(pod_matrix, res);
    }
}