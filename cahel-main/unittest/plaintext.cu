#include "cahel.h"
#include <gtest/gtest.h>
using namespace std;
using namespace cahel;
namespace CAHELTest
{
    TEST(PlaintextTest, ConstructorBFV)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        ASSERT_NO_THROW(CAHELGPUPlaintext ptxt(context));
    }
    TEST(PlaintextTest, ConstructorCKKS)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        double scale = pow(2.0, 40);
        CAHELGPUContext context(parms);
        ASSERT_NO_THROW(CAHELGPUPlaintext plain_matrix(context));
    }
    TEST(PlaintextTest, CopyBFV)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUBatchEncoder batchEncoder(context);
        size_t slot_count = batchEncoder.slots_;
        size_t row_size = slot_count / 2;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        pod_matrix[0] = 0ULL;
        pod_matrix[1] = 1ULL;
        pod_matrix[2] = 2ULL;
        pod_matrix[3] = 3ULL;
        pod_matrix[row_size] = 4ULL;
        pod_matrix[row_size + 1] = 5ULL;
        pod_matrix[row_size + 2] = 6ULL;
        pod_matrix[row_size + 3] = 7ULL;
        CAHELGPUPlaintext plain_matrix(context);
        batchEncoder.encode(pod_matrix, plain_matrix);
        ASSERT_NO_THROW(CAHELGPUPlaintext ptxt_copy(plain_matrix));
        CAHELGPUPlaintext ptxt_copy(plain_matrix);
        std::vector<int64_t> res;
        batchEncoder.decode(ptxt_copy, res);
        for (int i = 0; i < pod_matrix.size(); i++)
        {
            ASSERT_EQ(pod_matrix[i], res[i]);
        }
        ASSERT_NO_THROW(CAHELGPUPlaintext ptxt_op = plain_matrix);
        CAHELGPUPlaintext ptxt_op = plain_matrix;
        batchEncoder.decode(ptxt_op, res);
        for (int i = 0; i < pod_matrix.size(); i++)
        {
             ASSERT_EQ(pod_matrix[i], res[i]);
        }
    }
    TEST(PlaintextTest, CopyCKKS)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        double scale = pow(2.0, 40);
        CAHELGPUContext context(parms);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUCKKSEncoder encoder(context);
        size_t slot_count = encoder.slots_;
        vector<double> input;
        double rand_real;
        size_t size = slot_count;
        input.reserve(size);
        for (size_t i = 0; i < size; i++)
        {
            rand_real = (double)rand() / RAND_MAX;
            input.push_back(rand_real);
        }
        encoder.encode(context, input, scale, plain_matrix);
        ASSERT_NO_THROW(CAHELGPUPlaintext plain_copy(plain_matrix));
        CAHELGPUPlaintext plain_copy(plain_matrix);
        vector<double> re;
        encoder.decode(context, plain_copy, re);
        for (int i = 0; i < input.size(); i++)
        {
           ASSERT_NEAR(input[i], re[i], 0.00001);
        }
        ASSERT_NO_THROW(CAHELGPUPlaintext plain_op = plain_matrix);
        CAHELGPUPlaintext plain_op = plain_matrix;
        encoder.decode(context, plain_op, re);
        for (int i = 0; i < input.size(); i++)
        {
             ASSERT_NEAR(input[i], re[i], 0.00001);
        }
    }
    TEST(PlaintextTest, MoveBFV)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUBatchEncoder batchEncoder(context);
        size_t slot_count = batchEncoder.slots_;
        size_t row_size = slot_count / 2;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        pod_matrix[0] = 0ULL;
        pod_matrix[1] = 1ULL;
        pod_matrix[2] = 2ULL;
        pod_matrix[3] = 3ULL;
        pod_matrix[row_size] = 4ULL;
        pod_matrix[row_size + 1] = 5ULL;
        pod_matrix[row_size + 2] = 6ULL;
        pod_matrix[row_size + 3] = 7ULL;
        CAHELGPUPlaintext plain_matrix(context);
        batchEncoder.encode(pod_matrix, plain_matrix);
//        ASSERT_NO_THROW(CAHELGPUPlaintext ptxt_copy(move(plain_matrix)));
        CAHELGPUPlaintext ptxt_copy(move(plain_matrix));
        std::vector<int64_t> res;
        batchEncoder.decode(ptxt_copy, res);
        for (int i = 0; i < pod_matrix.size(); i++)
        {
            ASSERT_EQ(pod_matrix[i], res[i]);
        }
    }
    TEST(PlaintextTest, MoveCKKS)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        double scale = pow(2.0, 40);
        CAHELGPUContext context(parms);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUCKKSEncoder encoder(context);
        size_t slot_count = encoder.slots_;
        vector<double> input;
        double rand_real;
        size_t size = slot_count;
        input.reserve(size);
        for (size_t i = 0; i < size; i++)
        {
            rand_real = (double)rand() / RAND_MAX;
            input.push_back(rand_real);
        }
        encoder.encode(context, input, scale, plain_matrix);
        CAHELGPUPlaintext plain_copy(move(plain_matrix));
        vector<double> re;
        encoder.decode(context, plain_copy, re);
        for (int i = 0; i < input.size(); i++)
        {
            ASSERT_NEAR(input[i], re[i], 0.00001);
        }
    }
    TEST(PlaintextTest, SaveLoadBFV)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUBatchEncoder batchEncoder(context);
        size_t slot_count = batchEncoder.slots_;
        size_t row_size = slot_count / 2;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        pod_matrix[0] = 0ULL;
        pod_matrix[1] = 1ULL;
        pod_matrix[2] = 2ULL;
        pod_matrix[3] = 3ULL;
        pod_matrix[row_size] = 4ULL;
        pod_matrix[row_size + 1] = 5ULL;
        pod_matrix[row_size + 2] = 6ULL;
        pod_matrix[row_size + 3] = 7ULL;
        CAHELGPUPlaintext plain_matrix(context);
        batchEncoder.encode(pod_matrix, plain_matrix);
        std::stringstream ptxt_stream;
        plain_matrix.save(ptxt_stream);
        CAHELGPUPlaintext ptxt_load(context);
        ptxt_load.load(context, ptxt_stream);
        std::vector<int64_t> res;
        batchEncoder.decode(ptxt_load, res);
        for (int i = 0; i < pod_matrix.size(); i++)
        {
            ASSERT_EQ(pod_matrix[i], res[i]);
        }
    }
    TEST(PlaintextTest, SaveLoadCKKS)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        double scale = pow(2.0, 40);
        CAHELGPUContext context(parms);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUCKKSEncoder encoder(context);
        size_t slot_count = encoder.slots_;
        vector<double> input;
        double rand_real;
        size_t size = slot_count;
        input.reserve(size);
        for (size_t i = 0; i < size; i++)
        {
            rand_real = (double)rand() / RAND_MAX;
            input.push_back(rand_real);
        }
        encoder.encode(context, input, scale, plain_matrix);
        std::stringstream ptxt_stream;
        plain_matrix.save(ptxt_stream);
        CAHELGPUPlaintext ptxt_load(context);
        ptxt_load.load(context, ptxt_stream);
        vector<double> re;
        encoder.decode(context, ptxt_load, re);
        for (int i = 0; i < input.size(); i++)
        {
            ASSERT_NEAR(input[i], re[i], 0.00001);
        }
    }
}