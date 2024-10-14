#include "cahel.h"
#include <gtest/gtest.h>
using namespace std;
using namespace cahel;
namespace CAHELTest
{
    /*TEST(EvaluatorTest, NegateCKKS)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        double scale = pow(2.0, 40);
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCKKSEncoder encoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext dec_plain(context);
        std::vector<double> dec_res;
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
        secret_key.gen_secretkey(context);
        encoder.encode(context, input, scale, plain_matrix);
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext destination(context);
        negate_inplace(context, sym_cipher);
        secret_key.decrypt(context, sym_cipher, dec_plain);
        encoder.decode(context, dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count / 2; idx++)
        {
            ASSERT_NEAR(input[idx] * (-1), dec_res[idx], 0.00001);
        }
    }*/
    TEST(EvaluatorTest, AddCKKS)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        double scale = pow(2.0, 40);
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUCKKSEncoder encoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext dec_plain(context);
        std::vector<double> dec_res;
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
        secret_key.gen_secretkey(context);
        encoder.encode(context, input, scale, plain_matrix);
        // sym version
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext destination(context);
        add(context, sym_cipher, sym_cipher, destination);
        secret_key.decrypt(context, destination, dec_plain);
        encoder.decode(context, dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count / 2; idx++)
        {
            ASSERT_NEAR(input[idx] * 2, dec_res[idx], 0.00001);
        }
        // asym version
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
        add(context, asym_cipher, asym_cipher, destination);
        secret_key.decrypt(context, destination, dec_plain);
        encoder.decode(context, dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count / 2; idx++)
        {
            ASSERT_NEAR(input[idx] * 2, dec_res[idx], 0.00001);
        }
    }
    TEST(EvaluatorTest, AddBFV)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        auto pt = PlainModulus::Batching(poly_modulus_degree, 20);
        parms.set_plain_modulus(pt);
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUBatchEncoder batchEncoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext dec_plain(context);
        std::vector<int64_t> dec_res;
        size_t slot_count = batchEncoder.slots_;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            pod_matrix[idx] = idx;
        }
        secret_key.gen_secretkey(context);
        batchEncoder.encode(pod_matrix, plain_matrix);
        // sym version
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext destination(context);
        add(context, sym_cipher, sym_cipher, destination);
        secret_key.decrypt(context, destination, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count / 2; idx++)
        {
            ASSERT_EQ(pod_matrix[idx] * 2, dec_res[idx]);
        }
        // asym version
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
        add(context, asym_cipher, asym_cipher, destination);
        secret_key.decrypt(context, destination, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count / 2; idx++)
        {
            ASSERT_EQ(pod_matrix[idx] * 2, dec_res[idx]);
        }
    }
    TEST(EvaluatorTest, AddManyBFV)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUBatchEncoder batchEncoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext dec_plain(context);
        std::vector<int64_t> dec_res;
        size_t slot_count = batchEncoder.slots_;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            pod_matrix[idx] = idx;
        }
        secret_key.gen_secretkey(context);
        batchEncoder.encode(pod_matrix, plain_matrix);
        // sym version
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext destination(context);
        vector<CAHELGPUCiphertext> some_cipher;
        for (int i = 0; i < 10; i++)
        {
            some_cipher.push_back(sym_cipher);
        }
        add_many(context, some_cipher, destination);
        secret_key.decrypt(context, destination, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_EQ(pod_matrix[idx] * 10, dec_res[idx]);
        }
        // asym version
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
        vector<CAHELGPUCiphertext> some_cipher_asy;
        for (int i = 0; i < 10; i++)
        {
            some_cipher_asy.push_back(asym_cipher);
        }
        add_many(context, some_cipher_asy, destination);
        secret_key.decrypt(context, destination, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_EQ(pod_matrix[idx] * 10, dec_res[idx]);
        }
    }
    TEST(EvaluatorTest, AddManyCKKS)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
        double scale = pow(2.0, 40);
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUCKKSEncoder encoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext dec_plain(context);
        std::vector<double> dec_res;
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
        secret_key.gen_secretkey(context);
        encoder.encode(context, input, scale, plain_matrix);
        // sym version
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext destination(context);
        vector<CAHELGPUCiphertext> some_cipher;
        for (int i = 0; i < 10; i++)
        {
            some_cipher.push_back(sym_cipher);
        }
        add_many(context, some_cipher, destination);
        secret_key.decrypt(context, destination, dec_plain);
        encoder.decode(context, dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_NEAR(input[idx] * 10, dec_res[idx], 0.00001);
        }
        // asym version
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
        vector<CAHELGPUCiphertext> some_cipher_asy;
        for (int i = 0; i < 10; i++)
        {
            some_cipher_asy.push_back(asym_cipher);
        }
        CAHELGPUCiphertext asy_destination(context);
        add_many(context, some_cipher_asy, asy_destination);
        secret_key.decrypt(context, asy_destination, dec_plain);
        encoder.decode(context, dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_NEAR(input[idx] * 10, dec_res[idx], 0.00001);
        }
    }
    TEST(EvaluatorTest, SubBFV)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUBatchEncoder batchEncoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext one(context);
        CAHELGPUPlaintext dec_plain(context);
        std::vector<int64_t> dec_res;
        size_t slot_count = batchEncoder.slots_;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        std::vector<int64_t> inp(slot_count, 1ULL);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            pod_matrix[idx] = idx + 2;
        }
        secret_key.gen_secretkey(context);
        batchEncoder.encode(pod_matrix, plain_matrix);
        batchEncoder.encode(inp, one);
        // sym version
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext sym_cipher_one(context);
        secret_key.encrypt_symmetric(context, one, sym_cipher_one, false);
        CAHELGPUCiphertext destination(context);
        sub(context, sym_cipher, sym_cipher_one, destination);
        secret_key.decrypt(context, destination, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_EQ(pod_matrix[idx] - 1, dec_res[idx]);
        }
        // asym version
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
        CAHELGPUCiphertext asym_cipher_one(context);
        public_key.encrypt_asymmetric(context, one, asym_cipher_one, false);
        sub(context, asym_cipher, asym_cipher_one, destination);
        secret_key.decrypt(context, destination, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_EQ(pod_matrix[idx] - 1, dec_res[idx]);
        }
    }
    TEST(EvaluatorTest, SubCKKS)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        double scale = pow(2.0, 40);
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCiphertext sym_one(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUCiphertext asym_one(context);
        CAHELGPUCKKSEncoder encoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext dec_plain(context);
        CAHELGPUPlaintext one(context);
        std::vector<double> dec_res;
        size_t slot_count = encoder.slots_;
        vector<double> input;
        vector<double> inp;
        double rand_real;
        size_t size = slot_count;
        input.reserve(size);
        for (size_t i = 0; i < size; i++)
        {
            rand_real = (double)rand() / RAND_MAX;
            input.push_back(rand_real);
            inp.push_back(1.1);
        }
        secret_key.gen_secretkey(context);
        encoder.encode(context, input, scale, plain_matrix);
        encoder.encode(context, inp, scale, one);
        // sym version
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        secret_key.encrypt_symmetric(context, one, sym_one, false);
        CAHELGPUCiphertext destination(context);
        sub(context, sym_cipher, sym_one, destination);
        secret_key.decrypt(context, destination, dec_plain);
        encoder.decode(context, dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_NEAR(input[idx]-1.1, dec_res[idx], 0.00001);
        }
        // asym version
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
        public_key.encrypt_asymmetric(context, one, asym_one, false);
        CAHELGPUCiphertext asy_destination(context);
        sub(context, asym_cipher, asym_one, asy_destination);
        secret_key.decrypt(context, asy_destination, dec_plain);
        encoder.decode(context, dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_NEAR(input[idx]-1.1, dec_res[idx], 0.00001);
        }
    }
    TEST(EvaluatorTest, MultiplyBFV)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUBatchEncoder batchEncoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext dec_plain(context);
        CAHELGPURelinKey relin_keys(context);
        std::vector<int64_t> dec_res;
        size_t slot_count = batchEncoder.slots_;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            pod_matrix[idx] = idx;
        }
        secret_key.gen_secretkey(context);
        secret_key.gen_relinkey(context, relin_keys);
        CAHELGPURelinKey relin_keys_op(relin_keys);
        batchEncoder.encode(pod_matrix, plain_matrix);
        // sym version
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext sym_cipher_copy(context);

        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher_copy, false);
        CAHELGPUCiphertext destination(context);
        multiply(context, sym_cipher, sym_cipher_copy, destination);
        relinearize_inplace(context, destination, relin_keys_op);
        secret_key.decrypt(context, destination, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < 20; idx++)
        {
            ASSERT_EQ(pod_matrix[idx] * pod_matrix[idx], dec_res[idx]);
        }
        // asym version
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
        CAHELGPUCiphertext destination_asy(context);
        CAHELGPUCiphertext asym_cipher_copy(asym_cipher);
        multiply(context, asym_cipher, asym_cipher_copy, destination_asy);
        relinearize_inplace(context, destination_asy, relin_keys);
        secret_key.decrypt(context, destination_asy, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < 20; idx++)
        {
            ASSERT_EQ(pod_matrix[idx] * pod_matrix[idx], dec_res[idx]);
        }
    }
    TEST(EvaluatorTest, MultiplyCKKS)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
        double scale = pow(2.0, 40);
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUCKKSEncoder encoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext dec_plain(context);
        CAHELGPURelinKey relin_keys(context);
        std::vector<double> dec_res;
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
        secret_key.gen_secretkey(context);
        secret_key.gen_relinkey(context, relin_keys);
        encoder.encode(context, input, scale, plain_matrix);
        // sym version
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext destination(context);
        multiply(context, sym_cipher, sym_cipher, destination);
        relinearize_inplace(context, destination, relin_keys);
        secret_key.decrypt(context, destination, dec_plain);
        encoder.decode(context, dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_NEAR(input[idx] * input[idx], dec_res[idx], 0.00001);
        }
        // asym version
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
        CAHELGPUCiphertext asy_destination(context);
        multiply(context, asym_cipher, asym_cipher, asy_destination);
        relinearize_inplace(context, asy_destination, relin_keys);
        secret_key.decrypt(context, asy_destination, dec_plain);
        encoder.decode(context, dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_NEAR(input[idx] * input[idx], dec_res[idx], 0.00001);
        }
    }
    TEST(EvaluatorTest, SquareBFV)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUBatchEncoder batchEncoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext dec_plain(context);
        CAHELGPURelinKey relin_keys(context);
        std::vector<int64_t> dec_res;
        size_t slot_count = batchEncoder.slots_;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            pod_matrix[idx] = idx;
        }
        secret_key.gen_secretkey(context);
        secret_key.gen_relinkey(context, relin_keys);
        batchEncoder.encode(pod_matrix, plain_matrix);
        // sym version
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext destination(context);
        square(context, sym_cipher, destination);
        relinearize_inplace(context, destination, relin_keys);
        secret_key.decrypt(context, destination, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < 20; idx++)
        {
            ASSERT_EQ(pod_matrix[idx] * pod_matrix[idx], dec_res[idx]);
        }
        // asym version
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
        CAHELGPUCiphertext destination_asy(context);
        square(context, asym_cipher, destination_asy);
        relinearize_inplace(context, destination_asy, relin_keys);
        secret_key.decrypt(context, destination_asy, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < 20; idx++)
        {
            ASSERT_EQ(pod_matrix[idx] * pod_matrix[idx], dec_res[idx]);
        }
    }
    TEST(EvaluatorTest, SquareCKKS)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
        double scale = pow(2.0, 40);
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUCKKSEncoder encoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext dec_plain(context);
        CAHELGPURelinKey relin_keys(context);
        std::vector<double> dec_res;
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
        secret_key.gen_secretkey(context);
        secret_key.gen_relinkey(context, relin_keys);
        encoder.encode(context, input, scale, plain_matrix);
        // sym version
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext destination(context);
        square(context, sym_cipher, destination);
        relinearize_inplace(context, destination, relin_keys);
        secret_key.decrypt(context, destination, dec_plain);
        encoder.decode(context, dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_NEAR(input[idx] * input[idx], dec_res[idx], 0.00001);
        }
        // asym version
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
        CAHELGPUCiphertext asy_destination(context);
        square(context, asym_cipher, asy_destination);
        relinearize_inplace(context, asy_destination, relin_keys);
        secret_key.decrypt(context, asy_destination, dec_plain);
        encoder.decode(context, dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_NEAR(input[idx] * input[idx], dec_res[idx], 0.00001);
        }
    }
    TEST(EvaluatorTest, AddPlainBFV)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUBatchEncoder batchEncoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext dec_plain(context);
        std::vector<int64_t> dec_res;
        size_t slot_count = batchEncoder.slots_;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            pod_matrix[idx] = idx;
        }
        secret_key.gen_secretkey(context);
        batchEncoder.encode(pod_matrix, plain_matrix);
        // sym version
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext destination(context);
        add_plain(context, sym_cipher, plain_matrix, destination);
        secret_key.decrypt(context, destination, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < 20; idx++)
        {
            ASSERT_EQ(pod_matrix[idx] + pod_matrix[idx], dec_res[idx]);
        }
        // asym version
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
        CAHELGPUCiphertext destination_asy(context);
        add_plain(context, asym_cipher, plain_matrix, destination_asy);
        secret_key.decrypt(context, destination_asy, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < 20; idx++)
        {
            ASSERT_EQ(pod_matrix[idx] + pod_matrix[idx], dec_res[idx]);
        }
    }
    TEST(EvaluatorTest, AddPlainCKKS)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        double scale = pow(2.0, 40);
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUCKKSEncoder encoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext dec_plain(context);
        CAHELGPURelinKey relin_keys(context);
        std::vector<double> dec_res;
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
        // sym version
        secret_key.gen_secretkey(context);
        encoder.encode(context, input, scale, plain_matrix);
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext destination(context);
        add_plain(context, sym_cipher, plain_matrix, destination);
        secret_key.decrypt(context, destination, dec_plain);
        encoder.decode(context, dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_NEAR(input[idx] + input[idx], dec_res[idx], 0.00001);
        }
        // asym version
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
        CAHELGPUCiphertext asy_destination(context);
        add_plain(context, asym_cipher, plain_matrix, asy_destination);
        secret_key.decrypt(context, asy_destination, dec_plain);
        encoder.decode(context, dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_NEAR(input[idx] + input[idx], dec_res[idx], 0.00001);
        }
    }
    TEST(EvaluatorTest, SubPlainBFV)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUBatchEncoder batchEncoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext dec_plain(context);
        std::vector<int64_t> dec_res;
        size_t slot_count = batchEncoder.slots_;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            pod_matrix[idx] = idx;
        }
        secret_key.gen_secretkey(context);
        batchEncoder.encode(pod_matrix, plain_matrix);
        // sym version
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext destination(context);
        sub_plain(context, sym_cipher, plain_matrix, destination);
        secret_key.decrypt(context, destination, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_EQ(0, dec_res[idx]);
        }
        // asym version
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
        CAHELGPUCiphertext destination_asy(context);
        sub_plain(context, sym_cipher, plain_matrix, destination_asy);
        secret_key.decrypt(context, destination_asy, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_EQ(0, dec_res[idx]);
        }
    }
    TEST(EvaluatorTest, SubPlainCKKS)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        double scale = pow(2.0, 40);
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUCKKSEncoder encoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext dec_plain(context);
        CAHELGPURelinKey relin_keys(context);
        std::vector<double> dec_res;
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
        secret_key.gen_secretkey(context);
        encoder.encode(context, input, scale, plain_matrix);
        // sym version
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext destination(context);
        sub_plain(context, sym_cipher, plain_matrix, destination);
        secret_key.decrypt(context, destination, dec_plain);
        encoder.decode(context, dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_NEAR(0, dec_res[idx], 0.00001);
        }
        // asym version
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
        CAHELGPUCiphertext asy_destination(context);
        sub_plain(context, asym_cipher, plain_matrix, asy_destination);
        secret_key.decrypt(context, asy_destination, dec_plain);
        encoder.decode(context, dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_NEAR(0, dec_res[idx], 0.00001);
        }
    }
    TEST(EvaluatorTest, MultiplyPlainBFV)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUBatchEncoder batchEncoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext dec_plain(context);
        std::vector<int64_t> dec_res;
        size_t slot_count = batchEncoder.slots_;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            pod_matrix[idx] = idx;
        }
        secret_key.gen_secretkey(context);
        batchEncoder.encode(pod_matrix, plain_matrix);
        // sym version
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext destination(context);
        multiply_plain(context, sym_cipher, plain_matrix, destination);
        secret_key.decrypt(context, destination, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < 20; idx++)
        {
            ASSERT_EQ(pod_matrix[idx] * pod_matrix[idx], dec_res[idx]);
        }
        // asym version
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
        CAHELGPUCiphertext destination_asy(context);
        multiply_plain(context, asym_cipher, plain_matrix, destination_asy);
        secret_key.decrypt(context, destination_asy, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < 20; idx++)
        {
            ASSERT_EQ(pod_matrix[idx] * pod_matrix[idx], dec_res[idx]);
        }
    }
    TEST(EvaluatorTest, MultiplyPlainCKKS)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
        double scale = pow(2.0, 40);
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUCKKSEncoder encoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext dec_plain(context);
        CAHELGPURelinKey relin_keys(context);
        std::vector<double> dec_res;
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
        secret_key.gen_secretkey(context);
        encoder.encode(context, input, scale, plain_matrix);
        // sym version
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext destination(context);
        multiply_plain(context, sym_cipher, plain_matrix, destination);
        secret_key.decrypt(context, destination, dec_plain);
        encoder.decode(context, dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_NEAR(input[idx] * input[idx], dec_res[idx], 0.00001);
        }
        // asym version
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
        CAHELGPUCiphertext asy_destination(context);
        multiply_plain(context, asym_cipher, plain_matrix, asy_destination);
        secret_key.decrypt(context, asy_destination, dec_plain);
        encoder.decode(context, dec_plain, dec_res);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            ASSERT_NEAR(input[idx] * input[idx], dec_res[idx], 0.00001);
        }
    }
    TEST(EvaluatorTest, MultiplyManyBFV)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUBatchEncoder batchEncoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext dec_plain(context);
        CAHELGPURelinKey relin_keys(context);
        std::vector<int64_t> dec_res;
        size_t slot_count = batchEncoder.slots_;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            pod_matrix[idx] = idx;
        }
        secret_key.gen_secretkey(context);
        secret_key.gen_relinkey(context, relin_keys);
        batchEncoder.encode(pod_matrix, plain_matrix);
        // sym version
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext destination(context);
        vector<CAHELGPUCiphertext> some_cipher;
        for (int i = 0; i < 3; i++)
        {
            some_cipher.push_back(sym_cipher);
        }
        multiply_many(context, some_cipher, relin_keys, destination);
        secret_key.decrypt(context, destination, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < 20; idx++)
        {
            ASSERT_EQ(pod_matrix[idx] * pod_matrix[idx] * pod_matrix[idx], dec_res[idx]);
        }
        // asym version
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
        CAHELGPUCiphertext destination_asy(context);
        vector<CAHELGPUCiphertext> some_cipher_asy;
        for (int i = 0; i < 3; i++)
        {
            some_cipher_asy.push_back(asym_cipher);
        }
        multiply_many(context, some_cipher_asy, relin_keys, destination_asy);
        secret_key.decrypt(context, destination_asy, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < 20; idx++)
        {
            ASSERT_EQ(pod_matrix[idx] * pod_matrix[idx] * pod_matrix[idx], dec_res[idx]);
        }
    }

    TEST(EvaluatorTest, ExponentiateBFV)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        auto plainModulus = PlainModulus::Batching(poly_modulus_degree, 20);
        parms.set_plain_modulus(plainModulus);
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUBatchEncoder batchEncoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUPlaintext dec_plain(context);
        CAHELGPURelinKey relin_keys(context);
        std::vector<int64_t> dec_res;
        size_t slot_count = batchEncoder.slots_;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            pod_matrix[idx] = idx;
        }
        secret_key.gen_secretkey(context);
        secret_key.gen_relinkey(context, relin_keys);
        batchEncoder.encode(pod_matrix, plain_matrix);
        // sym version
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext destination(context);
        uint64_t num = 3;
        exponentiate_inplace(context, sym_cipher, num, relin_keys);
        secret_key.decrypt(context, sym_cipher, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < 20; idx++)
        {
            ASSERT_EQ((idx * idx * idx) % plainModulus.value(), dec_res[idx]) << "  " << plainModulus.value();
        }
        // asym version
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
        num = 3;
        exponentiate_inplace(context, asym_cipher, num, relin_keys);
        secret_key.decrypt(context, asym_cipher, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < 20; idx++)
        {
            ASSERT_EQ((idx * idx * idx) % plainModulus.value(), dec_res[idx]) << "  " << plainModulus.value();
        }
    }
}
