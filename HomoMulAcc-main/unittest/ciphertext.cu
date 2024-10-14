#include "cahel.h"
#include <gtest/gtest.h>
using namespace std;
using namespace cahel;
namespace CAHELTest
{
    TEST(BFVCipherTest, Constructor)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        ASSERT_NO_THROW(CAHELGPUCiphertext cipher(context));
    }
    TEST(BFVCipherTest, Copy)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUCiphertext sym_cipher(context);
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
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        ASSERT_NO_THROW(CAHELGPUCiphertext cipher_copy(sym_cipher));
        CAHELGPUCiphertext cipher_copy(sym_cipher);
        ASSERT_EQ(sym_cipher == cipher_copy, 1);
        ASSERT_NO_THROW(CAHELGPUCiphertext cipher_copy_op = sym_cipher);
        CAHELGPUCiphertext cipher_copy_op = sym_cipher;
        ASSERT_EQ(sym_cipher == cipher_copy_op, 1);
    }
    TEST(BFVCipherTest, SaveLoad)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUCiphertext sym_cipher(context);
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
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        std::stringstream cipher_stream;
        sym_cipher.save(cipher_stream);
        CAHELGPUCiphertext cipher_load(context);
        cipher_load.load(context, cipher_stream);
        ASSERT_EQ(sym_cipher == cipher_load, 1);
    }

    TEST(CKKSCipherTest, Constructor)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        double scale = pow(2.0, 40);
        CAHELGPUContext context(parms);
        ASSERT_NO_THROW(CAHELGPUCiphertext cipher(context));
    }
    TEST(CKKSCipherTest, Copy)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        switch (poly_modulus_degree)
        {
        case 4096:
            parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
            break;
        case 8192:
            parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
            break;
        case 16384:
            parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 60}));
            break;
        case 32768:
            parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
                                                         {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));
            break;
        default:
            throw std::invalid_argument("unsupported polynomial degree");
            return;
        }
        double scale = pow(2.0, 40);
        CAHELGPUContext context(parms);
        CAHELGPUCKKSEncoder encoder(context);
        size_t slot_count = encoder.slot_count();
        vector<double> input;
        size_t msg_size = slot_count;
        input.reserve(msg_size);
        double rand_real;
        srand(time(0));
        for (size_t i = 0; i < msg_size; i++)
        {
            rand_real = (double)rand() / RAND_MAX;
            input.push_back(rand_real);
        }
        CAHELGPUCiphertext cipher(context);
        CAHELGPUPlaintext x_plain(context);
        encoder.encode(context, input, scale, x_plain);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        secret_key.gen_secretkey(context);
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, x_plain, cipher, false);
        CAHELGPUCiphertext cipher_copy(cipher);
        ASSERT_EQ(cipher == cipher_copy, 1);
        CAHELGPUCiphertext cipher_copy_op = cipher;
        ASSERT_EQ(cipher == cipher_copy_op, 1);

    }
    TEST(CKKSCipherTest, Move)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        switch (poly_modulus_degree)
        {
        case 4096:
            parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
            break;
        case 8192:
            parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
            break;
        case 16384:
            parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 60}));
            break;
        case 32768:
            parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
                                                         {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));
            break;
        default:
            throw std::invalid_argument("unsupported polynomial degree");
            return;
        }
        double scale = pow(2.0, 40);
        CAHELGPUContext context(parms);
        CAHELGPUCKKSEncoder encoder(context);
        size_t slot_count = encoder.slot_count();
        vector<double> input;
        size_t msg_size = slot_count;
        input.reserve(msg_size);
        double rand_real;
        srand(time(0));
        for (size_t i = 0; i < msg_size; i++)
        {
            rand_real = (double)rand() / RAND_MAX;
            input.push_back(rand_real);
        }
        CAHELGPUCiphertext cipher(context);
        CAHELGPUPlaintext x_plain(context);
        encoder.encode(context, input, scale, x_plain);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        secret_key.gen_secretkey(context);
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, x_plain, cipher, false);
        CAHELGPUCiphertext temp = cipher;
        CAHELGPUCiphertext cipher_copy(move(cipher));
        ASSERT_EQ(cipher_copy == temp, 1);
        CAHELGPUCiphertext cipher_copy_op = move(cipher_copy);
        ASSERT_EQ(temp == cipher_copy_op, 1);
    }
    TEST(CKKSCipherTest, SaveLoad)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        switch (poly_modulus_degree)
        {
        case 4096:
            parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
            break;
        case 8192:
            parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
            break;
        case 16384:
            parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 60}));
            break;
        case 32768:
            parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));
            break;
        default:
            throw std::invalid_argument("unsupported polynomial degree");
            return;
        }
        double scale = pow(2.0, 40);
        CAHELGPUContext context(parms);
        CAHELGPUCKKSEncoder encoder(context);
        size_t slot_count = encoder.slot_count();
        vector<double> input;
        size_t msg_size = slot_count;
        input.reserve(msg_size);
        double rand_real;
        srand(time(0));
        for (size_t i = 0; i < msg_size; i++)
        {
            rand_real = (double)rand() / RAND_MAX;
            input.push_back(rand_real);
        }
        CAHELGPUCiphertext cipher(context);
        CAHELGPUPlaintext x_plain(context);
        encoder.encode(context, input, scale, x_plain);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        secret_key.gen_secretkey(context);
        secret_key.gen_publickey(context, public_key);
        public_key.encrypt_asymmetric(context, x_plain, cipher, false);
        std::stringstream cipher_stream;
        cipher.save(cipher_stream);
        CAHELGPUCiphertext cipher_load(context);
        cipher_load.load(context, cipher_stream);
        ASSERT_EQ(cipher == cipher_load, 1);
    }
}