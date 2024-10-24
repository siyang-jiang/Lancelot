#include "cahel.h"
#include <gtest/gtest.h>
using namespace std;
using namespace cahel;
namespace CAHELTest
{
    TEST(PublickeyTest, Constructor)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        secret_key.gen_secretkey(context);
        ASSERT_NO_THROW(secret_key.gen_publickey(context, public_key));
    }
    TEST(PublickeyTest, Copy)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        secret_key.gen_secretkey(context);
        secret_key.gen_publickey(context, public_key);
        CAHELGPUPublicKey temp(public_key);
        CAHELGPUPublicKey public_key_copy(move(public_key));
        ASSERT_EQ(temp.pk_ == public_key_copy.pk_, 1);

    }
    TEST(PublickeyTest, Move)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        secret_key.gen_secretkey(context);
        secret_key.gen_publickey(context, public_key);
        CAHELGPUPublicKey temp(public_key);
        ASSERT_EQ(temp.pk_ == public_key.pk_, 1);
        CAHELGPUPublicKey public_key_copy(move(public_key));
        ASSERT_EQ(temp.pk_ == public_key_copy.pk_, 1);


    }
    TEST(PublickeyTest, Encrypt)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUBatchEncoder batchEncoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        size_t slot_count = batchEncoder.slots_;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            pod_matrix[idx] = idx;
        }
        batchEncoder.encode(pod_matrix, plain_matrix);
        secret_key.gen_secretkey(context);
        secret_key.gen_publickey(context, public_key);
        ASSERT_NO_THROW(public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false));
    }
    TEST(PublickeyTest, SaveLoad)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPublicKey public_key(context);
        secret_key.gen_secretkey(context);
        secret_key.gen_publickey(context, public_key);
        std::stringstream pubkey_stream;
        public_key.save(pubkey_stream);
        CAHELGPUPublicKey public_key_load(context);
        public_key_load.load(context, pubkey_stream);
        ASSERT_EQ(public_key.pk_ == public_key_load.pk_, 1);
    }
    TEST(SecretkeyTest, Constructor)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        ASSERT_NO_THROW(CAHELGPUSecretKey secret_key(parms));
        CAHELGPUSecretKey secret_key(parms);
        ASSERT_NO_THROW(secret_key.gen_secretkey(context));
    }
    TEST(SecretkeyTest, Copy)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        secret_key.gen_secretkey(context);
        ASSERT_NO_THROW(CAHELGPUSecretKey secret_key_copy(secret_key));
        ASSERT_NO_THROW(CAHELGPUSecretKey secret_key_copy = secret_key);
    }
    TEST(SecretkeyTest, Move)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        secret_key.gen_secretkey(context);
        ASSERT_NO_THROW(CAHELGPUSecretKey secret_key_copy(move(secret_key)));
        ASSERT_NO_THROW(CAHELGPUSecretKey secret_key_copy = move(secret_key));
    }
    TEST(SecretkeyTest, Encrypt)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUBatchEncoder batchEncoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUSecretKey secret_key(parms);
        secret_key.gen_secretkey(context);
        size_t slot_count = batchEncoder.slots_;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            pod_matrix[idx] = idx;
        }
        batchEncoder.encode(pod_matrix, plain_matrix);
        ASSERT_NO_THROW(secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false));
    }
    TEST(SecretkeyTest, DecryptSym)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUBatchEncoder batchEncoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPlaintext dec_plain(context);
        secret_key.gen_secretkey(context);
        size_t slot_count = batchEncoder.slots_;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            pod_matrix[idx] = idx;
        }
        batchEncoder.encode(pod_matrix, plain_matrix);
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        std::vector<int64_t> dec_res;
        CAHELGPUSecretKey secret_key_mo = move(secret_key);
        secret_key_mo.decrypt(context, sym_cipher, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (int i = 0; i < pod_matrix.size(); i++)
            ASSERT_EQ(pod_matrix[i], dec_res[i]);
    }
    TEST(SecretkeyTest, DecryptAsym)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUBatchEncoder batchEncoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUCiphertext asym_cipher(context);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPlaintext dec_plain(context);
        secret_key.gen_secretkey(context);
        CAHELGPUPublicKey public_key(context);
        secret_key.gen_publickey(context, public_key);
        size_t slot_count = batchEncoder.slots_;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
            pod_matrix[idx] = idx;
        }
        batchEncoder.encode(pod_matrix, plain_matrix);
        CAHELGPUSecretKey secret_key_op(secret_key);
        public_key.encrypt_asymmetric(context, plain_matrix, asym_cipher, false);
//        secret_key_op.decrypt(context, asym_cipher, dec_plain);
        secret_key.decrypt(context, asym_cipher, dec_plain);
        std::vector<int64_t> dec_res;
        batchEncoder.decode(dec_plain, dec_res);
        ASSERT_EQ(pod_matrix, dec_res);
    }
    TEST(RelinkeyTest, Constructor)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        ASSERT_NO_THROW(CAHELGPURelinKey relin_keys(context));
        CAHELGPURelinKey relin_keys(context);
        secret_key.gen_secretkey(context);
        ASSERT_NO_THROW(secret_key.gen_relinkey(context, relin_keys));
    }
    TEST(RelinkeyTest, Copy)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPURelinKey relin_keys(context);
        secret_key.gen_secretkey(context);
        secret_key.gen_relinkey(context, relin_keys);
        ASSERT_NO_THROW(CAHELGPURelinKey relin_keys_copy(relin_keys));
        ASSERT_NO_THROW(CAHELGPURelinKey relin_keys_copy = relin_keys);
    }
    TEST(RelinkeyTest, Move)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPURelinKey relin_keys(context);
        secret_key.gen_secretkey(context);
        secret_key.gen_relinkey(context, relin_keys);
        ASSERT_NO_THROW(CAHELGPURelinKey relin_keys_copy(move(relin_keys)));
        ASSERT_NO_THROW(CAHELGPURelinKey relin_keys_copy = move(relin_keys));
    }
    TEST(RelinkeyTest, SaveLoad)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPURelinKey relin_keys(context);
        secret_key.gen_secretkey(context);
        secret_key.gen_relinkey(context, relin_keys);
        std::stringstream relikey_stream;
        relin_keys.save(relikey_stream);
        CAHELGPURelinKey relin_keys_load(context);
        relin_keys_load.load(context, relikey_stream);
        ASSERT_EQ(relin_keys.parms_id(), relin_keys_load.parms_id());
        ASSERT_EQ(relin_keys.pk_num_, relin_keys_load.pk_num_);
        ASSERT_EQ(relin_keys.gen_flag_, relin_keys_load.gen_flag_);
        for (int i = 0; i < relin_keys.pk_num_; i++)
        {
            ASSERT_EQ(relin_keys.public_keys_[i].pk_ == relin_keys_load.public_keys_[i].pk_, 1);
        }
    }
    TEST(GaloiskeyTest, Constructor)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 16384;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        //    ASSERT_NO_THROW(CAHELGPUGaloisKey galois_key(context));
        CAHELGPUGaloisKey galois_key(context);
        secret_key.gen_secretkey(context);
        ASSERT_NO_THROW(secret_key.create_galois_keys(context, galois_key));
    }
    TEST(GaloiskeyTest, Copy)
    {

        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUGaloisKey galois_key(context);
        secret_key.gen_secretkey(context);
        secret_key.create_galois_keys(context, galois_key);
        ASSERT_NO_THROW(CAHELGPUGaloisKey galois_key_copy(galois_key));
        ASSERT_NO_THROW(CAHELGPUGaloisKey galois_key_copy = galois_key);
    }
    TEST(GaloiskeyTest, Move)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUGaloisKey galois_key(context);
        secret_key.gen_secretkey(context);
        secret_key.create_galois_keys(context, galois_key);
        ASSERT_NO_THROW(CAHELGPUGaloisKey galois_key_copy(move(galois_key)));
        ASSERT_NO_THROW(CAHELGPUGaloisKey galois_key_copy = move(galois_key));
    }
    TEST(GaloiskeyTest, Apply)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 16384;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUBatchEncoder batchEncoder(context);
        CAHELGPUPlaintext plain_matrix(context);
        CAHELGPUCiphertext sym_cipher(context);
        CAHELGPUSecretKey secret_key(parms);
        CAHELGPUPlaintext dec_plain(context);
        secret_key.gen_secretkey(context);
        CAHELGPUGaloisKey galois_key(context);
        secret_key.create_galois_keys(context, galois_key);
        size_t slot_count = batchEncoder.slots_;
        std::vector<int64_t> dec_res;
        std::vector<int64_t> pod_matrix(slot_count, 0ULL);
        for (size_t idx{0}; idx < slot_count; idx++)
        {
           pod_matrix[idx] = idx;
        }
        batchEncoder.encode(pod_matrix, plain_matrix);
        secret_key.encrypt_symmetric(context, plain_matrix, sym_cipher, false);
        CAHELGPUCiphertext temp = sym_cipher;
        //        CAHELGPUGaloisKey galois_key_op =  galois_key;
        //        rotate_columns_inplace(context, sym_cipher, galois_key_op);
        //        secret_key.decrypt(context, sym_cipher, dec_plain);
        //        batchEncoder.decode(dec_plain, dec_res);
        //        for (size_t idx{0}; idx < 50; idx++)
        //        {
        //            ASSERT_EQ(dec_res[idx + slot_count / 2], idx);
        //        }
        CAHELGPUGaloisKey galois_key_mo =  move(galois_key);
        rotate_columns_inplace(context, temp, galois_key_mo);
        secret_key.decrypt(context, temp, dec_plain);
        batchEncoder.decode(dec_plain, dec_res);
        for (size_t idx{0}; idx < 50; idx++)
        {
            ASSERT_EQ(dec_res[idx + slot_count / 2], idx);
        }
    }

    TEST(GaloiskeyTest, SaveLoad)
    {
        EncryptionParameters parms(scheme_type::bfv);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        CAHELGPUContext context(parms);
        CAHELGPUSecretKey secret_key(parms);
        secret_key.gen_secretkey(context);
        CAHELGPUGaloisKey galois_key(context);
        secret_key.create_galois_keys(context, galois_key);
        std::stringstream galokey_stream;
        galois_key.save(galokey_stream);
        CAHELGPUGaloisKey galois_key_load(context);
        galois_key_load.load(context, galokey_stream);
        ASSERT_EQ(galois_key.parms_id(), galois_key_load.parms_id());
        ASSERT_EQ(galois_key.relin_key_num_, galois_key_load.relin_key_num_);
        ASSERT_EQ(galois_key.gen_flag_, galois_key_load.gen_flag_);
        for (int i = 0; i < galois_key.relin_key_num_; i++)
        {
            ASSERT_EQ(galois_key.relin_keys_[i].parms_id(), galois_key_load.relin_keys_[i].parms_id());
            ASSERT_EQ(galois_key.relin_keys_[i].pk_num_, galois_key_load.relin_keys_[i].pk_num_);
            ASSERT_EQ(galois_key.relin_keys_[i].gen_flag_, galois_key.relin_keys_[i].gen_flag_);
            for (int j = 0; j < galois_key.relin_keys_[i].pk_num_; j++)
            {
                ASSERT_EQ(galois_key.relin_keys_[i].public_keys_[j].pk_ == galois_key_load.relin_keys_[i].public_keys_[j].pk_, 1);
            }
        }
    }
}
