#include "cahel.h"
#include <gtest/gtest.h>
using namespace std;
using namespace cahel;
namespace CAHELTest
{
    TEST(CKKSEncoderTest, Constructor)
    {
        EncryptionParameters parms(scheme_type::ckks);
        vector<int> array = {1, 2, 4, 8};
        int index = rand() % 4;
        size_t poly_modulus_degree = 4096 * array[index];
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
        ASSERT_NO_THROW(CAHELGPUCKKSEncoder encoder(context));
    }
    TEST(CKKSEncoderTest, Encode)
    {
        EncryptionParameters parms(scheme_type::ckks);
        vector<int> array = {1, 2, 4, 8};
        int index = rand() % 4;
        size_t poly_modulus_degree = 4096 * array[index];
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
        ASSERT_NO_THROW(CAHELGPUCKKSEncoder encoder(context));
        CAHELGPUCKKSEncoder encoder(context);
        CAHELGPUPlaintext plain_coeff3(context);
        ASSERT_NO_THROW(encoder.encode(context, 3.14159265, scale, plain_coeff3));
        size_t slot_count = encoder.slot_count();
        vector<cuDoubleComplex> input;
        double rand_real, rand_imag;
        size_t size = slot_count;
        input.reserve(size);
        for (size_t i = 0; i < size; i++)
        {
            rand_real = (double)rand() / RAND_MAX;
            rand_imag = (double)rand() / RAND_MAX;
            input.push_back(make_cuDoubleComplex(rand_real, rand_imag));
        }
        CAHELGPUPlaintext x_plain(context);
        ASSERT_NO_THROW(encoder.encode(context, input, scale, x_plain));
    }
    TEST(CKKSEncoderTest, Decode)
    {
        EncryptionParameters parms(scheme_type::ckks);
        vector<int> array = {1, 2, 4, 8};
        int index = rand() % 4;
        size_t poly_modulus_degree = 4096 * array[index];
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
        CAHELGPUPlaintext x_plain(context);
        encoder.encode(context, input, scale, x_plain);
        vector<double> result;
        encoder.decode(context, x_plain, result);
        for (int i = 0; i < input.size(); i++)
            ASSERT_NEAR(input[i], result[i], 0.00001);
    }
}