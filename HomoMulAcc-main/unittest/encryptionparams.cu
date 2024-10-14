#include "cahel.h"
#include <gtest/gtest.h>
using namespace std;
using namespace cahel;
namespace CAHELTest
{
    TEST(ParamsTest, Constructor)
    {
        ASSERT_NO_THROW(EncryptionParameters parms(scheme_type::ckks));
        EncryptionParameters parms(scheme_type::ckks);

        size_t poly_modulus_degree = 4096;

        ASSERT_NO_THROW(parms.set_poly_modulus_degree(poly_modulus_degree));
        parms.set_poly_modulus_degree(poly_modulus_degree);

        ASSERT_NO_THROW(parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree)));
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));

        ASSERT_NO_THROW(EncryptionParameters parms_bfv(scheme_type::bfv));
        EncryptionParameters parms_bfv(scheme_type::bfv);

        size_t poly_modulus_degree_bfv = 32768;

        ASSERT_NO_THROW(parms_bfv.set_poly_modulus_degree(poly_modulus_degree_bfv));
        parms_bfv.set_poly_modulus_degree(poly_modulus_degree_bfv);

        ASSERT_NO_THROW(parms_bfv.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree_bfv)));
        ASSERT_NO_THROW(parms_bfv.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree_bfv, 20)));
    }
    TEST(ParamsTest, SaveLoad)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        std::stringstream parm_stream;
        parms.save(parm_stream);
        EncryptionParameters parms_load;
        parms_load.load(parm_stream);
        ASSERT_EQ(parms.scheme(), parms_load.scheme());
        ASSERT_EQ(parms.poly_modulus_degree(), parms_load.poly_modulus_degree());
        ASSERT_EQ(parms.coeff_modulus(), parms_load.coeff_modulus());
        //ASSERT_EQ(1,parms==parms_load);

        EncryptionParameters parms_bfv(scheme_type::bfv);
        size_t poly_modulus_degree_bfv = 4096;
        parms_bfv.set_poly_modulus_degree(poly_modulus_degree_bfv);
        parms_bfv.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree_bfv));
        parms_bfv.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree_bfv, 20));
        std::stringstream parm_stream_bfv;
        parms_bfv.save(parm_stream_bfv);
        EncryptionParameters parms_load_bfv;
        parms_load_bfv.load(parm_stream_bfv);
        ASSERT_EQ(parms_bfv.scheme(), parms_load_bfv.scheme());
        ASSERT_EQ(parms_bfv.poly_modulus_degree(), parms_load_bfv.poly_modulus_degree());
        ASSERT_EQ(parms_bfv.coeff_modulus(), parms_load_bfv.coeff_modulus());
        ASSERT_EQ(parms_bfv.plain_modulus(), parms_load_bfv.plain_modulus());
    }
}
