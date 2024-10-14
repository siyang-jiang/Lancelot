
#include "util/numth.h"
#include "util/rns.h"
#include "util/uintarithmod.h"
#include "util/uintarithsmallmod.h"
#include "modulus.h"
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>

using namespace cahel::util;
using namespace cahel;
using namespace std;

namespace CAHELtest
{
    namespace util
    {
        TEST(RNSBaseTest, Create)
        {
            ASSERT_THROW(RNSBase base({0}), invalid_argument);
            ASSERT_THROW(RNSBase base({0, 3}), invalid_argument);
            ASSERT_THROW(RNSBase base({2, 2}), invalid_argument);
            ASSERT_THROW(RNSBase base({2, 3, 4}), invalid_argument);
            ASSERT_THROW(RNSBase base({3, 4, 5, 6}), invalid_argument);
            ASSERT_NO_THROW(RNSBase base({3, 4, 5, 7}));
            ASSERT_NO_THROW(RNSBase base({2}));
            ASSERT_NO_THROW(RNSBase base({3}));
            ASSERT_NO_THROW(RNSBase base({4}));
        }

        TEST(RNSBaseTest, ArrayAccess)
        {
            {
                RNSBase base({2});
                ASSERT_EQ(size_t(1), base.size());
                ASSERT_EQ(Modulus(2), base[0]);
                ASSERT_THROW(
                    [&]()
                    {
                        return base[1].value();
                    }(),
                    out_of_range);
            }
            {
                RNSBase base({2, 3, 5});
                ASSERT_EQ(size_t(3), base.size());
                ASSERT_EQ(Modulus(2), base[0]);
                ASSERT_EQ(Modulus(3), base[1]);
                ASSERT_EQ(Modulus(5), base[2]);
                ASSERT_THROW(
                    [&]()
                    {
                        return base[3].value();
                    }(),
                    out_of_range);
            }
        }

        TEST(RNSBaseTest, Contains)
        {
            RNSBase base({2, 3, 5, 13});
            ASSERT_TRUE(base.contains(2));
            ASSERT_TRUE(base.contains(3));
            ASSERT_TRUE(base.contains(5));
            ASSERT_TRUE(base.contains(13));
            ASSERT_FALSE(base.contains(7));
            ASSERT_FALSE(base.contains(4));
            ASSERT_FALSE(base.contains(0));
        }

        TEST(RNSBaseTest, IsSubbaseOf)
        {
            {
                RNSBase base({2});
                RNSBase base2({2});
                ASSERT_TRUE(base.is_subbase_of(base2));
                ASSERT_TRUE(base2.is_subbase_of(base));
                ASSERT_TRUE(base.is_superbase_of(base2));
                ASSERT_TRUE(base2.is_superbase_of(base));
            }
            {
                RNSBase base({2});
                RNSBase base2({2, 3});
                ASSERT_TRUE(base.is_subbase_of(base2));
                ASSERT_FALSE(base2.is_subbase_of(base));
                ASSERT_FALSE(base.is_superbase_of(base2));
                ASSERT_TRUE(base2.is_superbase_of(base));
            }
            {
                // Order does not matter for subbase/superbase
                RNSBase base({3, 13, 7});
                RNSBase base2({2, 3, 5, 7, 13, 19});
                ASSERT_TRUE(base.is_subbase_of(base2));
                ASSERT_FALSE(base2.is_subbase_of(base));
                ASSERT_FALSE(base.is_superbase_of(base2));
                ASSERT_TRUE(base2.is_superbase_of(base));
            }
            {
                RNSBase base({3, 13, 7, 23});
                RNSBase base2({2, 3, 5, 7, 13, 19});
                ASSERT_FALSE(base.is_subbase_of(base2));
                ASSERT_FALSE(base2.is_subbase_of(base));
                ASSERT_FALSE(base.is_superbase_of(base2));
                ASSERT_FALSE(base2.is_superbase_of(base));
            }
        }

        TEST(RNSBaseTest, Extend)
        {
            RNSBase base({3});

            RNSBase base2 = base.extend(5);
            ASSERT_EQ(size_t(2), base2.size());
            ASSERT_EQ(base[0], base2[0]);
            ASSERT_EQ(Modulus(5), base2[1]);

            RNSBase base3 = base2.extend(7);
//            ASSERT_EQ(4l, pool.use_count());
            ASSERT_EQ(size_t(3), base3.size());
            ASSERT_EQ(base2[0], base3[0]);
            ASSERT_EQ(base2[1], base3[1]);
            ASSERT_EQ(Modulus(7), base3[2]);

            ASSERT_THROW(auto base4 = base3.extend(0), invalid_argument);
            ASSERT_THROW(auto base4 = base3.extend(14), logic_error);

            RNSBase base4({3, 4, 5});
            RNSBase base5({7, 11, 13, 17});
            RNSBase base6 = base4.extend(base5);
            ASSERT_EQ(size_t(7), base6.size());
            ASSERT_EQ(Modulus(3), base6[0]);
            ASSERT_EQ(Modulus(4), base6[1]);
            ASSERT_EQ(Modulus(5), base6[2]);
            ASSERT_EQ(Modulus(7), base6[3]);
            ASSERT_EQ(Modulus(11), base6[4]);
            ASSERT_EQ(Modulus(13), base6[5]);
            ASSERT_EQ(Modulus(17), base6[6]);

            ASSERT_THROW(auto base7 = base4.extend(RNSBase({7, 10, 11})), invalid_argument);
        }

        TEST(RNSBaseTest, Drop)
        {
            RNSBase base({3, 5, 7, 11});

            RNSBase base2 = base.drop();
            ASSERT_EQ(size_t(3), base2.size());
            ASSERT_EQ(base[0], base2[0]);
            ASSERT_EQ(base[1], base2[1]);
            ASSERT_EQ(base[2], base2[2]);

            RNSBase base3 = base2.drop().drop();
            ASSERT_EQ(size_t(1), base3.size());
            ASSERT_EQ(base[0], base3[0]);

            ASSERT_THROW(auto b = base3.drop(), logic_error);
            ASSERT_THROW(auto b = base3.drop(3), logic_error);
            ASSERT_THROW(auto b = base3.drop(5), logic_error);

            RNSBase base4 = base.drop(5);
            ASSERT_EQ(size_t(3), base4.size());
            ASSERT_EQ(base[0], base4[0]);
            ASSERT_EQ(base[2], base4[1]);
            ASSERT_EQ(base[3], base4[2]);

            ASSERT_THROW(auto b = base4.drop(13), logic_error);
            ASSERT_THROW(auto b = base4.drop(0), logic_error);
            ASSERT_NO_THROW(auto b = base4.drop(7).drop(11));
            ASSERT_THROW(auto b = base4.drop(7).drop(11).drop(3), logic_error);
        }

        TEST(RNSBaseTest, ComposeDecompose)
        {
            auto rns_test = [&](const RNSBase &base, vector<uint64_t> in, vector<uint64_t> out)
            {
                auto in_copy = in;
                base.decompose(in_copy.data());
                ASSERT_TRUE(in_copy == out);
                base.compose(in_copy.data());
//                cout<<in_copy[0]<<"  "<<in_copy[1]<<endl;
                ASSERT_TRUE(in_copy == in);
            };

            {
                RNSBase base({2});
                rns_test(base, {0}, {0});
                rns_test(base, {1}, {1});

            }
            {
                RNSBase base({5});
                rns_test(base, {0}, {0});
                rns_test(base, {1}, {1});
                rns_test(base, {2}, {2});
                rns_test(base, {3}, {3});
                rns_test(base, {4}, {4});

            }
            {
                RNSBase base({3, 5});
                rns_test(base, {0, 0}, {0, 0});
                rns_test(base, {1, 0}, {1, 1});
                rns_test(base, {2, 0}, {2, 2});
                rns_test(base, {3, 0}, {0, 3});
                rns_test(base, {4, 0}, {1, 4});
                rns_test(base, {5, 0}, {2, 0});
                rns_test(base, {8, 0}, {2, 3});
                rns_test(base, {12, 0}, {0, 2});
                rns_test(base, {14, 0}, {2, 4});

            }
            {
                RNSBase base({2, 3, 5});
                rns_test(base, {0, 0, 0}, {0, 0, 0});
                rns_test(base, {1, 0, 0}, {1, 1, 1});
                rns_test(base, {2, 0, 0}, {0, 2, 2});
                rns_test(base, {3, 0, 0}, {1, 0, 3});
                rns_test(base, {4, 0, 0}, {0, 1, 4});
                rns_test(base, {5, 0, 0}, {1, 2, 0});
                rns_test(base, {10, 0, 0}, {0, 1, 0});
                rns_test(base, {11, 0, 0}, {1, 2, 1});
                rns_test(base, {16, 0, 0}, {0, 1, 1});
                rns_test(base, {27, 0, 0}, {1, 0, 2});
                rns_test(base, {29, 0, 0}, {1, 2, 4});

            }
            {
                RNSBase base({13, 37, 53, 97});
                rns_test(base, {0, 0, 0, 0}, {0, 0, 0, 0});
                rns_test(base, {1, 0, 0, 0}, {1, 1, 1, 1});
                rns_test(base, {2, 0, 0, 0}, {2, 2, 2, 2});
                rns_test(base, {12, 0, 0, 0}, {12, 12, 12, 12});
                rns_test(base, {321, 0, 0, 0}, {9, 25, 3, 30});

            }
            {
                // Large example
                auto primes = get_primes(1024, 60, 4);
                vector<uint64_t> in_values{0xAAAAAAAAAAA, 0xBBBBBBBBBB, 0xCCCCCCCCCC, 0xDDDDDDDDDD};
                RNSBase base(primes);
                rns_test(
                    base, in_values,
                    {modulo_uint(in_values.data(), in_values.size(), primes[0]),
                     modulo_uint(in_values.data(), in_values.size(), primes[1]),
                     modulo_uint(in_values.data(), in_values.size(), primes[2]),
                     modulo_uint(in_values.data(), in_values.size(), primes[3])});
            }
        }

        TEST(RNSBaseTest, ComposeDecomposeArray)
        {
            auto rns_test = [&](const RNSBase &base, size_t count, vector<uint64_t> in, vector<uint64_t> out)
            {
                auto in_copy = in;
                base.decompose_array(in_copy.data(), count);
                ASSERT_TRUE(in_copy == out);
                base.compose_array(in_copy.data(), count);
                ASSERT_TRUE(in_copy == in);
            };

            {
                RNSBase base({2});
                rns_test(base, 1, {0}, {0});
                rns_test(base, 1, {1}, {1});

            }
            {
                RNSBase base({5});
                rns_test(base, 3, {0, 1, 2}, {0, 1, 2});

            }
            {
                RNSBase base({3, 5});
                rns_test(base, 1, {0, 0}, {0, 0});
                rns_test(base, 1, {2, 0}, {2, 2});
                rns_test(base, 1, {7, 0}, {1, 2});
                rns_test(base, 2, {0, 0, 0, 0}, {0, 0, 0, 0});
                rns_test(base, 2, {1, 0, 2, 0}, {1, 2, 1, 2});
                rns_test(base, 2, {7, 0, 8, 0}, {1, 2, 2, 3});

            }
            {
                RNSBase base({3, 5, 7});
                rns_test(base, 1, {0, 0, 0}, {0, 0, 0});
                rns_test(base, 1, {2, 0, 0}, {2, 2, 2});
                rns_test(base, 1, {7, 0, 0}, {1, 2, 0});
                rns_test(base, 2, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0});
                rns_test(base, 2, {1, 0, 0, 2, 0, 0}, {1, 2, 1, 2, 1, 2});
                rns_test(base, 2, {7, 0, 0, 8, 0, 0}, {1, 2, 2, 3, 0, 1});
                rns_test(base, 3, {7, 0, 0, 8, 0, 0, 9, 0, 0}, {1, 2, 0, 2, 3, 4, 0, 1, 2});

            }
            {
                // Large example
                auto primes = get_primes(1024, 60, 2);
                vector<uint64_t> in_values{0xAAAAAAAAAAA, 0xBBBBBBBBBB, 0xCCCCCCCCCC,
                                           0xDDDDDDDDDD, 0xEEEEEEEEEE, 0xFFFFFFFFFF};
                RNSBase base(primes);
                rns_test(
                    base, 3, in_values,
                    {modulo_uint(in_values.data(), primes.size(), primes[0]),
                     modulo_uint(in_values.data() + 2, primes.size(), primes[0]),
                     modulo_uint(in_values.data() + 4, primes.size(), primes[0]),
                     modulo_uint(in_values.data(), primes.size(), primes[1]),
                     modulo_uint(in_values.data() + 2, primes.size(), primes[1]),
                     modulo_uint(in_values.data() + 4, primes.size(), primes[1])});

            }
        }

        TEST(BaseConverterTest, Initialize)
        {
            // Good cases
            ASSERT_NO_THROW(BaseConverter bct(RNSBase({2}), RNSBase({2})));
            ASSERT_NO_THROW(BaseConverter bct(RNSBase({2}), RNSBase({3})));
            ASSERT_NO_THROW(BaseConverter bct(RNSBase({2, 3, 5}), RNSBase({2})));
            ASSERT_NO_THROW(BaseConverter bct(RNSBase({2, 3, 5}), RNSBase({3, 5})));
            ASSERT_NO_THROW(BaseConverter bct(RNSBase({2, 3, 5}), RNSBase({2, 3, 5, 7, 11})));
            ASSERT_NO_THROW(BaseConverter bct(RNSBase({2, 3, 5}), RNSBase({7, 11})));
        }

        TEST(RNSToolTest, Initialize)
        {
            size_t poly_modulus_degree = 32;
            size_t coeff_base_count = 4;
            int prime_bit_count = 20;

            Modulus plain_t = 65537;
            RNSBase coeff_base(get_primes(poly_modulus_degree, prime_bit_count, coeff_base_count));

            ASSERT_NO_THROW(RNSTool rns_tool(poly_modulus_degree, coeff_base, plain_t));

            // Succeeds with 0 plain_modulus (case of CKKS)
            ASSERT_NO_THROW(RNSTool rns_tool(poly_modulus_degree, coeff_base, 0));

            // Fails when poly_modulus_degree is too small
            ASSERT_THROW(RNSTool rns_tool(1, coeff_base, plain_t), invalid_argument);
        }        
    } // namespace util
} // namespace sealtest
