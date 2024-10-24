#include "cahel.h"
#include "uintmath.h"
#include <gtest/gtest.h>
using namespace std;
using namespace cahel;

namespace CAHELTest
{
    TEST(UintmathTest, AddUintHost)
    {
        vector<uint64_t> temp1(2, 0ULL);
        vector<uint64_t> temp2(2, 0ULL);

        add_uint_host(temp1.data(), 2, 0ULL, temp2.data());
        ASSERT_FALSE(add_uint_host(temp1.data(), 2, 0ULL, temp2.data()));
        ASSERT_EQ(0ULL, temp2[0]);
        ASSERT_EQ(0ULL, temp2[1]);

        temp1[0] = 0xFFFFFFFF00000000ULL;
        temp1[1] = 0ULL;
        ASSERT_FALSE(add_uint_host(temp1.data(), 2, 0xFFFFFFFFULL, temp2.data()));
        add_uint_host(temp1.data(), 2, 0xFFFFFFFFULL, temp2.data());
        ASSERT_EQ(0xFFFFFFFFFFFFFFFFULL, temp2[0]);
        ASSERT_EQ(0ULL, temp2[1]);

        temp1[0] = 0xFFFFFFFF00000000ULL;
        temp1[1] = 0xFFFFFFFF00000000ULL;
        ASSERT_FALSE(add_uint_host(temp1.data(), 2, 0x100000000ULL, temp2.data()));
        add_uint_host(temp1.data(), 2, 0x100000000ULL, temp2.data());
        ASSERT_EQ(0ULL, temp2[0]);
        ASSERT_EQ(0xFFFFFFFF00000001ULL, temp2[1]);

        temp1[0] = 0xFFFFFFFFFFFFFFFFULL;
        temp1[1] = 0xFFFFFFFFFFFFFFFFULL;
        ASSERT_TRUE(add_uint_host(temp1.data(), 2, 1ULL, temp2.data()));
        add_uint_host(temp1.data(), 2, 1ULL, temp2.data());
        ASSERT_EQ(0ULL, temp2[0]);
        ASSERT_EQ(0ULL, temp2[1]);
    }
    TEST(UintmathTest, DivideUint128InplaceHost)
    {
        vector<uint64_t> input(2, 0ULL);
        vector<uint64_t> result(2, 0ULL);
        divide_uint128_inplace_host(input.data(), 1ULL, result.data());
        ASSERT_EQ(0ULL, result[0]);
        ASSERT_EQ(0ULL, result[1]);

        input[0] = 1;
        input[1] = 0;
        divide_uint128_inplace_host(input.data(), 1ULL, result.data());
        ASSERT_EQ(1ULL, result[0]);
        ASSERT_EQ(0ULL, result[1]);

        input[0] = 0x10101010;
        input[1] = 0x2B2B2B2B;
        divide_uint128_inplace_host(input.data(), 0x1000ULL, result.data());
        ASSERT_EQ(0xB2B0000000010101ULL, result[0]);
        ASSERT_EQ(0x2B2B2ULL, result[1]);

        input[0] = 1212121212121212ULL;
        input[1] = 3434343434343434ULL;
        divide_uint128_inplace_host(input.data(), 5656565656565656ULL, result.data());
        ASSERT_EQ(5252525252525252ULL, input[0]);
        ASSERT_EQ(0ULL, input[1]);
        ASSERT_EQ(11199808901895084909ULL, result[0]);
        ASSERT_EQ(0ULL, result[1]);
    }
}