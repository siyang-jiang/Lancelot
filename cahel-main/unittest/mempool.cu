#include <gtest/gtest.h>
//#include "cahel.h"
#include "gpu_mempool.cuh"

using namespace std;
//using namespace cahel;

namespace CAHELTest {
    TEST(PoolTest, test1) {
        cahel::rmmMempool pool;
        pool.device_add_bin(256);
        pool.device_add_bin(512);
        pool.device_add_bin(1024);
        pool.device_add_bin(2048);
        pool.device_add_bin(4096);
        void *ptr1 = pool.device_allocate(1);
//        pool.device_deallocate(ptr1, 256);
        void *ptr2 = pool.device_allocate(2);
        void *ptr3 = pool.device_allocate(1024);
        void *ptr4 = pool.device_allocate(2048);
        void *ptr5 = pool.device_allocate(4096);
        void *ptr6 = pool.device_allocate(4096);
        void *ptr7 = pool.device_allocate(4096);
        void *ptr8 = pool.device_allocate(4096);
        cout << rmm::detail::CUDA_ALLOCATION_ALIGNMENT << endl;

//        EXPECT_NE(ptr1, nullptr);
//        EXPECT_NE(ptr2, nullptr);
//        EXPECT_NE(ptr3, nullptr);
    }
}
