add_executable(unit_test
        plaintext.cu
        batchencoder.cu
        ciphertext.cu
        ckks_encoder.cu
        encryptionparams.cu
        evaluator.cu
        math.cu
        mempool.cu
        secretkey.cu
        unit_test_runner.cu

        util/common.cu
        util/hash.cu
        util/numth.cu
        util/rns.cu
        util/stringtouint64.cu
        util/uint64tostring.cu
        util/uintarith.cu
        util/uintarithmod.cu
        util/uintarithsmallmod.cu
        util/uintcore.cu
        )

target_link_libraries(unit_test
        PRIVATE cahel
        PUBLIC GTest::gtest)
