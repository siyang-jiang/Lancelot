# test SEAL examples
add_executable(example_context
        1_bfv_basics.cu
        2_encoder.cu
        3_bgv_basics.cu
        4_ckks_basics.cu
        5_bfv_opt.cu
        7_performance.cu
        example.cu
        )

target_link_libraries(example_context PRIVATE CAHEL)
