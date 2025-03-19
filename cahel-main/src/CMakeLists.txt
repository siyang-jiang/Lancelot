### To use gcc/g++ on a Macintosh, you must set the Compilers
### here, not inside the project
##if(APPLE)
##       set(CMAKE_C_COMPILER "/usr/local/bin/gcc-7")
##       set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-7")
##endif()
### TODO: for now, we use CLang for Mac
###
### In order to create OpenFHE's static libraries you should enable
### the BUILD_STATIC option. For that, you run "cmake .. -DBUILD_STATIC=ON".
### After having your link completed you will find static libs
### with the suffix "_static" in ./build/libs/.
### Examples: OPENFHEpke_static.a, OPENFHEcore_static.a, etc.
### After you run "make install" in your build directory, you can build your custom application.
### If you need your application to be linked statically, then run "cmake .. -DBUILD_STATIC=ON"

find_package(OpenFHE CONFIG REQUIRED)
if (OpenFHE_FOUND)
        message(STATUS "FOUND PACKAGE OpenFHE")
        message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
        message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
        message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
        message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
        message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
else()
        message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif ()

set( CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS} )

include_directories( ${OPENMP_INCLUDES} )
include_directories( ${OpenFHE_INCLUDE} )
include_directories( ${OpenFHE_INCLUDE}/third-party/include )
include_directories( ${OpenFHE_INCLUDE}/core )
include_directories( ${OpenFHE_INCLUDE}/pke )
include_directories( ${OpenFHE_INCLUDE}/binfhe )
### add directories for other OpenFHE modules as needed for your project

link_directories( ${OpenFHE_LIBDIR} )
link_directories( ${OPENMP_LIBRARIES} )

set( CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS} )
link_libraries( ${OpenFHE_SHARED_LIBRARIES} )

ADD_LIBRARY(CAHEL SHARED
        context.cu
        modulus.cu
        mempool.cu
        batchencoder.cu
        prng.cu
        polymath.cu
        secretkey.cu
        rns.cu
        rns_base.cu
        rns_bconv.cu
        scalingvariant.cu
        evaluate.cu
        croots.cu
        fft.cu
        ckks.cu
        galois.cu
        eval_key_switch.cu

        ntt/fntt_2d.cu
        ntt/ntt_1d.cu
        ntt/ntt_keyswitch_old.cu
        ntt/intt_2d.cu
        ntt/ntt_modup.cu
        ntt/ntt_moddown.cu

        util/blake2b.cu
        util/blake2xb.cu
        util/globals.cu
        util/galois.cu
        util/hash.cu
        util/ntt.cu
        util/numth.cu
        util/rns.cu
        util/uintarith.cu
        util/uintarithmod.cu
        util/uintarithsmallmod.cu
        util/uintcore.cu
        )

set(CAHEL_HEADER_FILES
        ../include/batchencoder.h
        ../include/cahel.h
        ../include/ciphertext.h
        ../include/ckks.h
        ../include/common.h
        ../include/context.h
        ../include/croots.h
        ../include/util/encryptionparams.h
        ../include/evaluate.h
        ../include/fft.h
        ../include/galois.cuh
        ../include/gpucontext.h
        ../include/gputype.h
        ../include/lock.cuh
        ../include/mempool.cuh
        ../include/util/modulus.h
        ../include/ntt.cuh
        ../include/plaintext.h
        ../include/polymath.cuh
        ../include/prng.cuh
        ../include/rns.cuh
        ../include/rns_base.cuh
        ../include/rns_bconv.cuh
        ../include/scalingvariant.cuh
        ../include/secretkey.h
        ../include/uintmath.cuh
        ../include/uintmodmath.cuh
        ../include/util.cuh

        ../include/util/blake2b.h
        ../include/util/blake2-impl.h
        ../include/util/common.h
        ../include/util/defines.h
        ../include/util/galois.h
        ../include/util/globals.h
        ../include/util/hash.h
        ../include/util/hestdparms.h
        ../include/util/ntt.h
        ../include/util/numth.h
        ../include/util/polycore.h
        ../include/util/rns.h
        ../include/util/uintarith.h
        ../include/util/uintarithmod.h
        ../include/util/uintarithsmallmod.h
        ../include/util/uintcore.h
)
