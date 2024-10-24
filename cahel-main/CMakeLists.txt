cmake_minimum_required(VERSION 3.20)

#include(cmake/CPM.cmake)
#
#if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/EXAMPLE_RAPIDS.cmake)
#    file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-23.06/RAPIDS.cmake
#            ${CMAKE_CURRENT_BINARY_DIR}/EXAMPLE_RAPIDS.cmake)
#endif()
#include(${CMAKE_CURRENT_BINARY_DIR}/EXAMPLE_RAPIDS.cmake)
#
#include(rapids-cmake)
#include(rapids-cpm)
#include(rapids-cuda)
#include(rapids-export)
#include(rapids-find)

project(CAHEL LANGUAGES CXX CUDA VERSION 1.3 DESCRIPTION "CUDA-Accelerated Homomorphic Encryption Library")

# [option] CAHEL_PROFILE (default: OFF)
option(CAHEL_PROFILE "Profiling mode" OFF)
message(STATUS "Profiling mode: ${CAHEL_PROFILE}")
if (CAHEL_PROFILE)
    add_compile_definitions(CAHEL_PROFILE)
endif ()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86 89)
endif()

# output the compile option in file compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# fetching boost and building is time consuming, so we ask users to install from package manager
#find_package(Boost REQUIRED)
# boost headers is available through include_directories
#include_directories(${Boost_INCLUDE_DIRS})

# include "include/" directory is enough
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# add dependencies
#rapids_find_package(rmm)

# library
add_subdirectory(src)

# examples
add_subdirectory(examples)

# python binding
add_subdirectory(python)

# build unit test
# default not build unit tests

#set(BUILD_UNIT_TESTS OFF)
#if(BUILD_UNIT_TESTS)
#    add_subdirectory(unittest)
#endif()

# config for installation

install(TARGETS CAHEL EXPORT CAHELConfig)
install(EXPORT CAHELConfig NAMESPACE CAHEL:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CAHEL)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/CAHEL)
