add_subdirectory(pybind11)
pybind11_add_module(pyCAHEL src/binding.cu)
target_link_libraries(pyCAHEL PRIVATE CAHEL)
