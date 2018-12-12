cmake_minimum_required(VERSION 3.12)
project(tensor_eigen)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(/usr/include/eigen3)

include_directories(/opt/intel/mkl/include)
link_directories(/opt/intel/mkl/lib/intel64)
link_libraries(libmkl_core.a libmkl_blas95_ilp64.a libmkl_rt.so)


include_directories(/usr/local/fftw/include)
link_directories(/usr/local/fftw/lib)
link_libraries(libfftw3.a libfftw3f.a)


FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(tensor_eigen main.cpp time.cpp tensor.h)