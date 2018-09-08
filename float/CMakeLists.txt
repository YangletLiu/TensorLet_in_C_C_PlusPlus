cmake_minimum_required(VERSION 3.12)
project(tensor)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

include_directories(${ARMADILLO_INCLUDE_DIRS})
link_libraries(armadillo gfortran)

include_directories(/opt/intel/mkl/include)
link_directories(/opt/intel/mkl/lib/intel64)
link_libraries(libmkl_core.a libmkl_blas95_ilp64.a libmkl_rt.so)


include_directories(/usr/local/fftw/include)
link_directories(/usr/local/fftw/lib)
link_libraries(libfftw3.a libfftw3f.a)


add_executable(tensor
        cmake-build-debug/cmake_install.cmake
        cmake-build-debug/CMakeCache.txt
        cmake-build-debug/Makefile
        cmake-build-debug/tensor
        cmake-build-debug/tensor.cbp
        CMakeLists.txt
        main.cpp
        tensor.cpp
        tensor.h)
