cmake_minimum_required(VERSION 3.12)
project(tensor)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

include_directories(${ARMADILLO_INCLUDE_DIRS})
link_libraries(armadillo gfortran)


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
