g++ -I/opt/intel/mkl/include main.cpp -o test -lmkl_rt -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64 -O2 -larmadillo -std=c++11
./test
