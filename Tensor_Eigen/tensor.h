//
// Created by jcfei on 4/30/18.
//

#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include <string.h>

#include <iostream>
#include<iomanip>

#include <stdlib.h>
#include <stdio.h>


#include <math.h>
#include <vector>
#include <complex>

#include <fftw3.h>
#include "mkl.h"
//#include "mkl_service.h"
#include "time.h"
#include <omp.h>

#include "Eigen/SVD"
#include "Eigen/Dense"

#include <unsupported/Eigen/CXX11/Tensor>


using namespace std;
using namespace Eigen;

// cp 结构
template<class T>
struct cp_mats{
    Tensor<T,3> A,B,C;
};

//tucker 返回结构
template<class T>
struct tucker_core{
    Tensor<T,3> core;
    Matrix<T, Dynamic, Dynamic> u1,u2,u3;
};

//tsvd 返回结构
template<class T>
struct tsvd_core{
    Tensor<T,3> U,Theta,V;
};

//tt 返回结构
template<class T>
struct tt{
    Tensor<T,3> U,Theta,V;
};

#endif //TENSOR_TENSOR_H