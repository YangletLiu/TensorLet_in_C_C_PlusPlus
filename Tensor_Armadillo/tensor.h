//
// Created by jcfei on 4/30/18.
//

#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include <string>
#include <iostream>
#include<iomanip>
#include <cstdlib>
#include <stdio.h>


#include <math.h>
#include <cmath>
#include <vector>
#include <complex>
#include <string>

#include <fftw3.h>
#include "mkl.h"

//#include "mkl_service.h"
#include "time.h"
#include <omp.h>
#include <armadillo>

using namespace std;
using namespace arma;


// cp 结构
template<class T>
struct cp_mats{
    Mat<T> A,B,C;
};

//tucker 返回结构
template<class T>
struct tucker_core{
    Cube<T> core;
    Mat<T> u1,u2,u3;
};

//tsvd 返回结构
template<class T>
struct tsvd_core{
    Cube<T> U,Theta,V;
};

#endif //TENSOR_TENSOR_H