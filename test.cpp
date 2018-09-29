//
// Created by jcfei on 18-9-29.
//

#include <iostream>
#include <vector>
#include <complex>
#include <math.h>
#include "time.h"
using namespace std;
//**********************************//
#include <fftw3.h>
#include "mkl.h"
#include <omp.h>
#include <armadillo>
//**********************************//
#include "tensor.h"
#include "tensor.cpp"
#include "cp_als.cpp"
#include "tucker_hosvd.cpp"
//**********************************//

double gettime(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*1000+tv.tv_usec/1000.0; //time:s
};

using namespace std;
using namespace arma;

int main() {
    double t0,t1;
    int I=3;
    int rank=0.2*I; rank=1;

    Tensor<float> a(I,I,I);
    //initialization
    for (int i = 0; i < a.n1; ++i) {
        for (int j = 0; j < a.n2; ++j) {
            for(int k=0; k< a.n3; ++k) {
                a(i,j,k) = randu<float>();
            }
        }
    }

    t0=gettime();
    tucker_core<float> result_tucker;
    result_tucker = hosvd(a,rank,rank,rank);
    t1=gettime();
    cout << "time:" <<t1-t0 <<endl;

    t0=gettime();
    cp_mats<float> result;
    result = cp_als(a,rank);
    t1=gettime();
    cout << "time:" <<t1-t0 <<endl;

    return 0;

}