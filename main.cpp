#include <math.h>
#include <iostream>
#include <armadillo>
#include <vector>
#include <complex>
#include <fftw3.h>
#include "mkl.h"


#include <omp.h>

#include "tensor.h"
#include "tensor.cpp"
#include "cp_als.cpp"
#include "cpals.cpp"
#include "tsvd.cpp"
#include "HOSVD.cpp"
#include "tucker_hosvd.cpp"


using namespace std;

#include "time.h"
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
    int R=0.2*I;
    R=1;
    //    arma_rng::set_seed(1);

    t0=gettime();
    Tensor<float> a(I,I,I);
    t1=gettime();
    cout << "time:" <<t1-t0 <<endl;

    for (int i = 0; i < a.n1; ++i) {
        for (int j = 0; j < a.n2; ++j) {
            for(int k=0; k< a.n3; ++k) {
                a(i,j,k) = randu<float>();
            }
        }
    }
//    arma_rng::set_seed(1);
//    Tensor<float> b(I,I,I);
//    for (int i = 0; i < b.n1; ++i) {
//        for (int j = 0; j < b.n2; ++j) {
//            for(int k=0; k<b.n3; ++k) {
//                b(i,j,k) = randu<float>();
////                cout << a(i,j,k) - b(i,j,k) << endl;
//            }
//        }
//    }

    Tensor<double> b(I,I,I);

//    t0=gettime();
//    tucker_core<float> result;
//    result = hosvd(a,R,R,R);
//    t1=gettime();
//    cout << "time:" <<t1-t0 <<endl;

    t0=gettime();
    cp_mats<float> result;
    result = cp_als(a,R);
    t1=gettime();
    cout << "time:" <<t1-t0 <<endl;

//    t0=gettime();
//        HOSVD(a,R,R,R);
//        hosvd(a,R,R,R);
//        cp_als(a,R);
//        cpals(a,R);
//    t1=gettime();
//    cout << "time:" <<t1-t0 <<endl;

//    Tensor<double> b(2,3,5),d(5,5,5), z(2,3,4),t(5,5,5);
//    cout<<z(0,1,2)<<endl;
//    z=z.zeros(2,3,4);
//    cout<<z(0,1,2)+123<<endl;
//
//    int *c=getsize(b);
//    cout<<c[0]<<endl; //tensor大小
//    cout<<sizeof(a)<<endl;
//    cout<<norm(a)<<endl;
//    cout<<t(1,2,3)<<endl;

//    mat m1 = ten2mat(a,1);
//    cout << m1 << endl;
//    mat m2 = ten2mat(a,2);
//    cout << m2 << endl;
//    mat m3 = ten2mat(a,3);,
//    cout << m3 << endl;

//    cout<<slice(a,0,2)<<endl;
//slice
//    cout<<fiber(a,1,2,2)<<endl;

    return 0;

}