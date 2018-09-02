#include <math.h>
#include <iostream>
#include <armadillo>
#include <vector>

#include <complex>
#include <fftw3.h>
#include "mkl.h"

#include "tensor.h"
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
    Tensor<double> a(3,3,3);
//    Tensor<double> g(1,1,1);
//    mat u1(2,1);
//    mat u2(2,1);
//    mat u3(2,1);
//    tuckercore<double> A{g,u1,u2,u3};
//
    t0=gettime();
        HOSVD(a,1,1,1);
    t1=gettime();

    cout << "time:" <<t1-t0 <<endl;
    cp_als(a,5);
//    Tensor<double> b(2,3,5),d(5,5,5), z(2,3,4),t(5,5,5);
//    cout<<z(0,1,2)<<endl;
//
//    z=z.zeros(2,3,4);
//    cout<<z(0,1,2)+123<<endl;
//
//    int *c=getsize(b);
//    cout<<c[0]<<endl; //tensor大小
//    cout<<sizeof(a)<<endl;
//    cout<<norm(a)<<endl;
//
//    cout << "Hello, World!" << endl;
//    cout<<t(1,2,3)<<endl;
//    cout<<norm(a)<<endl;

//slice
//    cout<<fiber(a,1,2,2)<<endl;
    mat m1 = ten2mat(a,1);
//    mat m2 = ten2mat(a,2);
    mat m3 = ten2mat(a,3);
    cout << m1 << endl;
//    cout<<slice(a,0,3)<<endl;
    cout << m3 << endl;
//    cout << m2 << endl;
    cout<<slice(a,0,2)<<endl;
    return 0;

}