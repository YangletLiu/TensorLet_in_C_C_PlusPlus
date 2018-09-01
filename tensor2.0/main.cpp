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
    Tensor<double> a(20,20,20);
    t0=gettime();
    HOSVD(a,10,10,10);
    t1=gettime();

    cout << "time:" <<t1-t0 <<endl;
    Tensor<double> b(2,3,5),d(5,5,5), z(2,3,4),t(5,5,5);
    cout<<a(1,2,4)<<endl;
    cout<<z(0,1,2)<<endl;

    z=z.zeros(2,3,4);
    cout<<z(0,1,2)+123<<endl;

    a*=1;
    int *c=getsize(b);
    cout<<c[0]<<endl; //tensor大小
    cout<<sizeof(a)<<endl;
    cout<<norm(a)<<endl;
    cout<<a(1,2,4)+3<<endl;

    cout << "Hello, World!" << endl;
    t=tprod(a,d);
    cout<<t(1,2,3)<<endl;
    cout<<norm(a)<<endl;
    cout<<fiber(a,1,2,2)<<endl;
    cout<<slice(a,1,1)(0,0)<<endl;

    return 0;

}