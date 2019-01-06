#include "tensor.h"
#include "runningtime.h"

#include "ten2mat.cpp"
#include "cpgen.cpp"
#include "cp_als.cpp"
#include "tucker_hosvd.cpp"
#include "tensor_hooi.cpp"
#include "t_svd.cpp"

#include "Tensor3D.h"
#include "Tensor3D.cpp"

#include <stdio.h>
#include <mkl.h>

using namespace std;

int main(){
    MKL_INT n1,n2,n3;
    MKL_INT aa[3]={10,10,10};

    double t0,t1;

    t0=gettime();
    Tensor3D<double> a(100,100,100); //element
    t1=gettime();
    cout << "time:" <<t1-t0 <<endl;

    cout << sizeof(a) << endl;
    cout << a.getsize()[2] << endl;
    cout << "element a: " << a(2,2,2) << endl;
    cout << "norm: " << a.frobenius_norm() << endl;

    Tensor3D<double> b(aa); // int array
    b = a;
    b += a;
    cout << b.getsize()[0] << endl;

    cout << "n" << b.getsize()[2];

    double *cc = (double*)mkl_calloc(10000,sizeof(double),64);  //返回成功为1
    cout << sizeof(cc) << endl;
    cout << cc << endl;

    Tensor3D<double> vd(a);
    cout << vd.getsize()[1];


    MKL_INT k = 2;
    b = k*a;
    b = a*b;
//    b = a+b+a;
    double * p;
    p = a.tens2mat(1);

    cout << "hello" << endl;

    return 0;
}