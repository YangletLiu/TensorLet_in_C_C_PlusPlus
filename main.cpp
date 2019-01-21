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
using namespace TensorLet_decomposition;

int main(){
    MKL_INT n1,n2,n3;
    n1=n2=n3=200;
//    n1=100;
//    n2=200;
//    n3=300;
    double t0,t1;
    t0=gettime();
    Tensor3D<double> a(n1,n2,n3); //element
    t1=gettime();
    cout << "Memory malloc time:" <<t1-t0 <<endl;

    t0=gettime();
    a.random_tensor();
    t1=gettime();
    cout << "Random initialize time:" <<t1-t0 <<endl;

    t0=gettime();
    cp_format<double> A = cp_als(a, 40);
    t1=gettime();
    cout << "CP time:" <<t1-t0 <<endl;

    MKL_free(A.cp_A);
    MKL_free(A.cp_B);
    MKL_free(A.cp_C);

    t0=gettime();
    tucker_format<double> B = tucker_hosvd(a,80,80,80);
    t1=gettime();
    cout << "Tucker time:" <<t1-t0 <<endl;

    MKL_free(B.core);
    MKL_free(B.u1);
    MKL_free(B.u2);
    MKL_free(B.u3);


    t0=gettime();
    tsvd_format<double> C = tsvd(a);
    t1=gettime();
    cout << "tsvd time:" <<t1-t0 <<endl;




//    t0=gettime();
//    double* X1_times_X1T = (double*)mkl_malloc(n1*n1*sizeof(double),64);
//    double* X2_times_X2T = (double*)mkl_malloc(n1*n1*sizeof(double),64);
//    double* X3_times_X3T = (double*)mkl_malloc(n1*n1*sizeof(double),64);
//
//    cblas_dsyrk(CblasColMajor,CblasUpper,CblasNoTrans,n1,n2*n3,1,a.pointer,n1,0,X1_times_X1T,n1);  //x1*x1^t
//    for(MKL_INT i=0;i<n3;i++){
//        cblas_dsyrk(CblasColMajor,CblasUpper,CblasTrans, n2, n1, 1, a.pointer+i*n1*n2, n1, 1, X2_times_X2T,n2);  // X(2) * X2^t rank update
//    }
//    cblas_dsyrk(CblasRowMajor,CblasUpper,CblasNoTrans,n3,n1*n2,1,a.pointer,n1*n2,0,X3_times_X3T,n3);  //x3*x3^t
//
//    t1=gettime();
//    cout << "Tucker time:" <<t1-t0 <<endl;

    cout << "hello" << endl;

    return 0;
}

//随机数生成
//t0=gettime();
//VSLStreamStatePtr stream;
//vslNewStream(&stream,VSL_BRNG_MCG31, 1);
//vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,10000000000,a.pointer,0,1);
//cout << a.pointer[999999999] << endl;

//    for (int i=1;i<2;i++){
//        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,100,r,0,1);
////        cout << r[0] << endl;
//        printf("%e \n",r[0]);
//    }
//vslDeleteStream(&stream);
//t1=gettime();
//cout << "time:" <<t1-t0 <<endl;

//how to calloc memory

//    double *cc = (double*)mkl_calloc(10000,sizeof(double),64);  //返回成功为1
//    cout << sizeof(cc) << endl;
//    cout << cc << endl;
//    mkl_free(cc);

// how to free memory
//    t0=gettime();
//    double * p = (double*)mkl_calloc(1000000,sizeof(double),64);
//    p = a.tens2mat(p,1);  //函数内声明后 调用
//    t1=gettime();
//    cout << "time:" <<t1-t0 <<endl;
//    mkl_free(p);   //必须 mkl_free


//    MKLVersion Version;
//    mkl_get_version(&Version);
//    printf("Major version: %d\n",Version.MajorVersion);
//    printf("Minor version: %d\n",Version.MinorVersion);
//    printf("Update version: %d\n",Version.UpdateVersion);
//    printf("Product status: %s\n",Version.ProductStatus);
//    printf("Build: %s\n",Version.Build);
//    printf("Platform: %s\n",Version.Platform);
//    printf("Processor optimization: %s\n",Version.Processor);
//    printf("================================================================\n");
//    printf("\n");

/*******************************
//basic operation test
*******************************/
//MKL_INT n1,n2,n3;
//n1=n2=n3=100;
//
//double t0,t1;
//t0=gettime();
//Tensor3D<double> a(n1,n2,n3); //element
//t1=gettime();
//cout << "Initialize time:" <<t1-t0 <<endl;
//cout << "size of class: " << sizeof(a) << endl;
//
//int *p1;
//p1 = a.getsize();
//cout << "shape: " << p1[2] << endl;
//cout << "element a: " << a(2,2,2) << endl;
//cout << "norm: " << a.frobenius_norm() << endl;
//
//MKL_INT aa[3]={10,10,10};
//Tensor3D<double> b(aa); // int array
////    b = a;
////    b += a;
//cout << "shape: " << b.getsize()[2] << '\n';
//b.random_tensor();
//for (int i = 0; i<1000; i++){
//cout << b.pointer[i] << endl;
//}
//
//Tensor3D<double> vd(a);
//cout << "shape: " << vd.getsize()[1] << endl;
//
//MKL_INT k = 2;
////    b = k*b;
////    b = a*b;
////    b = a+b+a;
//
//t0=gettime();
//a.random_tensor();
//t1=gettime();
//cout << "random time:" <<t1-t0 <<endl;
//cout << a.pointer[0] << endl;
//cout << a.pointer[1] << endl;
//cout << a.pointer[997002998] << endl;
//cout << a.pointer[999999999] << endl;
//
//cout << "norm: " << a.frobenius_norm() << endl;

//    for (int i = 0; i<n1*n2*n3; i++){
//        cout << a.pointer[i] << endl;
//    }

// 对称乘积
//double* tmp = (double*)mkl_malloc(n1*n1*sizeof(double),64);
//t0=gettime();
////    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n1,n1,n2*n3,1,a.pointer,n1,a.pointer,n1,0,tmp,n1);
//cblas_dsyrk(CblasColMajor,CblasUpper,CblasNoTrans,n1,n2*n3,1,a.pointer,n1,0,tmp,n1);
//t1=gettime();
//    cout << tmp[0] << endl;
//    cout << tmp[1] << endl;
//    cout << tmp[2] << endl;
//    cout << tmp[n1] << endl;
//    cout << tmp[n1+1] << endl;
//cout << "Product time:" <<t1-t0 <<endl;
