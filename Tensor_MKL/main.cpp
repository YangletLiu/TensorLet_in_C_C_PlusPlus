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
    mkl_free(cc);

    Tensor3D<double> vd(a);
    cout << vd.getsize()[1] << endl;

    MKL_INT k = 2;
    b = k*a;
    b = a*b;
//    b = a+b+a;

    t0=gettime();
    double * p = (double*)mkl_calloc(1000000,sizeof(double),64);
    p = a.tens2mat(p,1);  //函数内声明后 调用
    t1=gettime();
    cout << "time:" <<t1-t0 <<endl;
    mkl_free(p);

    //随机数生成
    t0=gettime();
    double r[1000000];
    VSLStreamStatePtr stream;
    vslNewStream(&stream,VSL_BRNG_MCG31, 1);
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,1000000,r,0,1);
    cout << r[999999] << " " << r[1000000]<< endl;
//    for (int i=1;i<2;i++){
//        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,100,r,0,1);
////        cout << r[0] << endl;
//        printf("%e \n",r[0]);
//    }

    vslDeleteStream(&stream);
    t1=gettime();
    cout << "time:" <<t1-t0 <<endl;

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

    cout << "hello" << endl;

    return 0;
}