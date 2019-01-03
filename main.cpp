#include "tensor.h"
#include "runningtime.h"

#include "ten2mat.cpp"
#include "cpgen.cpp"
#include "cp_als.cpp"
#include "tucker_hosvd.cpp"
#include "tensor_hooi.cpp"
#include "t_svd.cpp"

#include "Tensor3D.h"
#include <stdio.h>
#include <mkl.h>

using namespace std;

int main(){

    MKL_INT n1,n2,n3;
    double t0,t1;
    int aa[3]={10,10,10};

    double ccc =0.05;
    cout << "inhao" << endl;

    t0=gettime();
    Tensor3D<double> a(100,100,100);

    Tensor3D<double> b(aa);
    t1=gettime();
    cout << "time:" <<t1-t0 <<endl;

    cout << sizeof(a) << endl;
    cout << "n" << b.getsize()[2];
    double *cc = (double*)mkl_calloc(100000,sizeof(double),64);  //返回成功为1

    cout << "sizeof(double)" << sizeof(double) << endl;

    cout << sizeof(cc) << endl;
    cout << cc << endl;


    cout << a.getsize()[2];

    Tensor3D<double> vd(a);
    cout << vd.getsize()[1];
    return 0;
}

//int main(void) {
//    double *a, *b, *c;
//    int n, i;
//    double alpha, beta;
//    MKL_INT64 AllocatedBytes;
//    int N_AllocatedBuffers;
//    alpha = 1.1; beta = -1.2;
//    n = 1000;
//    mkl_peak_mem_usage(MKL_PEAK_MEM_ENABLE);
//    a = (double*)mkl_malloc(n*n*sizeof(double),64);
//    b = (double*)mkl_malloc(n*n*sizeof(double),64);
//    c = (double*)mkl_calloc(n*n,sizeof(double),64);
//    for (i=0;i<(n*n);i++) {
//        a[i] = (double)(i+1);
//        b[i] = (double)(-i-1);
//    }
//    cout << a << endl;
//
//    dgemm("N","N",&n,&n,&n,&alpha,a,&n,b,&n,&beta,c,&n);
//    AllocatedBytes = mkl_mem_stat(&N_AllocatedBuffers);
//    printf("\nDGEMM uses %d bytes in %d buffers",AllocatedBytes,N_AllocatedBuffers);
//    mkl_free_buffers();
//    mkl_free(a);
//    mkl_free(b);
//    mkl_free(c);
//    AllocatedBytes = mkl_mem_stat(&N_AllocatedBuffers);
//    if (AllocatedBytes > 0) {
//        printf("\nMKL memory leak!");
//        printf("\nAfter mkl_free_buffers there are %d bytes in %d buffers",
//               AllocatedBytes,N_AllocatedBuffers);
//    }
//
//    return 0;
//}