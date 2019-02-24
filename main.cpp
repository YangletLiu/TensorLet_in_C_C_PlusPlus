#include "tensor.h"
#include "runningtime.h"

#include "ten2mat.cpp"
#include "cp_als.cpp"
#include "tucker_hosvd.cpp"
#include "tensor_hooi.cpp"
#include "t_svd.cpp"
#include "tensor_train.cpp"
#include "cp_gen.cpp"
#include "mode_n_product.cpp"

#include "Tensor3D.h"
#include "Tensor3D.cpp"

#include <stdio.h>
#include <mkl.h>

using namespace std;
using namespace TensorLet_decomposition;



int main(){

    MKL_INT n1, n2, n3;
//    n1=7; n2=8; n3 = 9;
    n1 = n2 = n3 = 3;
//    n3 = 256;

    double t0,t1;
    t0=gettime();
    Tensor3D<double> a( n1, n2, n3 ); //element
    t1=gettime();
    cout << "Memory malloc time:" << t1 - t0 << endl;

    t0=gettime();
    a.random_tensor();
    t1=gettime();
    cout << "Random initialize time:" << t1 - t0 << endl;

//    for(int i = 0; i < 27; i++){
//        cout << a.pointer[i] << "," ;
//    }
//    cout << endl;

    MKL_INT rank = 0.2*n1;
    cout << "rank: " << rank << endl;

//    rank = 4;
//    a.cp_gen(rank);



/*******************************
             CP
*******************************/
//    t0=gettime();
//    cp_format<double> A = cp_als( a, rank , 1);
//    t1=gettime();
//    cout << "CP time:" << t1 - t0 << endl;
//
//    MKL_free( A.cp_A );
//    MKL_free( A.cp_B );
//    MKL_free( A.cp_C );




/*******************************
            Tucker
*******************************/
    rank = 3;
    MKL_INT ranks[3] = {rank, rank, rank};

    t0=gettime();
    tucker_format<double> B = tucker_hosvd( a, rank, rank, rank );
    t1=gettime();
    cout << "Tucker time:" << t1 - t0 << endl;


//    t0=gettime();
//    tucker_format<double> B1 = tucker_hosvd( a, ranks );
//    t1=gettime();
//    cout << "Tucker time:" << t1 - t0 << endl;
//
//    MKL_free( B.core );
//    MKL_free( B.u1 );
//    MKL_free( B.u2 );
//    MKL_free( B.u3 );




/*******************************
        t-SVD
*******************************/
//    t0=gettime();
//    tsvd_format<double> C = tsvd( a );
//    t1=gettime();
//    cout << "tsvd time:" << t1-t0 <<endl;



/*******************************
        tensor-train
*******************************/
//    t0=gettime();
//    tt_format<double> D = tensor_train( a );
//    t1=gettime();
//    cout << "tensor-train time:" << t1-t0 <<endl;



/*******************************
        mode n product
*******************************/
//    VSLStreamStatePtr stream;
//
//    srand((unsigned)time(NULL));
//    MKL_INT SEED = rand();
//    MKL_INT status[3];
//    vslNewStream(&stream,VSL_BRNG_MCG31, SEED);
//
//    double* X1_times_X1T = (double*)mkl_malloc(n1 * n1 * sizeof(double), 64);
//    double* X2_times_X2T = (double*)mkl_malloc(n2 * n2 * sizeof(double), 64);
//    double* X3_times_X3T = (double*)mkl_malloc(n3 * n3 * sizeof(double), 64);
//
//    status[0] = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n1 * n1, X1_times_X1T, 0, 1);
//    status[1] = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n2 * n2, X2_times_X2T, 0, 1);
//    status[2] = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n3 * n3, X3_times_X3T, 0, 1);
//
//    vslDeleteStream(&stream);
//
//    double* u1t_times_x1 = (double*)mkl_malloc(n1 * n2 * n3 * sizeof(double), 64);
//    t0=gettime();
//    a.mode_n_product(a.pointer, u1t_times_x1, 1);
//    t1=gettime();
//    cout << "U1X1 time: " << t1 - t0 << endl;
//    MKL_free(u1t_times_x1);
//
//
//    double* u2t_times_u1t_times_x1 = (double*)mkl_malloc(n1 * n2 * n3 * sizeof(double), 64);
//
//    t0=gettime();
//    a.mode_n_product(X2_times_X2T,u2t_times_u1t_times_x1,2);
//    mode_n_product(a,X2_times_X2T,u2t_times_u1t_times_x1,2);
//    t1=gettime();
//    cout << "U2X2 time: " << t1 - t0 << endl;
//
//    MKL_free(u2t_times_u1t_times_x1);
//
//    double* u3t_times_u1t_times_x1 = (double*)mkl_malloc(n1 * n2 * n3 * sizeof(double), 64);
//    t0=gettime();
//    a.mode_n_product(X3_times_X3T,u3t_times_u1t_times_x1,2);
//    mode_n_product(a,X3_times_X3T,u3t_times_u1t_times_x1,2);
//    t1=gettime();
//    cout << "U3X3 time: " << t1 - t0 << endl;
//
//    double norm_a = cblas_dnrm2(n1 * n2 * n3, a.pointer, 1);
//    double norm_u = cblas_dnrm2(n1 * n2 , X2_times_X2T, 1);
//    cout << norm_a << " " << norm_u << endl;

    cout << "hello" << endl;

    return 0;
}
