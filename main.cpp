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

    MKL_INT rank = 0.1*n1+2;
    cout << "rank: " << rank << endl;

//    rank = 4;
//    a.cp_gen(rank);

    MKL_INT r1 = 0.1*n1+2;
    MKL_INT r2 = 0.1*n1+2;
    MKL_INT r3 = 0.1*n1+2;

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
//    rank = 3;
//    MKL_INT ranks[3] = {rank, rank, rank};
//
//    t0=gettime();
//    tucker_format<double> B = tucker_hosvd( a, rank, rank, rank );
//    t1=gettime();
//    cout << "Tucker time:" << t1 - t0 << endl;


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

    cout << "hello" << endl;

    return 0;
}
