//
// Created by jcfei on 18-9-26.
//

#include "tensor.h"
#include "Tensor3D.h"

template<class datatype>
class tucker_format{
    Tensor3D<datatype>* core;
    datatype* u1,u2,u3;
};

namespace TensorLet_decomposition {

    template<class datatype>
    tucker_format<datatype> tucker_hosvd(Tensor3D<datatype> &a, int r1, int r2, int r3) {
        int *shape = a.getsize();  //dimension
        int n1 = shape[0]; int n2 =shape[1]; int n3 = shape[2];

        datatype* X1_times_X1T = (datatype*)mkl_malloc(n1*n1*sizeof(datatype),64);
        datatype* X2_times_X2T = (datatype*)mkl_malloc(n2*n2*sizeof(datatype),64);
        datatype* X3_times_X3T = (datatype*)mkl_malloc(n3*n3*sizeof(datatype),64);

//        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n1,n1,n2*n3,1,a.pointer,n1,a.pointer,n1,0,X1_times_X1T,n1);
        cblas_dsyrk(CblasColMajor,CblasUpper,CblasNoTrans,n1,n2*n3,1,a.pointer,n1,0,X1_times_X1T,n1);  //x1*x1^t

        for(MKL_INT i=0;i<n3;i++){
            cblas_dsyrk(CblasColMajor,CblasUpper,CblasTrans, n2, n1, 1, a.pointer+i*n1*n2, n1, 1, X2_times_X2T,n2);  // X(2) * X2^t rank update
        }

        cblas_dsyrk(CblasRowMajor,CblasUpper,CblasNoTrans,n3,n3,1,a.pointer,n1,0,X3_times_X3T,n1);  //x3*x3^t

        int info1, info2, info3;
        double* w1 = (double*)mkl_malloc(n1*sizeof(double),64);
        double* w2 = (double*)mkl_malloc(n1*sizeof(double),64);
        double* w3 = (double*)mkl_malloc(n1*sizeof(double),64);

        /* Solve eigenproblem */
        info1 = LAPACKE_dsyevd( LAPACK_COL_MAJOR, 'V', 'U', n1, X1_times_X1T, n1, w1 );

        info2 = LAPACKE_dsyevd( LAPACK_COL_MAJOR, 'V', 'U', n2, X2_times_X2T, n2, w2 );

        info3 = LAPACKE_dsyevd( LAPACK_ROW_MAJOR, 'V', 'U', n3, X3_times_X3T, n3, w3 );

        /* print eigenvalue */
//        for(int i = 0; i < n1*n1; i++ ) {
//            printf( " %6.2f \n", X1_times_X1T[i] );
//        }
//        for(int i = 0; i < n1; i++ ) {
//            printf( " %7.2f, %7.2f, %7.2f \n", w1[i], w2[i], w3[i]);
//        }

        /* Check for convergence */
        if( info1 > 0 || info2 >0 || info3 >0) {
            printf( "The algorithm failed to compute eigenvalues.\n" );
            exit( 1 );
        }

        /* Free workspace */
        MKL_free( (void*)w1 );
        MKL_free( (void*)w2 );
        MKL_free( (void*)w3 );

        tucker_format<datatype> result;
//        result.tucker_u1 = A;
//        result.tucker_u2 = B;
//        result.tucker_u3 = C;


//        int count = 0;
//        for (int i=0;i<n1*n1;i++){
//            if(X2_times_X2T[i] == 0){ count++; }
//        }
//        for (int i=0;i<n1;i++){
//            cout << X2_times_X2T[i+(n1-2)*n1] << endl;
//        }
//        cout << " count " << count << " " << n1*n1<< endl;

        return result;
    }

}

//    datatype* X1_times_X1T = (datatype*)mkl_malloc(n1*n1*sizeof(datatype),64);
//    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n1,n1,n2*n3,1,a.pointer,n1,a.pointer,n1,0,X1_times_X1T,n1);
//    cout << "A_times_B: " << X1_times_X1T[0] << endl;