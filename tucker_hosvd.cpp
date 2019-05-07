//
// Created by jcfei on 18-9-26.
//

#include "tensor.h"
#include "Tensor3D.h"
#include "tucker.h"

namespace TensorLet_decomposition {

    template<class datatype>
    tucker_format<datatype> tucker_hosvd( Tensor3D<datatype> &a, int &r1, int &r2, int &r3 ) {

        if( r1 == 0 || r2 == 0 || r3 == 0 ){
            printf("Tucker decomposition ranks cannot be zero.");
            exit(1);
        }

        int *shape = a.size();  //dimension

        int n1 = shape[0]; int n2 =shape[1]; int n3 = shape[2];

        datatype* X1_times_X1T = (datatype*)mkl_malloc(n1 * n1 * sizeof(datatype), 64);
        datatype* X2_times_X2T = (datatype*)mkl_malloc(n2 * n2 * sizeof(datatype), 64);
        datatype* X3_times_X3T = (datatype*)mkl_malloc(n3 * n3 * sizeof(datatype), 64);

        cblas_dsyrk( CblasColMajor, CblasUpper, CblasNoTrans,
                n1,n2 * n3, 1, a.pointer, n1,
                0, X1_times_X1T, n1 );  //x1*x1^t

        for(MKL_INT i = 0; i < n3; i++){
            cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                    n2, n1, 1, a.pointer + i * n1 * n2, n1,
                    1, X2_times_X2T, n2);  // X(2) * X2^t rank update
        }

        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                n3, n1*n2, 1, a.pointer, n1 * n2,
                0, X3_times_X3T, n3);  //x3*x3^t


        int info1, info2, info3;
        double* w1 = (double*)mkl_malloc(n1 * sizeof(double), 64);
        double* w2 = (double*)mkl_malloc(n2 * sizeof(double), 64);
        double* w3 = (double*)mkl_malloc(n3 * sizeof(double), 64);

//        for (int i=0;i<3;i++){
//            for (int j=0;j<3;j++){
//                cout << X3_times_X3T[j+i*n1] << " ";
//            }
//            cout << endl;
//        }

        /* Solve eigenproblem */
        info1 = LAPACKE_dsyevd( LAPACK_COL_MAJOR, 'V', 'U', n1, X1_times_X1T, n1, w1 );

        info2 = LAPACKE_dsyevd( LAPACK_COL_MAJOR, 'V', 'U', n2, X2_times_X2T, n2, w2 );

        info3 = LAPACKE_dsyevd( LAPACK_ROW_MAJOR, 'V', 'U', n3, X3_times_X3T, n3, w3 );

        /* Check for convergence */
        if( info1 > 0 || info2 > 0 || info3 > 0) {
            printf( "The algorithm failed to compute eigenvalues.\n" );
            exit( 1 );
        }

        /* Free workspace */
        MKL_free( (void*)w1 );
        MKL_free( (void*)w2 );
        MKL_free( (void*)w3 );

        double* u1t_times_x1 = (double*)mkl_malloc(r1 * n2 * n3 * sizeof(double), 64);
        double* u2t_times_u1t_times_x1 = (double*)mkl_malloc(r1 * r2 * n3 * sizeof(double), 64);

        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, r1, n2 * n3, n1,
                    1, X1_times_X1T + (n1 - r1) * n1, n1, a.pointer, n1,
                    0, u1t_times_x1, r1); // U1^t * X(1)

        for(MKL_INT i = 0; i < n3; ++i){
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, r1, r2, n2,
                        1, u1t_times_x1 + i * r1 * n2, r1, X2_times_X2T + (n2 - r2) * n2, n2,
                        0, u2t_times_u1t_times_x1 + i * r1 * r2, r1);  // ( U1^t * X(1) )_(1) * U2
        }

        MKL_free(u1t_times_x1);

        double* g = (double*)mkl_malloc(r1 * r2 * r3 * sizeof(double), 64);

        if( g == NULL ){
            printf("Cannot allocate enough memory for core tensor g.");
            exit(1);
        }

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, r1 * r2, r3, n3,
                    1, u2t_times_u1t_times_x1, r1 * r2, X3_times_X3T + (n3 - r3) * n3, n3,
                    0, g, r1 * r2); // g = x3 * u3

        MKL_free(u2t_times_u1t_times_x1);

        tucker_format<datatype> result;
        result.core = g;
        result.u1 = X1_times_X1T;
        result.u2 = X2_times_X2T;
        result.u3 = X3_times_X3T;

        return result;

    }

    template<class datatype>
    tucker_format<datatype> tucker_hosvd( Tensor3D<datatype> &a, int* rank ) {

        if( rank == NULL ){
            printf("Tucker decomposition rank cannot be NULL.");
            exit(1);
        }

        int r1 = rank[0];
        int r2 = rank[1];
        int r3 = rank[2];

        int *shape = a.size();  //dimension

        int n1 = shape[0]; int n2 =shape[1]; int n3 = shape[2];

        datatype* X1_times_X1T = (datatype*)mkl_malloc(n1 * n1 * sizeof(datatype), 64);
        datatype* X2_times_X2T = (datatype*)mkl_malloc(n2 * n2 * sizeof(datatype), 64);
        datatype* X3_times_X3T = (datatype*)mkl_malloc(n3 * n3 * sizeof(datatype), 64);

        cblas_dsyrk( CblasColMajor, CblasUpper, CblasNoTrans,
                     n1,n2 * n3, 1, a.pointer, n1,
                     0, X1_times_X1T, n1 );  //x1*x1^t

        for(MKL_INT i = 0; i < n3; ++i){
            cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                        n2, n1, 1, a.pointer + i * n1 * n2, n1,
                        1, X2_times_X2T, n2);  // X(2) * X2^t rank update
        }

        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                    n3, n1*n2, 1, a.pointer, n1 * n2,
                    0, X3_times_X3T, n3);  //x3*x3^t


        int info1, info2, info3;
        double* w1 = (double*)mkl_malloc(n1 * sizeof(double), 64);
        double* w2 = (double*)mkl_malloc(n2 * sizeof(double), 64);
        double* w3 = (double*)mkl_malloc(n3 * sizeof(double), 64);

//        for (int i=0;i<20;++i){
//            for (int j=0;j<20;j++){
//                cout << X3_times_X3T[j+i*n1] << " ";
//            }
//            cout << endl;
//        }

        /* Solve eigenproblem */
        info1 = LAPACKE_dsyevd( LAPACK_COL_MAJOR, 'V', 'U', n1, X1_times_X1T, n1, w1 );

        info2 = LAPACKE_dsyevd( LAPACK_COL_MAJOR, 'V', 'U', n2, X2_times_X2T, n2, w2 );

        info3 = LAPACKE_dsyevd( LAPACK_ROW_MAJOR, 'V', 'U', n3, X3_times_X3T, n3, w3 );

        /* Check for convergence */
        if( info1 > 0 || info2 >0 || info3 >0) {
            printf( "The algorithm failed to compute eigenvalues.\n" );
            exit( 1 );
        }

        double* u1t_times_x1 = (double*)mkl_malloc(r1 * n2 * n3 * sizeof(double), 64);

        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, r1, n2 * n3, n1,
                    1, X1_times_X1T + (n1 - r1) * n1, n1, a.pointer, n1,
                    0, u1t_times_x1, r1); // U1^t * X(1)


        double* u2t_times_u1t_times_x1 = (double*)mkl_malloc(r1 * r2 * n3 * sizeof(double), 64);

        for(MKL_INT i = 0; i < n3; ++i){
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, r1, r2, n2,
                        1, u1t_times_x1 + i * r1 * n2, r1, X2_times_X2T + (n2 - r2) * n2, n2,
                        0, u2t_times_u1t_times_x1 + i * r1 * r2, r1);  // ( U1^t * X(1) )_(1) * U2
        }

        MKL_free(u1t_times_x1);


        double* g = (double*)mkl_malloc(r1 * r2 * r3 * sizeof(double), 64);

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, r1 * r2, r3, n3,
                    1, u2t_times_u1t_times_x1, r1 * r2, X3_times_X3T + (n3 - r3) * n3, n3,
                    0, g, r1 * r2); // g = x3 * u3

        MKL_free(u2t_times_u1t_times_x1);

        /* Free workspace */
        MKL_free( (void*)w1 );
        MKL_free( (void*)w2 );
        MKL_free( (void*)w3 );

        tucker_format<datatype> result;
        result.core = g;
        result.u1 = X1_times_X1T;
        result.u2 = X2_times_X2T;
        result.u3 = X3_times_X3T;

        return result;

    }
}


