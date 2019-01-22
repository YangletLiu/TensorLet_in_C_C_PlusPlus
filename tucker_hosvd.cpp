//
// Created by jcfei on 18-9-26.
//

#include "tensor.h"
#include "Tensor3D.h"

template<class datatype>
class tucker_format{
public:
    datatype* core;
    datatype* u1;
    datatype* u2;
    datatype* u3;
};

namespace TensorLet_decomposition {

    template<class datatype>
    tucker_format<datatype> tucker_hosvd( Tensor3D<datatype> &a, int &r1, int &r2, int &r3) {
        int *shape = a.getsize();  //dimension

        int n1 = shape[0]; int n2 =shape[1]; int n3 = shape[2];

        datatype* X1_times_X1T = (datatype*)mkl_malloc(n1 * n1 * sizeof(datatype), 64);
        datatype* X2_times_X2T = (datatype*)mkl_malloc(n2 * n2 * sizeof(datatype), 64);
        datatype* X3_times_X3T = (datatype*)mkl_malloc(n3 * n3 * sizeof(datatype), 64);

//        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n1,n1,n2*n3,1,a.pointer,n1,a.pointer,n1,0,X1_times_X1T,n1);

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

//        for (int i=0;i<20;i++){
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

        for(MKL_INT i = 0; i < n3; i++){
            cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, r2, r1, n2,
                    1, X2_times_X2T + (n2 - r2) * n2, n2, u1t_times_x1 + i * r1 * n2, r2,
                    0, u2t_times_u1t_times_x1 + i * r1 * r2, r1);  // U2^t * U1^t * X(1)
        }

        MKL_free(u1t_times_x1);


        double* g = (double*)mkl_malloc(r1 * r2 * r3 * sizeof(double), 64);

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, r3, r1 * r2, n3,
                    1, X3_times_X3T + (n3 - r3) * n3, r3, u2t_times_u1t_times_x1, r1 * r2,
                    0, g, r1 * r2); // g

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

//        result.tucker_u1 = A;
//        result.tucker_u2 = B;
//        result.tucker_u3 = C;


//        int count = 0;
//        for (int i=0;i<n1*n1;i++){
//            if(X2_times_X2T[i] == 0){ count++; }
//        }

//        cout << " count " << count << " " << n1*n1<< endl;

//        for (int i=0;i<20;i++){
//            for (int j=0;j<20;j++){
//                cout << X3_times_X3T[j+i*n1] << " ";
//            }
//            cout << endl;
//        }

        /* print eigenvalue */
//        for(int i = 0; i < n1*n1; i++ ) {
//            printf( " %6.5f \n", X3_times_X3T[i] );
//        }
//        for(int i = 0; i < n1; i++ ) {
//            printf( " %7.2f, %7.2f, %7.2f \n", w1[i], w2[i], w3[i]);
//        }
    }

    template<class datatype>
    tucker_format<datatype> tucker_hosvd( Tensor3D<datatype> &a, int* rank) {
        int r1 = rank[0];
        int r2 = rank[1];
        int r3 = rank[2];

        int *shape = a.getsize();  //dimension

        int n1 = shape[0]; int n2 =shape[1]; int n3 = shape[2];

        datatype* X1_times_X1T = (datatype*)mkl_malloc(n1 * n1 * sizeof(datatype), 64);
        datatype* X2_times_X2T = (datatype*)mkl_malloc(n2 * n2 * sizeof(datatype), 64);
        datatype* X3_times_X3T = (datatype*)mkl_malloc(n3 * n3 * sizeof(datatype), 64);

//        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n1,n1,n2*n3,1,a.pointer,n1,a.pointer,n1,0,X1_times_X1T,n1);

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

//        for (int i=0;i<20;i++){
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

        for(MKL_INT i = 0; i < n3; i++){
            cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, r2, r1, n2,
                        1, X2_times_X2T + (n2 - r2) * n2, n2, u1t_times_x1 + i * r1 * n2, r2,
                        0, u2t_times_u1t_times_x1 + i * r1 * r2, r1);  // U2^t * U1^t * X(1)
        }

        MKL_free(u1t_times_x1);


        double* g = (double*)mkl_malloc(r1 * r2 * r3 * sizeof(double), 64);

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, r3, r1 * r2, n3,
                    1, X3_times_X3T + (n3 - r3) * n3, r3, u2t_times_u1t_times_x1, r1 * r2,
                    0, g, r1 * r2); // g

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

//        result.tucker_u1 = A;
//        result.tucker_u2 = B;
//        result.tucker_u3 = C;


//        int count = 0;
//        for (int i=0;i<n1*n1;i++){
//            if(X2_times_X2T[i] == 0){ count++; }
//        }

//        cout << " count " << count << " " << n1*n1<< endl;

//        for (int i=0;i<20;i++){
//            for (int j=0;j<20;j++){
//                cout << X3_times_X3T[j+i*n1] << " ";
//            }
//            cout << endl;
//        }

        /* print eigenvalue */
//        for(int i = 0; i < n1*n1; i++ ) {
//            printf( " %6.5f \n", X3_times_X3T[i] );
//        }
//        for(int i = 0; i < n1; i++ ) {
//            printf( " %7.2f, %7.2f, %7.2f \n", w1[i], w2[i], w3[i]);
//        }
    }

}

//    datatype* X1_times_X1T = (datatype*)mkl_malloc(n1*n1*sizeof(datatype),64);
//    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n1,n1,n2*n3,1,a.pointer,n1,a.pointer,n1,0,X1_times_X1T,n1);
//    cout << "A_times_B: " << X1_times_X1T[0] << endl;