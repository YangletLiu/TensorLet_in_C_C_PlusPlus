//
// Created by jcfei on 19-2-18.
//
#include "tensor.h"
#include "pinv.h"

double *pinv(double *a, double *result, int r) {
    MKL_INT  m = r;
    MKL_INT  n = r;
    MKL_INT  k = min(m,n);
    MKL_INT  lda = m; //column-major, n for row-major
    MKL_INT  ldu = m; //column-major, min(m,n) for row-major
    MKL_INT  ldvt = k; //column-major, n for row-major

    MKL_INT  lwork;
    MKL_INT info;
    double wkopt;
    double * work;
    char jobu='S';
    char jobvt= 'S';

    double* s=(double*)malloc(k*sizeof(double));
    double* u = (double*)malloc(ldu*k*sizeof(double));
    double* vt = (double*)malloc(ldvt*n*sizeof(double));

    lwork = -1;
    dgesvd( &jobu, &jobvt, &m, &n, &a[0], &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork, &info );
    lwork = (MKL_INT)wkopt;
    work = (double*)malloc( lwork*sizeof(double) );

    /* Compute SVD */
    dgesvd( &jobu, &jobvt, &m, &n, &a[0], &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info );
    /* Check for convergence */
    if( info > 0 ) {
        printf( "The algorithm computing SVD failed to converge.\n" );
        exit( 1 );
    }

    //u=(s^-1)*U
    MKL_INT incx = 1;
//#pragma omp parallel for
    for(int i=0; i<k; i++)
    {
        double ss;
        if(s[i] > 1.0e-9)
            ss=1.0/s[i];
        else
            ss=s[i];
        dscal(&m, &ss, &u[i*m], &incx);
    }

    //inv(A)=(Vt)^T *u^T
    double alpha=1.0, beta=0.0;

    MKL_INT ld_inva=n;

    dgemm( "T", "T", &n, &m, &k, &alpha, vt, &ldvt, u, &ldu, &beta, result, &ld_inva);

//    for(int i = 0; i < r*r; i++){
//        cout << result[i] << "," ;
//    }
//    cout << endl;

    return  result;
}
