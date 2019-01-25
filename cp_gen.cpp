//
// Created by jcfei on 18-9-28.
//

#include "tensor.h"
#include "Tensor3D.h"

using namespace std;

template<class datatype>
Tensor3D<datatype>& Tensor3D<datatype>::cp_gen(int r) {

    MKL_INT n1 = this->shape[0];
    MKL_INT n2 = this->shape[1];
    MKL_INT n3 = this->shape[2];

    datatype* A = (datatype*)mkl_malloc(n1 * r * sizeof(datatype), 64);
    datatype* B = (datatype*)mkl_malloc(n2 * r * sizeof(datatype), 64);
    datatype* C = (datatype*)mkl_malloc(n3 * r * sizeof(datatype), 64);

    if( A == NULL || B == NULL || C == NULL ){
        printf("Cannot allocate enough memory for A, B, C.");
        exit(1);
    }

    MKL_INT status[3]; // random state

    VSLStreamStatePtr stream;

// The seed need to test
    srand((unsigned)time(NULL));
    MKL_INT SEED = rand();

    vslNewStream(&stream,VSL_BRNG_MCG31, SEED);

    status[0] = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n1 * r, A, 0, 1);

    status[1] = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n2 * r, B, 0, 1);

    status[2] = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n3 * r, C, 0, 1);

    vslDeleteStream(&stream);

    if( status[0] + status[1] +status[2] != 0){
        printf("Random initialization failed for A, B, C.");
        exit(1);
    }

    datatype* b_kr_a = ( datatype* )mkl_malloc( n1 * n2 * r * sizeof( datatype ), 64 );

    if( b_kr_a == NULL ){
        printf("Cannot allocate enough memory for kr product.");
        exit(1);
    }

    for( MKL_INT i = 0; i < r; i++ ){
        cblas_dgemm( CblasColMajor, CblasNoTrans, CblasTrans,
                     n1, n2, 1, 1, A + i * n1, n1, B + i * n2, n2,
                     0, b_kr_a + i * n1 * n2, n1 );  // kr(b,a)
    }

    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                 n1 * n2, n3, r, 1, b_kr_a, n1 * n2, C, r,
                 0, this->pointer, n1 * n2 );

    MKL_free( A );
    MKL_free( B );
    MKL_free( C );
    MKL_free( b_kr_a );

    return *this;

}