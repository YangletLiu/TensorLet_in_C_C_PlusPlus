//
// Created by jcfei on 19-1-9.
//

#include "tensor.h"
#include "Tensor3D.h"

template<class datatype>
class tt_format{
    Tensor3D<datatype> g;
    datatype* G1,G2;
};

namespace TensorLet_decomposition {

    template<class datatype>
    tt_format<datatype> tensor_train( Tensor3D<datatype>& a) {

        //dimension
        int *shape = a.getsize();

        int n1 = shape[0]; int n2 =shape[1]; int n3 = shape[2];


        MKL_INT info;

        MKL_INT row = n1;
        MKL_INT column = n2 * n3;
        MKL_INT min_row_column = min( row, column );

        double* s = (double*)MKL_malloc(min_row_column * sizeof(double), 64);
        double* u = (double*)MKL_malloc( row * row * sizeof(double), 64);
        double* vt = (double*)MKL_malloc(min_row_column * column * sizeof(double), 64);


        double super[min_row_column-1];

        info = LAPACKE_dgesvd( LAPACK_COL_MAJOR, 'A', 'S', row, column, a.pointer, row, s,
                               u, row, vt, min_row_column, super );  // min_row_column row-wise

        if( info > 0 ) {
            printf( "The algorithm computing SVD failed to converge.\n" );
            exit( 1 );
        }

        tt_format<datatype> result;

        return result;

    }
}