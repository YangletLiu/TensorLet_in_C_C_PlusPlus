//
// Created by jcfei on 19-1-9.
//

#include "tensor.h"
#include "Tensor3D.h"
#include "tensor_train.h"

namespace TensorLet_decomposition {

    template<class datatype>
    tt_format<datatype> tensor_train( Tensor3D<datatype>& a, datatype epsilon = 1e-6 ) {

        // dimension
        int *shape = a.size();

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

        datatype norm_s;
        datatype norm_tmp;
        datatype ratio;

        norm_s = cblas_dnrm2(min_row_column, s, 1);

        datatype deta = epsilon / sqrt(2) * norm_s;
        deta = 1 - deta;

        int rank_1 = 0;

        for( int i = 1; i < min_row_column; ++i ){
            norm_tmp = cblas_dnrm2( i, s, 1);

            ratio = norm_tmp / norm_s;

            if( ratio > deta || i == min_row_column - 1 ){
                rank_1 = i;
                break;
            }
//            if( rank_1 > 0 ){
//                break;
//            }
        }

//        double sum = 0;
//        for (int i = 0; i < min_row_column; ++i){
//            sum = sum + s[i] * s[i];
//        }
//        sum = sqrt( sum );

        double* C_1 = (double*)MKL_malloc(rank_1 * column * sizeof(double), 64);

        for( int i = 0; i < rank_1; ++i ){
            cblas_daxpy( column, s[i], vt + rank_1 * i, 1, C_1 + i, rank_1 );
        }


        row = rank_1 * n2;
        column = n3;
        min_row_column = min( row, column );

        double* s_1 = (double*)MKL_malloc(min_row_column * sizeof(double), 64);
        double* u_1 = (double*)MKL_malloc( min_row_column * row * sizeof(double), 64);  // 分配 检测
        double* vt_1 = (double*)MKL_malloc(column * column * sizeof(double), 64);

        double super_1[min_row_column - 1];

        info = LAPACKE_dgesvd( LAPACK_COL_MAJOR, 'S', 'A', row, column, C_1, row, s_1,
                               u_1, row, vt_1, min_row_column, super_1 );  // min_row_column row-wise


//        double norm_s_1 = cblas_dnrm2( min_row_column, s_1, 1);
//        deta = epsilon / sqrt(2) * norm_s_1;
//        deta = 1 - deta;

        int rank_2 = 0;
        for( int i = 1; i < min_row_column; ++i ){
            norm_tmp = cblas_dnrm2( i, s_1, 1);

            ratio = norm_tmp / norm_s;

            if( ratio > deta || i == min_row_column - 1){
                rank_2 = i;
                break;
            }
//            if( rank_2 > 0 ){
//                break;
//            }
        }

        double* g3 = (double*)MKL_malloc(rank_2 * column * sizeof(double), 64);

        for( int i = 0; i < rank_2; ++i ){
            cblas_daxpy( column, s_1[i], vt_1 + rank_2 * i, 1, g3 + i, rank_2 );
        }

        tt_format<datatype> result;

        result.G1 = u;   // need to adjust length
        result.g2 = u_1;
        result.G3 = g3;

        // need clean up and rename;

        return result;

    }


}
