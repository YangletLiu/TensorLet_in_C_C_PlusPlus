//
// Created by jcfei on 19-2-17.
//


#include "tensor.h"
#include "Tensor3D.h"

template<class datatype>
datatype* mode_n_product(Tensor3D<datatype> &tensor, datatype *matrix, datatype *result, int mode) {

    MKL_INT *shape = tensor.getsize();  //dimension
    MKL_INT n1 = shape[0]; MKL_INT n2 =shape[1]; MKL_INT n3 = shape[2];

    if(mode == 1){
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2 * n3, n1,
                    1, matrix , n1, tensor.pointer, n1,
                    0, result, n1); // U1 * X(1)
    }

    if(mode == 2){
        for(MKL_INT i = 0; i < n3; i++){
            cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, n2, n1, n2,
                        1, matrix, n2, tensor.pointer + i * n1 * n2, n2,
                        0, result + i * n1 * n2, n1);  // U2 * X(2)
        }
    }

    if(mode == 3){
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n1 * n2, n3, n1,
                    1, tensor.pointer, n1 * n2, matrix , n1,
                    0, result, n1 * n2 ); // U3 * X(3)
    }

    return result;
}