//
// Created by jcfei on 18-10-3.
//

#include "tensor.h"
#include "Tensor3D.h"

//template<class datatype>
//datatype* ten2mat(Tensor3D<datatype>& a, int order) {
//    datatype* result;
//    return result;
//};

//template<class datatype>
//datatype *mode_n_product(Tensor3D<datatype> &tensor, datatype *matrix, datatype *result, int mode) {
//
//    MKL_INT *shape = tensor.getsize();  //dimension
//    MKL_INT n1 = shape[0]; MKL_INT n2 =shape[1]; MKL_INT n3 = shape[2];
//
//    if(mode == 2){
//        for(MKL_INT i = 0; i < n3; i++){
//            cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, n2, n1, n2,
//                        1, matrix, n2, tensor.pointer + i * n1 * n2, n2,
//                        0, result + i * n1 * n2, n1);  // U2^t * ( U1
//        }
//    }
//
//    return result;
//}