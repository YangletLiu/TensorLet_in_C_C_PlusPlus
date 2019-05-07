//
// Created by jcfei on 19-5-7.
//

#ifndef TENSOR_CPD_H
#define TENSOR_CPD_H

#include "tensor.h"
#include "Tensor3D.h"


template <class datatype>
class cp_format{
public:
    datatype* cp_A;
    datatype* cp_B;
    datatype* cp_C;
    datatype* cp_lamda;
};

//namespace TensorLet_decomposition{
//
//    template<class datatype>
//    cp_format<datatype> cp_als(Tensor3D<datatype> &a, int r, int max_iter = 500, double tolerance = 1e-6);
//
//    template<class datatype>
//    cp_format<datatype> cp_als( Tensor3D<datatype> &a, int r, double tolerance = 1e-6);
//
//}


#endif //TENSOR_CPD_H
