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
