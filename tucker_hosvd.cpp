//
// Created by jcfei on 18-9-26.
//

#include "tensor.h"
#include "Tensor3D.h"

template<class datatype>
class tucker_format{
    Tensor3D<datatype> core;
    datatype* u1,u2,u3;
};

namespace TensorLet_decomposition {

    template<class datatype>
    tucker_format<datatype> tucker_hosvd(Tensor3D<datatype> &a, int r1, int r2, int r3) {
        int *shape = a.getsize();  //dimension
        int n1 = shape[0]; int n2 =shape[1]; int n3 = shape[2];

        datatype* cal = (datatype*)mkl_malloc(n1*n1*sizeof(datatype),64);

        datatype* X1_times_X1T = (datatype*)mkl_malloc(n1*n1*sizeof(datatype),64);

        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n1,n1,n2*n3,1,a.pointer,n1,a.pointer,n1,0,X1_times_X1T,n1);
        cblas_dsyrk(CblasColMajor,CblasUpper,CblasNoTrans,n1,n2*n3,1,a.pointer,n1,0,X1_times_X1T,n1);

    }

}

//    datatype* X1_times_X1T = (datatype*)mkl_malloc(n1*n1*sizeof(datatype),64);
//    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n1,n1,n2*n3,1,a.pointer,n1,a.pointer,n1,0,X1_times_X1T,n1);
//    cout << "A_times_B: " << X1_times_X1T[0] << endl;