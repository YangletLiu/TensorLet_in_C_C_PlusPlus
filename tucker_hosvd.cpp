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


    }

}
//
//    int n1=a.n_rows; int n2=a.n_cols; int n3=a.n_slices;
//
////mode-1
////    reshape(a,n1*n2,n3,1);
//    Mat<T> a1 = reshape(a,n1,n2*n3,1);
//
//    Mat<T> tmp(n1,n1);
//    tmp= a1*a1.t();
//
//    Mat<T> U;//U,V均为正交矩阵
//    Col<T> S;//S为奇异值构成的列向量
//    Mat<T> U1(n1,r1);
//
//    eig_sym(S,U,tmp);
//    U1 = U.cols(n1-r1,n1-1);
//    a1.reset();
//
////mode-2
//    tmp.resize(n2,n2);
//    tmp.zeros();
//    Mat<T> cal(n1,n2);
//
//    for (int i=0; i< n3;i++){
//        cal = a.slice(i);
//        tmp = tmp + cal.t()*cal;
//    }
////    a.for_each([](Mat<T> tmp){ return cal += tmp.t()*tmp;});
//
//    eig_sym(S,U,tmp);
//    Mat<T> U2(n2,r2);
//    U2 = U.cols(n2-r2,n2-1);
//
////mode-3
//    tmp.resize(n3,n3);
////    Mat<T> a3(n1*n2,n3);
//    a1.set_size(n1*n2,n3);
//    for (int j = 0; j < n3; j++) {
//        Mat<T> tmp1 = a.slice(j);
//        cal = vectorise(tmp1);
//        a1.col(j) = cal;
//    }
//
//    Mat<T> U3(n3,r3);
//    tmp= a1.t()*a1;
//
//    eig_sym(S,U,tmp);
//    U3 = U.cols(n3-r3,n3-1);
////    inplace_trans(U3);
////    m1.reset(); //直接改变矩阵大小...
//
//    U.reset();S.reset();
////    Cube<T>  g_tmp(r1,r2,n3);
//    tmp.resize(r1,r2);
//
//    a1.set_size(r1*r2,n3);
//
//    for (int i=0; i< n3;i++){
//        cal = a.slice(i);
//        tmp = U1.t()*cal*U2;
//        cal = vectorise(tmp);
//        a1.col(i) = cal;
//    }
//
//    Cube<T> g(r1,r2,r3);
//    Mat<T> tmpp(n3,r1*r2);
//
//    Col<T> ctmp(r1*r2,1);
//    tmpp  = a1*U3;
//    for (int k=0; k< r3;k++){
//        ctmp = tmpp.col(k);
//        g.slice(k) = reshape(ctmp,r1,r2);
//    }
//
//    tucker_core<T> A;
//    A.u1 = U1;
//    A.u2 = U2;
//    A.u3 = U3;
//    A.core = g;
//    return A;
//}
