//
// Created by jcfei on 18-12-25.
//

#ifndef TENSOR_INTERFACE_H
#define TENSOR_INTERFACE_H

#include "Tensor3D.h"

// 矩阵类声明
template <class datatype>
class Mat;

// 张量分解名字空间
namespace tensorlet_decomposition{
    //CP 结构
    template <class datatype>
    class cp_decomposition{
        Mat<datatype> A,B,C;
    };


    //tucker 返回结构
    template<class datatype>
    class tucker_decomposition{
        Tensor3D<datatype> core;
        Mat<datatype> u1,u2,u3;
    };

    //t-SVD 返回结构
    template<class datatype>
    class tsvd_decomposition{
        Tensor3D<datatype> U,Theta,V;
    };

    //TT 返回结构
    template<class datatype>
    class tt_decomposition{
        Tensor3D<datatype> g;
        Mat<datatype> G1,G2;
    };

    template <class datatype>
    cp_decomposition<datatype> & cp_als(Tensor3D<datatype>& tensor, int rank, int max_iter, datatype tol);

    template <class datatype>
    tucker_decomposition<datatype> & tucker_hosvd(Tensor3D<datatype>& tensor, int* ranks);

    template <class datatype>
    tucker_decomposition<datatype> & tucker_hooi(Tensor3D<datatype>& tensor, int* ranks, int max_iter, datatype tol);

    template <class datatype>
    tsvd_decomposition<datatype> & tsvd(Tensor3D<datatype>& tensor);

    template <class datatype>
    tt_decomposition<datatype> & tt(Tensor3D<datatype>& tensor);

/******************************************************************************************************/
    //生成低秩张量
    template <class datatype>
    Tensor3D<datatype> & cp_gen(cp_decomposition<datatype>& a);

    template <class datatype>
    Tensor3D<datatype> & tucker_gen(tucker_decomposition<datatype>& a);

    template <class datatype>
    Tensor3D<datatype> & tsvd_gen(tsvd_decomposition<datatype>& a);

    template <class datatype>
    Tensor3D<datatype> & tt_gen(tt_decomposition<datatype>& a);

/******************************************************************************************************/


}   //namespace for decomposition

#endif //TENSOR_INTERFACE_H
