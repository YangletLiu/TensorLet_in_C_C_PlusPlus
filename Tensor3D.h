//
// Created by jcfei on 19-1-2.
//

#ifndef TENSOR_TENSOR3D_H
#define TENSOR_TENSOR3D_H

#include "tensor.h"

template <class datatype>
class Tensor3D{
//private:
public:
    MKL_INT shape[3];
    datatype *pointer;
public:
    Tensor3D();
    Tensor3D(MKL_INT, MKL_INT, MKL_INT);
    explicit Tensor3D(MKL_INT []);

    ~Tensor3D();

    Tensor3D(const Tensor3D<datatype>&);
//operators
    Tensor3D<datatype>& random_tensor();

    inline datatype& operator()(MKL_INT i, MKL_INT j, MKL_INT k);

    inline MKL_INT* size();
    inline MKL_INT elements_number();
    double frobenius_norm();

    Tensor3D<datatype>& operator=(const Tensor3D<datatype>&);
    Tensor3D<datatype>& operator+=(const Tensor3D<datatype>&);
    Tensor3D<datatype>& operator-=(const Tensor3D<datatype>&);

    bool operator==(const Tensor3D<datatype>&);


    Tensor3D<datatype>& cp_gen(MKL_INT r);

    datatype* tens2mat(datatype *, MKL_INT mode);
    datatype* tens2vec(datatype *, MKL_INT mode);

//    Tensor3D<datatype>& operator+(const Tensor3D<datatype>&);
//    Tensor3D<datatype>& operator-(const Tensor3D<datatype>&);

//    Tensor3D& cp_tensor(int rank);
//    Tensor3D& tucker_tensor(int *);

//    datatype* mode_n_product(datatype *matrix, datatype *result, MKL_INT mode);
};

// element-wise add
template <class datatype>
Tensor3D<datatype> operator+(Tensor3D<datatype> &, Tensor3D<datatype> &);

// element-wise subtraction
template<class datatype>
Tensor3D<datatype> operator-(Tensor3D<datatype> &a, Tensor3D<datatype> &b);

// element-wise product
template<class datatype>
Tensor3D<datatype> operator*(Tensor3D<datatype> &a, Tensor3D<datatype> &b);  //不加&， 返回局部变量？

// scalar-tensor product
template<class datatype>
Tensor3D<datatype>& operator*(MKL_INT k, Tensor3D<datatype> &a); //why out of class?

template<class datatype>
Tensor3D<datatype>& operator*(datatype k, Tensor3D<datatype> &a);
//    friend Tensor3D<datatype>& operator*(MKL_INT k, Tensor3D<datatype> &a);  ???why

template<class datatype>
Tensor3D<datatype> mat2tens(datatype*, MKL_INT mode, MKL_INT shape[]);

template<class datatype>
Tensor3D<datatype> mat2tens(datatype*, MKL_INT mode, MKL_INT n1, MKL_INT n2, MKL_INT n3);

//template<class datatype>
//Tensor3D<datatype> vec2tens(datatype*, MKL_INT mode, MKL_INT shape[]);
//
//template<class datatype>
//Tensor3D<datatype> vec2tens(datatype*, MKL_INT mode, MKL_INT n1, MKL_INT n2, MKL_INT n3);

#endif //TENSOR_TENSOR3D_H