//
// Created by jcfei on 19-1-4.
//

#include "tensor.h"
#include "Tensor3D.h"

template<class datatype>
Tensor3D<datatype>& Tensor3D<datatype>::operator=(const Tensor3D<datatype> &a) {
    shape = new MKL_INT[3];
    shape[0]= a.shape[0];
    shape[1]= a.shape[1];
    shape[2]= a.shape[2];
    pointer = (datatype*)mkl_malloc(shape[0]*shape[1]*shape[2]*sizeof(datatype),64);
    MKL_INT size = this->elements_number();
    cblas_dcopy(size,a.pointer,1,pointer,1);
    return *this;
}

template<class datatype>
Tensor3D<datatype>& Tensor3D<datatype>::operator+=(const Tensor3D<datatype>& a) {
    MKL_INT size = this->elements_number();
    cblas_daxpby(size,1,a.pointer,1, 1,this->pointer,1);
    return *this;
}

template<class datatype>
Tensor3D<datatype>& Tensor3D<datatype>::operator-=(const Tensor3D<datatype>& a) {
    MKL_INT size = this->elements_number();
    cblas_daxpby(size,-1,a.pointer,1, 1,this->pointer,1);
    return *this;
}

template<class datatype>
datatype* Tensor3D<datatype>::tens2mat(int mode) {
    MKL_INT size = this->elements_number();
    datatype * p = (datatype*)mkl_malloc(shape[0]*shape[1]*shape[2]*sizeof(datatype),64);
    if(mode == 1){
        cblas_dcopy(size,p,1,this->pointer,1);
        return p;
    }
}

// element-wise add
template<class datatype>
Tensor3D<datatype> operator+(Tensor3D<datatype>& a, Tensor3D<datatype>& b) {
    Tensor3D<datatype> result(a.shape);
//    Tensor3D<datatype> *result = new Tensor3D<datatype>(a.shape);
    MKL_INT size;
    size = a.elements_number();
    vdAdd(size,a.pointer,b.pointer,result.pointer);
    return result;
}

// element-wise subtraction
template<class datatype>
Tensor3D<datatype> operator-(Tensor3D<datatype>& a, Tensor3D<datatype>& b) {
    Tensor3D<datatype> result(a.shape);
    MKL_INT size;
    size = a.elements_number();
    vdSub(size,a.pointer,b.pointer,result.pointer);
    return result;
}

// element-wise product
template<class datatype>
Tensor3D<datatype> operator*(Tensor3D<datatype>& a, Tensor3D<datatype>& b) {
    Tensor3D<datatype> result(a.shape);
    MKL_INT size;
    size = a.elements_number();
    vdMul(size,a.pointer,b.pointer,result.pointer);
    return result;
}

// scalar-tensor product
template<class datatype>
Tensor3D<datatype>& operator*(MKL_INT k, Tensor3D<datatype>& a) {
    MKL_INT size;
    size = a.elements_number();
    cblas_dscal(size, k, a.pointer, 1);
    return a;
}

template<class datatype>
Tensor3D<datatype>& operator*(datatype k, Tensor3D<datatype>& a) {
    MKL_INT size;
    size = a.elements_number();
    cblas_dscal(size, k, a.pointer, 1);
    return a;
}


//template<class datatype>
//Tensor3D<datatype>& Tensor3D<datatype>::operator*(const Tensor3D<datatype>& a) {
//    MKL_INT size;
//    size = a.elements_number();
//    vdMul(size,a.pointer,this->pointer,this->pointer);
//    return *this;
//}
