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
    MKL_INT *shape;
    datatype * pointer;
public:
    Tensor3D();

    explicit Tensor3D(int []);
    Tensor3D(int, int, int);
    Tensor3D(const Tensor3D&);
    ~Tensor3D();

    Tensor3D<datatype>& operator=(const Tensor3D<datatype>&);
    Tensor3D<datatype>& operator+=(const Tensor3D<datatype>&);
    Tensor3D<datatype>& operator-=(const Tensor3D<datatype>&);
    Tensor3D<datatype>& operator+(const Tensor3D<datatype>&);
    Tensor3D<datatype>& operator-(const Tensor3D<datatype>&);
//    Tensor3D<datatype>& operator*(datatype);


    inline datatype& operator()(MKL_INT i, MKL_INT j, MKL_INT k);

    Tensor3D& rand_tensor(MKL_INT *);
//    Tensor3D& cp_tensor(int rank);
//    Tensor3D& tucker_tensor(int *);

    inline MKL_INT* getsize();
    inline MKL_INT elements_number();
    double frobenius_norm();

//    friend Tensor3D<datatype>& operator*(MKL_INT k, Tensor3D<datatype> &a);  ???why

//    Mat<datatype>& tens2mat(const Tensor3D<datatype>&, int mode);
//    Tensor3D<datatype>& mat2tens(const Mat<datatype>&, int mode);
//    Mat<datatype>& tens2vec(const Tensor3D<datatype>&, int mode);
//    Tensor3D<datatype>& vec2tens(const Mat<datatype>&, int mode);

};

// Constructors
template <class datatype>
Tensor3D<datatype>::Tensor3D(){
    shape = new MKL_INT[3];
    shape[0] = 1;
    shape[1] = 1;
    shape[2] = 1;
    pointer = (datatype*)mkl_malloc(1*sizeof(datatype),64);
}

template <class datatype>
Tensor3D<datatype>::Tensor3D(MKL_INT n1, MKL_INT n2, MKL_INT n3){
    shape = new MKL_INT[3];
    shape[0] = n1;
    shape[1] = n2;
    shape[2] = n3;
    pointer = (datatype*)mkl_malloc(n1*n2*n3*sizeof(datatype),64);
    if(pointer==NULL) cout << "Out of Memory" << endl;
}

template <class datatype>
Tensor3D<datatype>::Tensor3D(MKL_INT a[]){
    shape = new MKL_INT[3];
    shape[0]= a[0];
    shape[1]= a[1];
    shape[2]= a[2];
    pointer = (datatype*)mkl_malloc(shape[0]*shape[1]*shape[2]*sizeof(datatype),64);
}

//Copy function
template<class datatype>
Tensor3D<datatype>::Tensor3D(const Tensor3D& a) {
    shape = new MKL_INT[3];
    shape[0]= a.shape[0];
    shape[1]= a.shape[1];
    shape[2]= a.shape[2];
    pointer = (datatype*)mkl_malloc(shape[0]*shape[1]*shape[2]*sizeof(datatype),64);
    pointer = a.pointer;
}

//Destructor
template<class datatype>
Tensor3D<datatype>::~Tensor3D() {
    mkl_free(pointer);
    delete [] shape;
}

template<class datatype>
MKL_INT* Tensor3D<datatype>::getsize() {
    return this->shape;
}

template <class datatype>
MKL_INT Tensor3D<datatype>::elements_number() {
    return this->shape[0] * this->shape[1] * this->shape[2];
}

template<class datatype>
double Tensor3D<datatype>::frobenius_norm() {
    double result;
    MKL_INT size = this->elements_number();
    return cblas_dnrm2(size, this->pointer,1);
}


template<class datatype>
datatype &Tensor3D<datatype>::operator()(MKL_INT i, MKL_INT j, MKL_INT k) {

    return this->pointer[i-1+(j-1)*shape[0]+(k-1)*shape[0]*shape[1]];  // remain test
//    1][(i-1)][(k-1)*shape[1]*shape[2]

}

template<class datatype>
Tensor3D<datatype>& operator*(MKL_INT k, Tensor3D<datatype> &a); //why out of class?


template<class datatype>
Tensor3D<datatype>& operator*(datatype k, Tensor3D<datatype> &a); //why out of class?



#endif //TENSOR_TENSOR3D_H

//template<class datatype>
//Tensor3D<datatype> &Tensor3D<datatype>::operator*(datatype k) {
//    MKL_INT size = this->elements_number();
//    cblas_dscal(size, k, this->pointer, 1);
//    return *this;
//}

//template<class datatype>
//Tensor3D<datatype>& operator*(datatype& k, Tensor3D<datatype>& a) {
//    MKL_INT size = a.elements_number();
//    cblas_dscal(size, k, a.pointer, 1);
//    return a;
//}