//
// Created by jcfei on 19-1-2.
//

#ifndef TENSOR_TENSOR3D_H
#define TENSOR_TENSOR3D_H

#include "tensor.h"

template <class datatype>
class Tensor3D{
private:
    MKL_INT *shape;
    datatype * pointer;
public:
    Tensor3D();

    Tensor3D(int []);
    Tensor3D(int, int, int);
    Tensor3D(const Tensor3D &);
    ~Tensor3D();

    Tensor3D& operator=(const Tensor3D&);
    Tensor3D& operator+=(const Tensor3D&);
    Tensor3D& operator-=(const Tensor3D&);
    Tensor3D& operator*=(const Tensor3D&);

    inline datatype& operator()(MKL_INT i, MKL_INT j, MKL_INT k);

    Tensor3D& rand_tensor(MKL_INT *);
//    Tensor3D& cp_tensor(int rank);
//    Tensor3D& tucker_tensor(int *);

    int* getsize();

    double frobenius_norm(const Tensor3D<datatype>&);


//    Mat<datatype>& tens2mat(const Tensor3D<datatype>&, int mode);
//    Tensor3D<datatype>& mat2tens(const Mat<datatype>&, int mode);
//    Mat<datatype>& tens2vec(const Tensor3D<datatype>&, int mode);
//    Tensor3D<datatype>& vec2tens(const Mat<datatype>&, int mode);

};

template <class datatype>
Tensor3D<datatype>::Tensor3D(){
    shape[0] = 1;
    shape[1] = 1;
    shape[2] = 1;
    pointer = (datatype*)mkl_malloc(1*sizeof(datatype),64);
}

template <class datatype>
Tensor3D<datatype>::Tensor3D(MKL_INT n1, MKL_INT n2, MKL_INT n3){
    shape[0] = n1;
    shape[1] = n2;
    shape[2] = n3;
    pointer = (datatype*)mkl_malloc(n1*n2*n3*sizeof(datatype),64);
    if(pointer==NULL) cout << "Out of Memory" << endl;
}

template <class datatype>
Tensor3D<datatype>::Tensor3D(MKL_INT a[]){
    shape = a;
    pointer = (datatype*)mkl_malloc(shape[0]*shape[1]*shape[2]*sizeof(datatype),64);
}

template<class datatype>
MKL_INT *Tensor3D<datatype>::getsize() {
    return this->shape;
}

template<class datatype>
Tensor3D<datatype>::~Tensor3D() {
    mkl_free(pointer);
}


#endif //TENSOR_TENSOR3D_H
