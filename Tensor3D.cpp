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
datatype* Tensor3D<datatype>::tens2mat(datatype* p, MKL_INT mode) {
    MKL_INT size = this->elements_number();
    if(mode == 1){
        cblas_dcopy(size,this->pointer,1,p,1);
        return p;
    }
}

template<class datatype>
bool Tensor3D<datatype>::operator==(const Tensor3D<datatype> &) {
    return 0;
}

// element initialize  slow
// need 1000000 judge NULL
template<class datatype>
Tensor3D<datatype>& Tensor3D<datatype>::random_tensor() {
    MKL_INT n1 = this->shape[0];
    MKL_INT n2 = this->shape[1];
    MKL_INT n3 = this->shape[2];

    if(n1*n2*n3 <= 100000){
        srand((unsigned)time(NULL));
        MKL_INT SEED = rand();  //随机初始化
        VSLStreamStatePtr stream;
        vslNewStream(&stream,VSL_BRNG_MCG59, SEED);
        for (int i =0; i<n1*n2*n3;i++) {
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,n1*n2*n3,pointer,0,1);
        }
        vslDeleteStream(&stream);
    }
    else{
        MKL_INT J = n1*n2*n3/100000;
        MKL_INT I = J*100000;
        MKL_INT remainder = n1*n2*n3 - J * 100000;
        for (int i =0; i < J; i++) {
            srand((unsigned)time(NULL));
            MKL_INT SEED = rand();
            VSLStreamStatePtr stream;
            vslNewStream(&stream,VSL_BRNG_MCG59, SEED);
            MKL_LONG I0 = i*100000;
            double* p = pointer + I0;
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,100000,p,0,1);
            vslDeleteStream(&stream);
        }
        MKL_INT SEED = rand();
        srand((unsigned)time(NULL));
        VSLStreamStatePtr stream;
        vslNewStream(&stream,VSL_BRNG_MCG59, SEED);
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,remainder,pointer+I,0,1);
    }
    VSLStreamStatePtr stream;
    MKL_INT SEED = rand();  //随机初始化
    srand((unsigned)time(NULL));
    double p[2];
    vslNewStream(&stream,VSL_BRNG_MCG59, SEED);
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,2,p,0,1);
    *pointer = p[1];
    vslDeleteStream(&stream);

    return *this;

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
//Tensor3D<datatype>& Tensor3D<datatype>::operator*(const Tensor3D<datatype>& a)