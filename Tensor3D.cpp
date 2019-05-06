//
// Created by jcfei on 19-1-4.
//

#include "tensor.h"
#include "Tensor3D.h"

template <class datatype>
Tensor3D<datatype>::Tensor3D():shape{0,0,0}, pointer(nullptr) {
//    shape[0] = 0;
//    shape[1] = 0;
//    shape[2] = 0;
//    pointer = nullptr;
}

template <class datatype>
Tensor3D<datatype>::Tensor3D(MKL_INT n1, MKL_INT n2, MKL_INT n3):shape{n1, n2, n3}{
    pointer = (datatype*)mkl_malloc(n1*n2*n3*sizeof(datatype),64);
    if (pointer == NULL) printf("Cannot allocate enough memory\n");
}

template <class datatype>
Tensor3D<datatype>::Tensor3D(MKL_INT a[]){
    shape[0]= a[0];
    shape[1]= a[1];
    shape[2]= a[2];
    pointer = (datatype*)mkl_malloc(shape[0]*shape[1]*shape[2]*sizeof(datatype),64);
    if (pointer == NULL) printf("Cannot allocate enough memory\n");
}

//Copy function
template<class datatype>
Tensor3D<datatype>::Tensor3D(const Tensor3D<datatype>& a) {
    shape[0]= a.shape[0];
    shape[1]= a.shape[1];
    shape[2]= a.shape[2];
    pointer = (datatype*)mkl_malloc(shape[0]*shape[1]*shape[2]*sizeof(datatype),64);
    if (pointer == NULL) printf("Cannot allocate enough memory\n");
    // 添加copy函数
    for (int i = 0; i < shape[0]*shape[1]*shape[2]; ++i){
        pointer[i] = a.pointer[i];
    }
}

// element initialize  slow
// need 1000000 judge NULL
template<class datatype>
Tensor3D<datatype>& Tensor3D<datatype>::random_tensor() {
    MKL_INT n1 = this->shape[0];
    MKL_INT n2 = this->shape[1];
    MKL_INT n3 = this->shape[2];

    if(n1 * n2 * n3 <= 100000){
        srand((unsigned)time(NULL));

        MKL_INT SEED = rand();  //随机初始化

        VSLStreamStatePtr stream;

        vslNewStream(&stream, VSL_BRNG_MCG59, SEED);

        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n1*n2*n3, pointer, 0, 1);

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
            vslNewStream(&stream, VSL_BRNG_MCG59, SEED);

            MKL_LONG I0 = i*100000;
            double* p = pointer + I0;

            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 100000, p, 0, 1);

            vslDeleteStream(&stream);
        }

        MKL_INT SEED = rand();
        srand((unsigned)time(NULL));

        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_MCG59, SEED);

        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, remainder, pointer+I, 0, 1);
    }

    VSLStreamStatePtr stream;
    MKL_INT SEED = rand();  //随机初始化
    srand((unsigned)time(NULL));

    double p[2];
    vslNewStream(&stream, VSL_BRNG_MCG59, SEED);
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 2, p, 0, 1);

    pointer[0] = p[1];
    vslDeleteStream(&stream);

    return *this;
}

//Destructor
template<class datatype>
Tensor3D<datatype>::~Tensor3D() {
    mkl_free(pointer);
}

template<class datatype>
MKL_INT* Tensor3D<datatype>::size() {
    return this->shape;
}

template <class datatype>
MKL_INT Tensor3D<datatype>::elements_number() {
    return this->shape[0] * this->shape[1] * this->shape[2];
}

//000
template<class datatype>
double Tensor3D<datatype>::frobenius_norm() {
    double result;
    MKL_INT size = this -> elements_number();
    return cblas_dnrm2(size, this->pointer, 1);
}

//000
template<class datatype>
datatype &Tensor3D<datatype>::operator()(MKL_INT i, MKL_INT j, MKL_INT k) {
    return this -> pointer[i - 1 + (j-1) * shape[0] + (k-1) * shape[0] * shape[1]];
}

//000
template<class datatype>
Tensor3D<datatype>& Tensor3D<datatype>::operator=(const Tensor3D<datatype> &a) {
    shape[0]= a.shape[0];
    shape[1]= a.shape[1];
    shape[2]= a.shape[2];
    MKL_INT size = this -> elements_number();
    pointer = (datatype*)mkl_malloc(size * sizeof(datatype), 64);
    cblas_dcopy(size, a.pointer, 1, pointer, 1);
    return *this;
}

//000
template<class datatype>
Tensor3D<datatype>& Tensor3D<datatype>::operator+=(const Tensor3D<datatype>& a) {
    MKL_INT size = this->elements_number();
    cblas_daxpby(size,1,a.pointer,1, 1,this->pointer,1);
    return *this;
}

//000
template<class datatype>
Tensor3D<datatype>& Tensor3D<datatype>::operator-=(const Tensor3D<datatype>& a) {
    MKL_INT size = this->elements_number();
    cblas_daxpby(size,-1,a.pointer,1, 1,this->pointer,1);
    return *this;
}

template<class datatype>
bool Tensor3D<datatype>::operator==(const Tensor3D<datatype>& a) {
    if (shape[0] != a.shape[0] || shape[1] != a.shape[1] || shape[2] != a.shape[2]){
        return false;
    }
    for (int i = 0; i < shape[0]*shape[1]*shape[2]; ++i){
        if (pointer[i] != a.pointer[i])
            return false;
    }

    return true;
}

//000
template<class datatype>
Tensor3D<datatype> operator+(Tensor3D<datatype>& a, Tensor3D<datatype>& b) {
    Tensor3D<datatype> result(a.shape);
//    Tensor3D<datatype> *result = new Tensor3D<datatype>(a.shape);
    MKL_INT size;
    size = a.elements_number();
    vdAdd(size,a.pointer,b.pointer,result.pointer);
    return result;
}

//000
template<class datatype>
Tensor3D<datatype> operator-(Tensor3D<datatype>& a, Tensor3D<datatype>& b) {
    Tensor3D<datatype> result(a.shape);
    MKL_INT size;
    size = a.elements_number();
    vdSub(size,a.pointer,b.pointer,result.pointer);
    return result;
}

//000
template<class datatype>
Tensor3D<datatype> operator*(Tensor3D<datatype>& a, Tensor3D<datatype>& b) {
    Tensor3D<datatype> result(a.shape);
    MKL_INT size;
    size = a.elements_number();
    vdMul(size,a.pointer,b.pointer,result.pointer);
    return result;
}

//000
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

//000
template<class datatype>
datatype* Tensor3D<datatype>::tens2mat(datatype* p, MKL_INT mode) {
    MKL_INT size = this->elements_number();
    if(mode == 1){
        cblas_dcopy(size,this->pointer,1,p,1);
        return p;
    }
}

////000
//template<class datatype>
//datatype* Tensor3D<datatype>::mode_n_product(datatype *matrix, datatype *result, int mode) {
//    MKL_INT *shape = this->size();  //dimension
//    MKL_INT n1 = shape[0]; MKL_INT n2 =shape[1]; MKL_INT n3 = shape[2];
//
//    if(mode == 1){
//        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2 * n3, n1,
//                    1, matrix , n1, this->pointer, n1,
//                    0, result, n1); // U1 * X(1)
//    }
//
//    if(mode == 2){
//        for(MKL_INT i = 0; i < n3; i++){
//            cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, n2, n1, n2,
//                        1, matrix, n2, this->pointer + i * n1 * n2, n2,
//                        0, result + i * n1 * n2, n1);  // U2 * X(2)
//        }
//    }
//
////    if(mode == 3){
////        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n3, n1 * n2, n1,
////                    1, matrix , n1, this->pointer, n1,
////                    0, result, n1); // U1 * X(1)
////    }
//
//    return result;
//}
//


