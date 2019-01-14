//
// Created by jcfei on 18-9-9.
//

#include "tensor.h"
#include "Tensor3D.h"


template <class datatype>
class cp_format{
public:
    datatype* cp_A;
    datatype* cp_B;
    datatype* cp_C;
};
namespace TensorLet_decomposition{

template<class datatype>
cp_format<datatype> cp_als(Tensor3D<datatype> &a, int r, int max_iter = 1, datatype tolerance = 1e-6) {
    MKL_INT *shape = a.getsize();  //dimension
    MKL_INT n1 = shape[0]; MKL_INT n2 =shape[1]; MKL_INT n3 = shape[2];

    //random A,B,C
    datatype* A = (datatype*)mkl_malloc(n1*r*sizeof(datatype),64);
    datatype* B = (datatype*)mkl_malloc(n2*r*sizeof(datatype),64);
    datatype* C = (datatype*)mkl_malloc(n3*r*sizeof(datatype),64);

    VSLStreamStatePtr stream;

    srand((unsigned)time(NULL));
    MKL_INT SEED = rand();  //随机初始化
    vslNewStream(&stream,VSL_BRNG_MCG31, SEED);
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,n1*r,A,0,1);
    srand((unsigned)time(NULL));
    SEED = rand();  //随机初始化
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,n2*r,B,0,1);
    srand((unsigned)time(NULL));
    SEED = rand();  //随机初始化
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,n3*r,C,0,1);

    vslDeleteStream(&stream);

    datatype* X1_times_X1T = (datatype*)mkl_malloc(n1*n1*sizeof(datatype),64);
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n1,n1,n2*n3,1,a.pointer,n1,a.pointer,n1,0,X1_times_X1T,n1);
    cout << "A_times_B: " << X1_times_X1T[0] << endl;

/*******update A********
 ***********************/
    datatype* c_kr_b = (datatype*)mkl_malloc(n2*n3*r*sizeof(datatype),64);
    for(MKL_INT i=0;i<r;i++){
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n2,n3,1,1,B+i*n2,n2,C+i*n3,n3,0,c_kr_b+i*n2*n3,n2*n3);  // kr(c,b)
    }

    datatype* cal_a = (datatype*)mkl_malloc(n1*r*sizeof(datatype),64);
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n1,r,n2*n3,1,a.pointer,n1,c_kr_b,n2*n3,0,cal_a,n1); // X(1) * kr(c,b)
    MKL_free(c_kr_b);
    MKL_free(cal_a);

/*******update B********
 ***********************/
    datatype* c_kr_a = (datatype*)mkl_malloc(n3*n1*r*sizeof(datatype),64);
    for(MKL_INT i=0;i<r;i++){
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n1,n3,1,1,A+i*n1,n1,C+i*n3,n3,0,c_kr_a+i*n3*n1,n3*n1);  // kr(c,a)
    }
    datatype* cal_b = (datatype*)mkl_malloc(n2*r*sizeof(datatype),64);
    for(MKL_INT i=0;i<n3;i++){
        cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,n2,r,n1,1,a.pointer+i*n1*n2,n1,c_kr_a+i*n2,n1,1,cal_b,n2);
    }

    MKL_free(c_kr_a);
    MKL_free(cal_b);

/*******update C********
 ***********************/

    cp_format<datatype> result0;
    result0.cp_A = A;
    result0.cp_B = B;
    result0.cp_C = C;

    return result0;
    }

}

//    tmp= a1*a1.t();
//    for (int turn=0; turn<1;turn++) {
//        tmp.zeros();
//        tmp.set_size(n1, r);
//        Mat<T> kr_tmp(n2*n3,r);
//        for (int i = 0; i < r; i++) {
//            kr_tmp.col(i) = kron(C.col(i), B.col(i));
//        }
//        tmp = a1 * kr_tmp;
//        A = tmp * pinv((C.t() * C) % (B.t() * B));
//        Mat<T> tmpp = (C.t() * C) % (B.t() * B);
//        solve( A, tmp, tmpp);
//        A = normalise(A);
//
//        tmp.set_size(n2, r);
//        tmp.zeros();
//        kr_tmp.set_size(n1, r);
//        for (int j = 0; j < n3; j++) {
//            for (int i = 0; i < r; i++) {
//                kr_tmp.col(i) = C(j, i) * A.col(i);
//            }
//            cal = a1.cols(j*n2, j*n2+n2-1);
//            tmp = tmp + cal.t() * kr_tmp; //slice computing
//        }
//        B = tmp * pinv((C.t() * C) % (A.t() * A));
//        tmpp = (C.t() * C) % (A.t() * A);
//        solve( B, tmp, tmpp);
//
//        B = normalise(B);
//
//        tmp.set_size(n3, r);
//        tmp.zeros();
//        cal.set_size(1,n1*n2);
//        kr_tmp.set_size(n1*n2,r);
//        for (int j = 0; j < n2; j++) {
//            for (int i = 0; i < r; i++) {
//                kr_tmp.col(i) = B(j, i) * A.col(i);
//            }
//            cal = a1.cols(j*n2, j*n2+n2-1);
//            cal.reshape(n1,n2);
//            tmp = tmp + cal.t() * kr_tmp; //slice computing
//        }
//        a1.set_size(n1*n2,n3);
//        Mat<T> a3(n1*n2,n3);
//        for (int i = 0; i < r; i++) {
//            kr_tmp.col(i) = kron(B.col(i), A.col(i));
//        }
//
//        for (int j = 0; j < n3; j++) {
//            Mat<T> tmp1 = a.slice(j);
//            cal = vectorise(tmp1);
//            a3.col(j) = cal;
//        }
//        tmp = a3.t() * kr_tmp;
//        mat tmpp = (B.t() * B) % (A.t() * A);
//        C = solve(tmpp, tmp);
//        C = tmp * pinv((B.t() * B) % (A.t() * A));
//
//        if (turn < 0){
//            C = normalise(C);
//        }
//    }