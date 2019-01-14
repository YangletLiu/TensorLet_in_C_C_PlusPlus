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

    MKL_INT i=0;

    for(MKL_INT i=0;i<r; i++){
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n2,n3,1,1,B+i*n2,n2,C+i*n3,n3,0,c_kr_b+i*n2*n3,n2);  // kr(c,b)
    }
    cout << "A_times_B: " << X1_times_X1T[0] << endl;

    datatype* cal_a = (datatype*)mkl_malloc(n1*r*sizeof(datatype),64);
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n1,r,n2*n3,1,a.pointer,n1,c_kr_b,n2*n3,0,cal_a,n1); // X(1) * kr(c,b)

    MKL_free(c_kr_b);
    MKL_free(cal_a);



/*******update B********
 ***********************/
    datatype* c_kr_a = (datatype*)mkl_malloc(n3*n1*r*sizeof(datatype),64);
    for(MKL_INT i=0;i<r;i++){
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n1,n3,1,1,A+i*n1,n1,C+i*n3,n3,0,c_kr_a+i*n3*n1,n1);  // kr(c,a)
    }
    datatype* cal_b = (datatype*)mkl_malloc(n2*r*sizeof(datatype),64);
    for(MKL_INT i=0;i<n3;i++){
        cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,n2,r,n1,1,a.pointer+i*n1*n2,n1,c_kr_a+i*n2,n1,1,cal_b,n2);  // X(2) * kr(c,a) rank update
    }

    MKL_free(c_kr_a);
    MKL_free(cal_b);

/*******update C********
 ***********************/
    datatype* b_kr_a = (datatype*)mkl_malloc(n1*n2*r*sizeof(datatype),64);
    for(MKL_INT i=0;i<r;i++){
//        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n1,n2,1,1,A+i*n1,n1,B+i*n2,n2,0,b_kr_a+i*n1*n2,n1*n2);  // kr(b,a)
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n2,n1,1,1,B+i*n2,n2,A+i*n1,n1,0,b_kr_a+i*n1*n2,n2);  // kr(a,b)

    }
    datatype* cal_c = (datatype*)mkl_malloc(n3*r*sizeof(datatype),64);

    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n3,r,n1*n2,1,a.pointer,n1*n2,b_kr_a,r,0,cal_c,r); //  X(3) * kr(b,a) CblasRowMajor
    MKL_free(b_kr_a);
    MKL_free(cal_c);

    cp_format<datatype> result0;
    result0.cp_A = A;
    result0.cp_B = B;
    result0.cp_C = C;

    return result0;
    }

}