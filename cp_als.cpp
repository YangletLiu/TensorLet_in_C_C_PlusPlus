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

/*******update A********
 ***********************/
    datatype* c_kr_b = (datatype*)mkl_malloc(n2*n3*r*sizeof(datatype),64);

    for(MKL_INT i=0;i<r; i++){
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n2,n3,1,1,B+i*n2,n2,C+i*n3,n3,0,c_kr_b+i*n2*n3,n2);  // kr(c,b)
    }
//    cout << "A_times_B: " << X1_times_X1T[0] << endl;

    datatype* cal_a = (datatype*)mkl_malloc(n1*r*sizeof(datatype),64);
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n1,r,n2*n3,1,a.pointer,n1,c_kr_b,n2*n3,0,cal_a,n1); // X(1) * kr(c,b)

    datatype* a_times_at = (datatype*)mkl_malloc(r*r*sizeof(datatype),64);
    datatype* b_times_bt = (datatype*)mkl_malloc(r*r*sizeof(datatype),64);
    datatype* c_times_ct = (datatype*)mkl_malloc(r*r*sizeof(datatype),64);
    datatype* c_times_ct_times_b_times_bt = (datatype*)mkl_malloc(r*r*sizeof(datatype),64);
    datatype* c_times_ct_times_a_times_at = (datatype*)mkl_malloc(r*r*sizeof(datatype),64);
    datatype* b_times_bt_times_a_times_at = (datatype*)mkl_malloc(r*r*sizeof(datatype),64);

    cblas_dsyrk(CblasColMajor,CblasUpper,CblasTrans,r,r,n3,C,n3,0,c_times_ct,r);
    cblas_dsyrk(CblasColMajor,CblasUpper,CblasTrans,r,r,n2,B,n2,0,b_times_bt,r);
    vdMul(r*r,c_times_ct,b_times_bt,c_times_ct_times_b_times_bt);
    //pinv need test
    int info = -1;
    MKL_INT* ivpv=(MKL_INT*)mkl_malloc(r * sizeof(MKL_INT), 64);
    datatype* work=(datatype*)mkl_malloc(r * sizeof(datatype),64);

    MKL_INT order = r;
    dsytrf("U",&order,c_times_ct_times_b_times_bt,&r,ivpv,work,&r,&info);
    dsytri("U", &order, c_times_ct_times_b_times_bt, &r, ivpv, work, &info);

//    cblas_dsymm(CblasColMajor, CblasRight, CblasUpper, n1, r, 1, cal_a, r, c_times_ct_times_b_times_bt,n1,0,A,n1);
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n1,r,r,1,cal_a,n1,c_times_ct_times_b_times_bt,r,0,A,n1);

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
    cblas_dsyrk(CblasColMajor,CblasUpper,CblasTrans,r,r,n1,A,n3,0,a_times_at,r);
    vdMul(r*r,c_times_ct,a_times_at,c_times_ct_times_a_times_at);

    dsytrf("U",&r,c_times_ct_times_a_times_at,&r,ivpv,work,&r,&info);
    dsytri("U", &r, c_times_ct_times_a_times_at, &r, ivpv, work, &info);

    cblas_dsymm(CblasColMajor, CblasRight, CblasUpper, n2, r, 1, cal_b, r, c_times_ct_times_a_times_at,n2,0,B,n2);

    MKL_free(c_kr_a);
    MKL_free(cal_b);

/*******update C********
 ***********************/

    datatype* b_kr_a = (datatype*)mkl_malloc(n1*n2*r*sizeof(datatype),64);
    for(MKL_INT i=0;i<r;i++){
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n1,n2,1,1,A+i*n1,n1,B+i*n2,n2,0,b_kr_a+i*n1*n2,n1);  // kr(b,a)
//        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n2,n1,1,1,B+i*n2,n2,A+i*n1,n1,0,b_kr_a+i*n1*n2,n2);  // kr(a,b)
    }

    datatype* cal_c = (datatype*)mkl_malloc(n3*r*sizeof(datatype),64);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n3,r,n1*n2,1,a.pointer,n1*n2,b_kr_a,n1*n2,0,cal_c,r); //  X(3) * kr(b,a) CblasRowMajor
    MKL_free(b_kr_a);

    cblas_dsyrk(CblasColMajor,CblasUpper,CblasTrans,r,r,n2,B,n2,0,b_times_bt,r);
    vdMul(r*r,b_times_bt,a_times_at,b_times_bt_times_a_times_at);

    dsytrf("U",&r,b_times_bt_times_a_times_at,&r,ivpv,work,&r,&info);
    dsytri("U", &r, b_times_bt_times_a_times_at, &r, ivpv, work, &info);
    cblas_dsymm(CblasRowMajor, CblasRight, CblasUpper, n3, r, 1, cal_c, r, b_times_bt_times_a_times_at,r,0,C,r);

    MKL_free(cal_c);

    MKL_free(ivpv);
    MKL_free(work);
    MKL_free(a_times_at);
    MKL_free(b_times_bt);
    MKL_free(c_times_ct);
    MKL_free(c_times_ct_times_b_times_bt);
    MKL_free(c_times_ct_times_a_times_at);
    MKL_free(b_times_bt_times_a_times_at);

    cp_format<datatype> result;
    result.cp_A = A;
    result.cp_B = B;
    result.cp_C = C;

    return result;
    }

}