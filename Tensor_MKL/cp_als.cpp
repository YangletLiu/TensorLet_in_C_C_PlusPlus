//
// Created by jcfei on 18-9-9.
//

#include "tensor.h"
#include "Tensor3D.h"


template <class datatype>
class cp_format{
    datatype* A,B,C;
};

template<class datatype>
cp_format<datatype> cp_als(Tensor3D<datatype> &a, int r, int max_iter = 1, datatype tolerance = 1e-6) {
    int *shape = a.getsize();  //dimension
    int n1 = shape[0]; int n2 =shape[1]; int n3 = shape[2];
    datatype* A,B,C;    //random A,B,C
    datatype* cal, tmp;
    datatype* a1 = a.tens2mat(a1,1);

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
    datatype* a2 = a.tens2mat(a2,1);
    datatype* a3 = a.tens2mat(a3,1);

    cp_format<datatype> result0;
    result0.A = A;
    result0.B = B;
    result0.C = C;

    return result0;
}