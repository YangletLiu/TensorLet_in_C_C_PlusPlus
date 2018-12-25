//
// Created by jcfei on 18-9-9.
//

#include "tensor.h"

template<class T>
cp_mats<T> cp_als(Cube<T> &a, int r) {
    int n1=a.n_rows; int n2=a.n_cols; int n3=a.n_slices;  //dimension

    Mat<T> A(n1,r); Mat<T> B=randn<Mat<T>>(n2,r); Mat<T> C=randn<Mat<T>>(n3,r); //random A,B,C

    Mat<T> cal(n1,n2); Mat<T> tmp(n1,r);

//    cube b(a);
    reshape(a,n1*n2,n3,1);
    Mat<T> a1 = a.slice(0);
    reshape(a,n1,n2,n3);
    tmp= a1*a1.t();
    a.reshape(n1,n2,n3);

    for (int turn=0; turn<1;turn++) {
        tmp.zeros();
        tmp.set_size(n1, r);
        Mat<T> kr_tmp(n2*n3,r);
        for (int i = 0; i < r; i++) {
            kr_tmp.col(i) = kron(C.col(i), B.col(i));
        }
        tmp = a1 * kr_tmp;
        A = tmp * pinv((C.t() * C) % (B.t() * B));
//        Mat<T> tmpp = (C.t() * C) % (B.t() * B);
//        solve( A, tmp, tmpp);
        A = normalise(A);

        tmp.set_size(n2, r);
        tmp.zeros();
        kr_tmp.set_size(n1, r);
        for (int j = 0; j < n3; j++) {
            for (int i = 0; i < r; i++) {
                kr_tmp.col(i) = C(j, i) * A.col(i);
            }
            cal = a1.cols(j*n2, j*n2+n2-1);
            tmp = tmp + cal.t() * kr_tmp; //slice computing
        }
        B = tmp * pinv((C.t() * C) % (A.t() * A));
//        tmpp = (C.t() * C) % (A.t() * A);
//        solve( B, tmp, tmpp);

        B = normalise(B);


        tmp.set_size(n3, r);
        tmp.zeros();
        cal.set_size(1,n1*n2);
        kr_tmp.set_size(n1*n2,r);
//        for (int j = 0; j < n2; j++) {
//            for (int i = 0; i < r; i++) {
//                kr_tmp.col(i) = B(j, i) * A.col(i);
//            }
//            cal = a1.cols(j*n2, j*n2+n2-1);
//            cal.reshape(n1,n2);
//            tmp = tmp + cal.t() * kr_tmp; //slice computing
//        }
//        a1.set_size(n1*n2,n3);
        Mat<T> a3(n1*n2,n3);
        for (int i = 0; i < r; i++) {
            kr_tmp.col(i) = kron(B.col(i), A.col(i));
        }

        for (int j = 0; j < n3; j++) {
            Mat<T> tmp1 = a.slice(j);
            cal = vectorise(tmp1);
            a3.col(j) = cal;
        }
        tmp = a3.t() * kr_tmp;
//        mat tmpp = (B.t() * B) % (A.t() * A);
//        C = solve(tmpp, tmp);
        C = tmp * pinv((B.t() * B) % (A.t() * A));

        if (turn < 0){
            C = normalise(C);
        }
    }

    cp_mats<T> result0;
    result0.A = A;
    result0.B = B;
    result0.C = C;

//    cout << A << endl;
//    cout << B << endl;
//    cout << C << endl;

    return result0;
}

template<class T>
cp_mats<T> cp_als(Cube<T> &a, int r,int max_iter) {
    int n1=a.n_rows; int n2=a.n_cols; int n3=a.n_slices;  //dimension

    Mat<T> A(n1,r); Mat<T> B=randu<Mat<T>>(n2,r); Mat<T> C=randu<Mat<T>>(n3,r); //random A,B,C

    Mat<T> cal(n1,n2); Mat<T> tmp(n1,r);
    for (int turn=0; turn< max_iter;turn++) {
        tmp.zeros();
        tmp.set_size(n1, r);
        Mat<T> kr_tmp(n2*n3,r);
        Mat<T> tmpp = (C.t() * C) % (B.t() * B);

        tmp.set_size(n2, r);
        tmp.zeros();
        kr_tmp.set_size(n2, r);
        Mat<T> vvv;
        for (int j = 0; j < n3; j++) {
//            vvv = C.row(j);
            for (int i = 0; i < r; i++) {
                register T ddd = C(j,i);
//                T ddd = vvv(i);
                kr_tmp.col(i) = ddd * B.col(i);
            }
            cal = a.slice(j);
//            cal = a1.cols(j*n2, j*n2+n2-1);
            tmp = tmp + cal * kr_tmp; //slice computing
        }
        tmpp = (C.t() * C) % (B.t() * B);
        A = tmp * pinv(tmpp);
        A = normalise(A);

        kr_tmp.set_size(n1, r);
        for (int j = 0; j < n3; j++) {
            for (int i = 0; i < r; i++) {
                T ddd = C(j,i);
                kr_tmp.col(i) = ddd * A.col(i);
            }
            cal = a.slice(j);
//            cal = a1.cols(j*n2, j*n2+n2-1);
            tmp = tmp + cal.t() * kr_tmp; //slice computing
        }

        tmpp = (C.t() * C) % (A.t() * A);
        B = tmp * pinv(tmpp);
        B = normalise(B);

        tmp.set_size(n3, r);
        tmp.zeros();
        cal.set_size(1,n1*n2);
        kr_tmp.set_size(n1*n2,r);

        Mat<T> a3(n1*n2,n3);
        for (int i = 0; i < r; i++) {
            kr_tmp.col(i) = vectorise(B.col(i) *trans(A.col(i)));
        }

        for (int j = 0; j < n3; j++) {
            Mat<T> tmp1 = a.slice(j);
            cal = vectorise(tmp1);
            a3.col(j) = cal;
        }
        tmp = a3.t() * kr_tmp;

//        C = solve(tmpp, tmp);
        tmpp = (B.t() * B) % (A.t() * A);

        C = tmp * pinv(tmpp);

        Mat<T> X_con = kr_tmp * C.t();
        X_con = X_con - a3;

        double error;
        error = sqrt(accu(X_con%X_con));

        if (turn < max_iter-1){
            C = normalise(C);
        }
    }

    cp_mats<T> result0;
    result0.A = A;
    result0.B = B;
    result0.C = C;

//    cout << A << endl;
//    cout << B << endl;
//    cout << C << endl;

    return result0;
}


template<class T>
cp_mats<T> cp_als(Cube<T> &a, int r,int max_iter, T expected_error) {
    int n1=a.n_rows; int n2=a.n_cols; int n3=a.n_slices;  //dimension

    Mat<T> A(n1,r); Mat<T> B=randu<Mat<T>>(n2,r); Mat<T> C=randu<Mat<T>>(n3,r); //random A,B,C

//    Mat<T> a1(n1,n2*n3);
//    a.reshape(n1,n2*n3,1);
//    a1 = a.slice(0);
//    a.reshape(n1,n2,n3);

    Mat<T> cal(n1,n2); Mat<T> tmp(n1,r);
    for (int turn=0; turn< max_iter;turn++) {
        tmp.zeros();
        tmp.set_size(n1, r);
        Mat<T> kr_tmp(n2*n3,r);
        Mat<T> tmpp = (C.t() * C) % (B.t() * B);

        tmp.set_size(n1, r);
        tmp.zeros();
        kr_tmp.set_size(n2, r);
        Mat<T> vvv;
        for (int j = 0; j < n3; j++) {
//            vvv = C.row(j);
            for (int i = 0; i < r; i++) {
                register T ddd = C(j,i);
//                T ddd = vvv(i);
                kr_tmp.col(i) = ddd * B.col(i);
            }
            cal = a.slice(j);
//            cal = a1.cols(j*n2, j*n2+n2-1);
            tmp = tmp + cal * kr_tmp; //slice computing
        }
        tmpp = (C.t() * C) % (B.t() * B);
        A = tmp * pinv(tmpp);
        A = normalise(A);

        kr_tmp.set_size(n1, r);
        tmp.set_size(n2, r);

        for (int j = 0; j < n3; j++) {
            for (int i = 0; i < r; i++) {
                T ddd = C(j,i);
                kr_tmp.col(i) = ddd * A.col(i);
            }
            cal = a.slice(j);
//            cal = a1.cols(j*n2, j*n2+n2-1);
            tmp = tmp + cal.t() * kr_tmp; //slice computing
        }
        tmpp = (C.t() * C) % (A.t() * A);
        B = tmp * pinv(tmpp);
        B = normalise(B);

        tmp.set_size(n3, r);
        tmp.zeros();
        cal.set_size(1,n1*n2);
        kr_tmp.set_size(n1*n2,r);

        Mat<T> a3(n1*n2,n3);
        for (int i = 0; i < r; i++) {
            kr_tmp.col(i) = vectorise(B.col(i) *trans(A.col(i)));
        }

        for (int j = 0; j < n3; j++) {
            Mat<T> tmp1 = a.slice(j);
            cal = vectorise(tmp1);
            a3.col(j) = cal;
        }
        tmp = a3.t() * kr_tmp;

//        C = solve(tmpp, tmp);
        tmpp = (B.t() * B) % (A.t() * A);

        C = tmp * pinv(tmpp);

        Mat<T> X_con = kr_tmp * C.t();
        X_con = X_con - a3;

        T error;
        error = sqrt(accu(X_con%X_con));

        if (error <= expected_error){
            cp_mats<T> result0;
            result0.A = A;
            result0.B = B;
            result0.C = C;
            return result0;
        }
        else if(turn < max_iter-1){
            C = normalise(C);
        } else{
            cp_mats<T> result0;
            result0.A = A;
            result0.B = B;
            result0.C = C;
            return result0;
        }
    }

    cp_mats<T> result0;
    result0.A = A;
    result0.B = B;
    result0.C = C;

    return result0;
}
