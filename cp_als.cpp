//
// Created by jcfei on 18-9-9.
//

#include "tensor.h"
#include <iostream>
#include <cmath>
using namespace std;

template<class T>
fmat cp_als(Tensor<T> &a, int r) {
    int n1=a.n1; int n2=a.n2; int n3=a.n3;
    fmat A=randu<fmat>(n1,r); fmat B=randu<fmat>(n2,r); fmat C=randu<fmat>(n3,r); //randu(fmat)

    fmat cal(n1,n2); fmat tmp(n1,r); fmat krtmp(n2,r);

    for (int turn=0; turn<1;turn++) {
        tmp.zeros();
        for (int j = 0; j < n3; j++) {
            for (int i = 0; i < r; i++) {
                krtmp.col(i) = C(j, i) * B.col(i);
            }
            cal = slice(a, j, 3);
            tmp = tmp + cal * krtmp; //slice computing
        }
        A = tmp * inv((C.t() * C) % (B.t() * B));
        A = normalise(A);

        tmp.set_size(n2, r);
        tmp.zeros();
        krtmp.set_size(n1, r);
        for (int j = 0; j < n3; j++) {
            for (int i = 0; i < r; i++) {
                krtmp.col(i) = C(j, i) * A.col(i);
            }
            cal = slice(a, j, 3);
            tmp = tmp + cal.t() * krtmp; //slice computing
        }
        B = tmp * inv((C.t() * C) % (A.t() * A));
        B = normalise(B);

        tmp.set_size(n3, r);
        tmp.zeros();
        krtmp.set_size(n1, r);
        for (int j = 0; j < n2; j++) {
            for (int i = 0; i < r; i++) {
                krtmp.col(i) = B(j, i) * A.col(i);
            }
            cal = slice(a, j, 2);
            tmp = tmp + cal.t() * krtmp; //slice computing
        }
        C = tmp * inv((B.t() * B) % (A.t() * A));
        if (turn != 1){
            C = normalise(C);
        }
    }

//    Tensor<T> t_con(n1,n2,n3);
//
//    for (int i=0;i<n1;i++){
//        for (int j=0;j<n2;j++){
//            for(int k=0;k<n3;k++){
//                T sum = 0;
//                for (int l=0;l<r;l++){
//                    sum = sum + A(i,l)*B(j,l)*C(k,l);
//                }
//                t_con(i,j,k) = sum;
//            }
//        }
//    }
//    mat m1=ten2mat(t_con,1);
//    mat m2=ten2mat(a,1);
//    cout << m1-m2 << endl;

//    cout << A << endl;
//    cout << B << endl;
//    cout << C << endl;

    return B;
}
