//
// Created by jcfei on 4/30/18.
//
//#include <stdexcept>
#include <iostream>
#include <cmath>
#include "tensor.h"

using namespace std;

template<class T>
tucker_core<T> hooi(Cube<T> &a, int r1, int r2, int r3) {
    int n1=a.n_rows; int n2=a.n_cols; int n3=a.n_slices;

    Mat<T> tmp(n1,n1);
//    Mat<T> tmp2(n2,n2);

//mode-1
    tmp.zeros();
//    tmp2.zeros();
    Mat<T> cal(n1,n2);

    for (int i=0; i< n3;i++){
        cal = a.slice(i);
        tmp = tmp + cal*cal.t();
//        tmp2 = tmp2+ cal.t()*cal;
    }

    Mat<T> U;//U,V均为正交矩阵
    Col<T> S;//S为奇异值构成的列向量
    Mat<T> U1(n1,r1);
    eig_sym(S,U,tmp);
    U1 = U.cols(n1-r1,n1-1);

//mode-2
    tmp.resize(n2,n2);
    tmp.zeros();
    for (int i=0; i< n3;i++){
        cal = a.slice(i);
        tmp = tmp + cal.t()*cal;
    }
    eig_sym(S,U,tmp);
    Mat<T> U2(n2,r2);
    U2 = U.cols(n2-r2,n2-1);

//mode-3
    tmp.resize(n3,n3);
    cal.resize(n1,n3);
    for (int j = 0; j < n2; j++) {
         cal = a.tube(span(0,n3-1),span(j,j));
         tmp = cal * cal.t();
    }
    Mat<T> U3(n3,r3);
    eig_sym(S,U,tmp);
    U3 = U.cols(n3-r3,n3-1);
    U.reset();S.reset();

    tmp.resize(r1,r2);
    Mat<T> a1(r1*r2,n3);
    for (int i=0; i< n3;i++){
        cal = a.slice(i);
        tmp = U1.t()*cal*U2;
        cal = vectorise(tmp);
        a1.col(i) = cal;
    }

    Cube<T> g(r1,r2,r3);

    Col<T> ctmp(r1*r2,1);
    a1 = a1*U3;
    for (int k=0; k< r3;k++){
        ctmp = a1.col(k);
        g.slice(k) = reshape(ctmp,r1,r2);
    }

    tucker_core<T> A;
    A.u1 = U1;
    A.u2 = U2;
    A.u3 = U3;
    A.core = g;
    return A;
}

