//
// Created by jcfei on 18-9-26.
//

#include "tensor.h"
#include <iostream>
#include <cmath>
using namespace std;

//返回结构
template<class T>
struct tucker_core{
    Mat<T> u1, u2, u3;
};

template<class T>
tucker_core<T> hosvd(Tensor<T> &a, int r1, int r2, int r3) {
    int n1=a.n1; int n2=a.n2; int n3=a.n3;
    Mat<T> tmp(n3,n3);
    Mat<T> cal(n1,n3);

    Mat<T> a1 = ten2mat(a,3);
    tmp = a1*a1.t();

    Mat<T> U;//U,V均为正交矩阵
    Col<T> S;//S为奇异值构成的列向量
    Col<T> U3(n3,r3);
    eig_sym(S,U,tmp);
    U3 = U.cols(n3-r3,n3-1);

    tmp.resize(n1,n1);
    cal.resize(n1,n2);
//    m1.reset(); //直接改变矩阵大小...内存管理好神奇。。。

    a1.set_size(n1,n2*n3);
    a1 = ten2mat(a,1);
    tmp= a1*a1.t();

    eig_sym(S,U,tmp);
    Mat<T> U1(n1,r1);
    U1 = U.cols(n1-r1,n1-1);

    tmp.resize(n2,n2);

    a1.set_size(n2,n1*n3);
    a1 = ten2mat(a,2);
    tmp= a1*a1.t();

    eig_sym(S,U,tmp);
    Mat<T> U2(n2,r2);
    U2 = U.cols(n2-r2,n2-1);

    U.reset();S.reset();
//    S.reset();U.reset();
    Tensor<T>  g_tmp(r1,r2,n3);
    tmp.resize(r1,r2);
    for (int i=0; i< n3;i++){
        cal = slice(a,i,3);
        tmp = U1.t()*cal*U2;
        for(int j=0;j<r1;j++){
            for(int k=0;k<r2;k++){
                g_tmp(j,k,i) = tmp(j,k);
            }
        }
    }
//    tmp.reset();

    Tensor<T> g(r1,r2,n3);
    cal.resize(r1,r2);
    tmp.resize(r1,r3);
    for (int i=0; i< r2;i++){
        cal = slice(g_tmp,i,2);
        tmp = cal.t()*U3.t();
        for(int j=0;j<r1;j++){
            for(int k=0;k<r3;k++){
                g(j,k,i) = tmp(j,k);
            }
        }
    }

//    cout << U1 << endl;
//    cout << U2 << endl;
//    cout << U3 << endl;

    tucker_core<T> A;
    A.u1 = U1;
    A.u2 = U2;
    A.u3 = U3;
    return A;
}
