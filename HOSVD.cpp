//
// Created by jcfei on 18-9-9.
//
#include "tensor.h"
#include <iostream>
#include <cmath>
using namespace std;


//返回结构
template<class T>
struct tuckercore{
    Tensor<T> & core;
    fmat u1,u2,u3;
};


template<class T>
tuckercore <T> HOSVD(Tensor<T> &a, int r1, int r2, int r3) {
    int n1=a.n1; int n2=a.n2; int n3=a.n3;
    fmat tmp(n3,n3);
    fmat cal(n1,n3);
    tmp.zeros();

    for (int i=0; i< n2;i++){
        cal = slice(a,i,2);
        tmp = tmp + cal.t()*cal; //slice computing
    }

    fmat U;//U,V均为正交矩阵
    fvec S;//S为奇异值构成的列向量
    fmat U3(n3,r3);
    eig_sym(S,U,tmp);
    U3 = U.cols(n3-r3,n3-1);

    tmp.resize(n1,n1);
    cal.resize(n1,n2);
    tmp.zeros();
//    m1.reset(); //直接改变矩阵大小 不需要删除堆栈？？内存管理好神奇。。。

    for (int i=0; i< n3;i++){
        cal = slice(a,i,3);
        tmp = tmp + cal*cal.t();
    }

    eig_sym(S,U,tmp);
    fmat U1(n1,r1);
    U1 = U.cols(n1-r1,n1-1);

    tmp.resize(n2,n2);
    tmp.zeros();
    for (int i=0; i< n3;i++){
        cal = slice(a,i,3);
        tmp = tmp + cal.t()*cal;
    }

    eig_sym(S,U,tmp);
    fmat U2(n2,r2);
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

    tuckercore<T> A{g,U1,U2,U3};
    return A;
}
