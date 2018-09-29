//
// Created by jcfei on 4/30/18.
//
//#include <stdexcept>
#include "tensor.h"
#include <iostream>
#include <cmath>
using namespace std;


/*********************************************
 ******Tensor privative helper functions******
 *********************************************/

template<class T>
Mat<T> slice(const Tensor<T> &t, int m, int order) {
    if(order==1){
        Mat<T> c(t.n2,t.n3);
        for (int i=0;i<t.n2;++i){
            for (int j=0;j<t.n3;++j){
                c(i,j)=t.p[m][i][j];
            }
        }
        return c;
    }
    if(order==2){
        Mat<T> c(t.n1,t.n3);
        for (int i=0;i<t.n1;++i){
            for (int j=0;j<t.n3;++j){
                c(i,j)=t.p[i][m][j];
            }
        }
        return c;
    }
    if(order==3){
        Mat<T> c(t.n1,t.n2);
        for (int i=0;i<t.n1;++i){
            for (int j=0;j<t.n2;++j){
                c(i,j)=t.p[i][j][m];
            }
        }
        return c;
    }
}

template<class T>
double norm(Tensor<T> &a) {
    double norm=0;
    for (int i = 0; i < a.n1; ++i) {
        for (int j = 0; j < a.n2; ++j) {
            for(int k=0; k< a.n3; ++k) {
                norm=norm+a.p[i][j][k]*a.p[i][j][k];
            }
        }
    }
    return norm;
}

template<class T>
Tensor<T> Transpose(Tensor<T> &a) {
    Tensor<T> tem=a;
    for (int i = 0; i < a.n1; ++i) {
        for (int j = 0; j < a.n2; ++j) {
            for(int k=0; k< a.n3; ++k) {
                if (k==0) tem.p[i][j][k]=a.p[j][i][k];
                else tem.p[i][j][k]=a.p[j][i][a.n3-k];
            }
        }
    }
    return tem;
}

template<class T1, class T2>
double dotProduct(Tensor<T1> a, Tensor<T2> b) {
    double sum = 0;
    if(a.n1==b.n1 && a.n2==b.n2 && a.n3==b.n3) {
        for (int i = 0; i < a.n1; ++i) {
            for (int j = 0; j < a.n2; ++j) {
                for (int k = 0; k < a.n3; ++k) {
                    sum = sum + a.p[i][j][k] * b.p[i][j][k];
                }
            }
        }
    }
    else{
        cout<<"Warning: The size is not match."<<endl;
        exit(0);
    }
    return sum;
}

template<class T1, class T2>
Tensor<T1> tprod(Tensor<T1> &a, Tensor<T2> &b) {
    Tensor<T1> tmp(a.n1,b.n2,a.n3);
    tmp=tmp.zeros(a.n1,b.n2,a.n3);
    if (a.n2==b.n1 && a.n3==b.n3){
        for(int k=0;k<a.n3;++k){
            for (int i=0;i<a.n1;i++){
                for (int j=0;j<b.n2;++j){
                    double s=0;
                    for (int l=0;l<a.n2;++l){
                        s=s+a.p[i][l][k]*b.p[l][j][k];
                    }
                    tmp.p[i][j][k]=s;
                }
            }
        }
        return tmp;
    }
}

template<class T>
mat ttm(Tensor<T> &a, mat &b, int c) {
    mat result;
    mat a_c = ten2mat(a,c);
    result = b*a_c;
    return result;
}