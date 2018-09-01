//
// Created by jcfei on 4/30/18.
//

#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include "string.h"
#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;


class Tensor{
public:
    Tensor(int,int,int); //创建三维张量
    Tensor();
    ~Tensor();
    Tensor(const Tensor&);
    Tensor& operator=(const Tensor&);
    Tensor& operator+=(const Tensor&);
    Tensor& operator-=(const Tensor&);
    Tensor& operator*=(const Tensor&);
    Tensor& operator*=(double);
    Tensor& operator/=(double);
    //返回值
    inline double& operator()(int x, int y, int z) {
        if (x<n1 && y<n2 && z<n3) {
            return p[x][y][z];
        }
        else {
            cout<<"Warning: Index beyond range."<<endl;
            exit(0);
        }
    }

    //初始化
    static Tensor zeros(int, int, int); //零张量
    static Tensor Identity(int, int, int); //零张量

    friend int *size(const Tensor &);
    friend vec fiber(const Tensor &,int,int,int);
    friend mat slice(const Tensor &,int,int);
    friend double norm(Tensor &);
    friend Tensor Transpose(Tensor &);
    friend double dotProduct(Tensor,Tensor);
    friend Tensor tprod(Tensor &,Tensor &);
    friend mat ten2mat(Tensor &, int);
    friend vec ten2vec(Tensor &);
//    private:
    int n1,n2,n3;
    double ***p;
    void allocSpace();
};

struct TM{
    Tensor & core;
    mat u1,u2,u3;
};

TM HOSVD(Tensor &, int , int , int );

//n-mode product
mat ttm(Tensor &, mat &, int);


#endif //TENSOR_TENSOR_H
