//
// Created by jcfei on 4/30/18.
//
//#include <stdexcept>
#include "tensor.h"
#include <iostream>
using namespace std;

//定义三维张量，初始化为0
Tensor::Tensor(int n1, int n2, int n3) : n1(n1), n2(n2),n3(n3)
{
    allocSpace();
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            for(int k=0; k<n3; ++k) {
                p[i][j][k] = 0;
            }
        }
    }
}


/******Tensor privative helper functions******
*********************************************/
//分配内存空间（三维动态数组）
void Tensor::allocSpace()
{
    p =new double**[n1];
    for (int i=0; i<n1;i++) {
        p[i] = new double *[n2];
        for (int j=0;j<n2;j++){
            p[i][j]=new double [n3];
        }
    }
}