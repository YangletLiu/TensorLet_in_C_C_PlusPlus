//
// Created by jcfei on 4/30/18.
//
//#include <stdexcept>
#include "tensor.h"
#include <iostream>
#include <cmath>
using namespace std;

//定义三维张量，初始化为0
Tensor::Tensor(int n1, int n2, int n3) : n1(n1), n2(n2),n3(n3)
{
    allocSpace();
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            for(int k=0; k<n3; ++k) {
                p[i][j][k] = 2.1;
            }
        }
    }
}

Tensor::Tensor() :n1(1),n2(1),n3(1){
    allocSpace();
    p[0][0][0]=1;
}

Tensor::~Tensor() {
    for (int i=0; i<n1;i++) {
        for (int j=0;j<n2;j++){
            delete[] p[i][j];
        }
        delete[] p[i];
    }
    delete[] p;
    p=NULL;
    cout<<"destructing T"<<endl;
}

//zeros 张量
Tensor Tensor::zeros(int n1, int n2, int n3) {
    Tensor tem(n1,n2,n3);
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            for(int k=0; k<n3; ++k) {
                tem.p[i][j][k] = 0;
            }
        }
    }
    cout<<"hh"<<endl;
    return tem;
}

//单位张量
Tensor Tensor::Identity(int n1, int n2, int n3) {
    if(n1==n2){
        Tensor tem(n1,n2,n3);
        for (int i = 0; i < n1; ++i) {
            for (int j = 0; j < n2; ++j) {
                for(int k=0; k<n3; ++k) {
                    if(k==0 && i==j) {
                    tem.p[i][j][k] = 1;
                    }
                    else {
                        tem.p[i][j][k]=0;
                    }
                }
            }
        }
        return tem;
    }
        else{
            cout<<"Warning: no identity tensor";
            exit(0);
        }
    }

//求范数
double norm(Tensor a) {
    double norm=0;
        for (int i = 0; i < a.n1; ++i) {
            for (int j = 0; j < a.n2; ++j) {
                for(int k=0; k< a.n3; ++k) {
                    norm=norm+a.p[i][j][k]*a.p[i][j][k];
                }
            }
        }
        return norm;
    }       //写在类外时，不能在前面加friend.   //为什么要写两个Tensor,第一个tensor是表示类型

//转置运算；不支持复数暂时
Tensor Transpose(Tensor a) {
    Tensor tem=a;
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

//Innerproduct
double dotProduct(Tensor a, Tensor b) {
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

/*********************************************
 ******Tensor privative helper functions******
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





//提取某一坐标值
//double& Tensor::operator()(int x, int y, int z)
//{
//    if (x<n1 && y<n2 && z<n3)
//   {
//        return p[x][y][z];
//    }
//    else
//        {
//        cout<<"Error: Index beyond range."<<endl;
//        exit(0);
//        }
//}

//注释快捷键 ctrl + /

//展示单位张量的值
//for(int k=0; k<3; ++k) {
//    for (int i = 0; i < 3; ++i) {
//        for (int j = 0; j < 3; ++j) {
//            cout<<b(i,j,k);
//            }
//        cout<<endl;
//    }
//    cout<<endl;
//}
