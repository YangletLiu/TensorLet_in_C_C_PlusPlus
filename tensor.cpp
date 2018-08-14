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
                p[i][j][k] = 1;
            }
        }
    }
}

Tensor::Tensor() :n1(1),n2(1),n3(1){
    allocSpace();
    p[0][0][0]=1;
}
//析构函数
Tensor::~Tensor() {
    for (int i=0; i<n1;i++) {
        for (int j=0;j<n2;j++){
            delete[] p[i][j];
        }
        delete[] p[i];
    }
    delete[] p;
    p=NULL;
//    cout<<"destructing T"<<endl;
}

Tensor::Tensor(const Tensor & T) {
    allocSpace();
    for (int i=0; i< T.n1; ++i){
        for(int j=0;j<T.n2;++j){
            for(int k=0; k<T.n3;++k){
                p[i][j][k]=T.p[i][j][k];
            }
        }
    }
}


//重载运算符
Tensor &Tensor::operator=(const Tensor & T) {
    if(this == &T){
        return *this;
    }
    if(n1 != T.n1 || n2 != T.n2 || n3 != T.n3){
        for (int i=0; i<n1;i++) {
            for (int j=0;j<n2;j++){
                delete[] p[i][j];
            }
            delete[] p[i];
        }
        delete[] p;
        n1=T.n1;n2=T.n2;n3=T.n3;
        allocSpace();
    }
    for (int i=0; i<n1; ++i){
        for (int j=0; j<n2;++j){
            for(int k=0;k<n3;++k){
                p[i][j][k]=T.p[i][j][k];
            }
        }
    }
    return *this;
}

Tensor &Tensor::operator+=(const Tensor & T) {
    if (n1 == T.n1 && n2 == T.n2 && n3 == T.n3){
    for (int i=0; i<T.n1; ++i){
        for (int j=0; j<T.n2;++j){
            for(int k=0;k<T.n3;++k){
                p[i][j][k]+=T.p[i][j][k];
            }
        }
    }
    return *this;}
    else{
        cout<<"check size match"<<endl;
        exit(0);
    }
}

Tensor &Tensor::operator-=(const Tensor & T) {
    if (n1 == T.n1 && n2 == T.n2 && n3 == T.n3){
        for (int i=0; i<T.n1; ++i){
            for (int j=0; j<T.n2;++j){
                for(int k=0;k<T.n3;++k){
                    p[i][j][k]-=T.p[i][j][k];
                }
            }
        }
        return *this;}
    else{
        cout<<"check size match"<<endl;
        exit(0);
    }
}

Tensor &Tensor::operator*=(const Tensor & T) {
    if (n1 == T.n1 && n2 == T.n2 && n3 == T.n3){
        for (int i=0; i<T.n1; ++i){
            for (int j=0; j<T.n2;++j){
                for(int k=0;k<T.n3;++k){
                    p[i][j][k]*=T.p[i][j][k];
                }
            }
        }
        return *this;}
    else{
        cout<<"check size match"<<endl;
        exit(0);
    }
}

Tensor &Tensor::operator*=(double num) {
    for (int i=0; i<n1; ++i){
        for (int j=0; j<n2;++j){
            for(int k=0;k<n3;++k){
                p[i][j][k]*=num;
            }
        }
    }
    return *this;
}

Tensor &Tensor::operator/=(double num) {
    if (num != 0){
        for (int i=0; i<n1; ++i){
            for (int j=0; j<n2;++j){
                for(int k=0;k<n3;++k){
                    p[i][j][k]/=num;
                }
            }
        }
    }
    else {
        cout << "divide number is zero."<< endl;
        exit(0);
    }
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
    cout<<"zeros Tensor"<<endl;
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
//size
int* size(const Tensor & a) {
    static int p[3]; 
    p[0]=a.n1;
    p[1]=a.n2;
    p[2]=a.n3;
    return p;
}
//diber
double *fiber(const Tensor & t, int m, int n ,int a) {
    if(a==1){
        double *c=new double [t.n1];
        for (int i=0; i<t.n1;++i){
                c[i]=t.p[i][m][n];
            }
            return c;
        }
    if(a==2){
        double *c=new double [t.n2];
        for (int i=0; i<t.n1;++i){
            c[i]=t.p[i][m][n];
        }
        return c;
    }
    if(a==3){
        double *c=new double [t.n3];
        for (int i=0; i<t.n1;++i){
            c[i]=t.p[i][m][n];
        }
        return c;
    }
}

double **slice(const Tensor & t, int m, int a) {
    if(a==1){
        double **c=new double*  [t.n2];
        for (int i=0;i<t.n2;++i){
            c[i]=new double [t.n3];
        }
        for (int i=0;i<t.n2;++i){
            for (int j=0;j<t.n3;++j){
                c[i][j]=t.p[m][i][j];
            }
        }
        return c;
    }
    if(a==2){
        double **c=new double* [t.n1];
        for (int i=0;i<t.n1;++i){
            c[i]=new double [t.n3];
        }
        for (int i=0;i<t.n1;++i){
            for (int j=0;j<t.n3;++j){
                c[i][j]=t.p[i][m][j];
            }
        }
        return c;
    }
    if(a==3){
        double **c=new double* [t.n1];
        for (int i=0;i<t.n1;++i){
            c[i]=new double [t.n2];
        }
        for (int i=0;i<t.n1;++i){
            for (int j=0;j<t.n2;++j){
                c[i][j]=t.p[i][j][m];
            }
        }
        return c;
    }
}

//求范数
double norm(Tensor &a) {
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

//转置运算；
Tensor Transpose(Tensor &a) {
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

Tensor tprod(Tensor & a, Tensor & b) {
    Tensor tmp(a.n1,b.n2,a.n3);
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
