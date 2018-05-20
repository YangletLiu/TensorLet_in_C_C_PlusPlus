//
// Created by jcfei on 4/30/18.
//

#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include "string.h"
#include <iostream>
using namespace std;

class Tensor{
public:
    Tensor(int,int,int); //创建三维张量
    Tensor();
    ~Tensor();
    static Tensor zeros(int, int, int); //零张量
    static Tensor Identity(int, int, int); //零张量

    inline double& operator()(int x, int y, int z) {
        if (x<n1 && y<n2 && z<n3) {
            return p[x][y][z];
        }
        else {
            cout<<"Warning: Index beyond range."<<endl;
            exit(0);
        }
    }

    friend double norm(Tensor);
    friend Tensor Transpose(Tensor);
    friend double dotProduct(Tensor,Tensor);

    private:
    int n1,n2,n3;
    double ***p;
    void allocSpace();
};



#endif //TENSOR_TENSOR_H
