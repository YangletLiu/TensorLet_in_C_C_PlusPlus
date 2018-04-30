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
    Tensor(int,int,int);
    inline double& operator()(int x, int y, int z) { return p[x][y][z]; }

private:
    int n1,n2,n3;
    double ***p;   
    void allocSpace();
};



#endif //TENSOR_TENSOR_H
