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
fmat slice(const Tensor<T> &t, int m, int order) {
    if(order==1){
        fmat c(t.n2,t.n3);
        for (int i=0;i<t.n2;++i){
            for (int j=0;j<t.n3;++j){
                c(i,j)=t.p[m][i][j];
            }
        }
        return c;
    }
    if(order==2){
        fmat c(t.n1,t.n3);
        for (int i=0;i<t.n1;++i){
            for (int j=0;j<t.n3;++j){
                c(i,j)=t.p[i][m][j];
            }
        }
        return c;
    }
    if(order==3){
        fmat c(t.n1,t.n2);
        for (int i=0;i<t.n1;++i){
            for (int j=0;j<t.n2;++j){
                c(i,j)=t.p[i][j][m];
            }
        }
        return c;
    }
}