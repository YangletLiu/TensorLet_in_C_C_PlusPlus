//
// Created by jcfei on 18-10-3.
//

#include "tensor.h"

template<class T>
Mat<T> ten2mat(Cube<T> &a, int order) {
    int n1 = a.n_rows;
    int n2 = a.n_cols;
    int n3 = a.n_slices;

    if (order == 1) {
        Mat<T> a1(n1,n2*n3);
        a1 = reshape(a,n1,n2*n3,1);
        return a1;
    }
    if (order == 2){
        Mat<T> a2(n2,n1*n3);
        Mat<T> *cal;
        for (int k=0; k<n3; k++){
            cal = &a.slice(k);
            a2.cols(k*n1,k*n1+n1-1) = (*cal).t();
        }
        return a2;
    }
    if (order == 3){
        Mat<T> a3(n1*n2,n3);
        for (int j = 0; j < n3; j++) {
            Mat<T> tmp1 = a.slice(j);
            a3.col(j) = vectorise(tmp1);
        }
        return a3.t();
    }
};