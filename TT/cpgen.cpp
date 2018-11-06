//
// Created by jcfei on 18-9-28.
//

#include "tensor.h"
#include <iostream>
#include <cmath>
using namespace std;

template <class T>
Cube<T> cpgen(int n1,int n2,int n3,int r);

template<class T>
Cube<T> cpgen(int n1, int n2, int n3, int r) {
    Cube<T> result = zeros(n1,n2,n3);
    Mat<T> A = randu<Mat<T>> (n1,r); Mat<T> B=randu<Mat<T>>(n2,r); Mat<T> C=randu<Mat<T>>(n3,r); //random A,B,C
//    cout << A << endl << B << endl << C << endl;

    for(int k=0;k<n3;k++){
        Mat<T> tmp1=zeros(n1,n2);
        for(int i=0;i<r;i++){
            Mat<T> tmp ;
            tmp = kron(B.col(i), A.col(i));
            tmp.reshape(n1,n2);
            tmp =  C(k,i) * tmp;
            tmp1 += tmp;
        }
        result.slice(k) = tmp1;
    }
//    cout << "generate: " << result << endl;
    return result;
}


//simple original
//    result = zeros(n1,n2,n3);
//    for(int i=0;i<n1;i++){
//        for(int j=0;j<n2;j++){
//            for(int k=0;k<n3;k++){
//                T tmp=0; //类型错误...
//                for(int r0=0; r0<r;r0++){
//                    tmp = tmp + A(i,r0) * B(j,r0) * C(k,r0);
//                }
//                result(i,j,k) = tmp;
//            }
//        }
//    }
//    cout << "generate: " << result << endl;


//template<class T>
//Cube<T> cpgen(int n1, int n2, int n3, int r) {
//    Cube<T> result = zeros(n1*n2*n3,1,1);
//    Mat<T> A = randu<Mat<T>> (n1,r); Mat<T> B=randu<Mat<T>>(n2,r); Mat<T> C=randu<Mat<T>>(n3,r); //random A,B,C
////    cout << A << endl << B << endl << C << endl;
//
//    Col<T> tmp1=zeros(n1*n2*n3);
//    for(int i=0;i<r;i++){
//        Mat<T> tmp ;
//        tmp = kron(B.col(i), A.col(i));
//        result.slice(0) += kron(C.col(i),tmp);
//    }
//    reshape(result,n1,n2,n3);
////    cout << "generate: " << result << endl;
//
//    return result;
//}
