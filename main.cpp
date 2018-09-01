#include <iostream>
#include "tensor.h"
using namespace std;

#include <complex>
#include <fftw3.h>
#include <math.h>
#include <iostream>
#include <armadillo>

#include <iostream>
#include <vector>

#include "time.h"

using namespace std;
using namespace arma;

#define N 8

int main() {
    Tensor a(5,5,5), b(2,3,5),d(5,5,5), z(2,3,4),t(5,5,5);
    cout<<a(1,2,4)<<endl;
    cout<<z(0,1,2)<<endl;

    mat M1 = ten2mat(z,1);
    vec m1 = ten2vec(z);
    cout << m1 << endl;
    cout << M1 <<endl;
//    mat M2 = ten2mat(z,2);
//    cout << M2 <<endl;
//    mat M3 = ten2mat(z,3);
//    cout << M3 <<endl;
    TM xo=HOSVD(z,1,2,3);

//    cout << xo.m1 <<endl <<xo.m2 <<endl;

    z=z.zeros(2,3,4);
    cout<<z(0,1,2)+123<<endl;
      //z=b.Identity(2,2,3);
 //   cout<<b(0,1,2)<<endl;
 //   b=Transpose(b);
 //   b(1,2,2)=2;
    //double x=dotProduct(b,c);
    //cout<<x<<endl;

//    for(int k=0; k<3; ++k) {
//    for (int i = 0; i < 3; ++i) {
//        for (int j = 0; j < 3; ++j) {
//                cout<<b(i,j,k);
//            }
//            cout<<endl;
//        }
//        cout<<endl;
//    }
    a*=1;
    int *c=size(b);
    cout<<c[0]<<endl; //tensor大小
//    cout<<sizeof(a)<<endl;
    cout<<norm(a)<<endl;
    cout<<a(1,2,4)+3<<endl;
    
    cout << "Hello, World!" << endl;
    t=tprod(a,d);
    cout<<t(1,2,3)<<endl;
    cout<<norm(a)<<endl;
    cout<<fiber(a,1,2,2)<<endl;
    cout<<slice(a,1,1)(0,0)<<endl;

    return 0;

}