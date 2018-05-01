#include <iostream>
#include "tensor.h"
using namespace std;

int main() {
    Tensor a(5,5,5),b(2,3,4), c(4,3,2), z(1,2,3);
    z=z.zeros(2,3,4);
    b=b.Identity(3,3,3);
    b=Transpose(b);
    b(1,2,2)=2;
    c=b;    //类的相等
    double x=dotProduct(b,c);
    cout<<x<<endl;

//    for(int k=0; k<3; ++k) {
//    for (int i = 0; i < 3; ++i) {
//        for (int j = 0; j < 3; ++j) {
//                cout<<b(i,j,k);
//            }
//            cout<<endl;
//        }
//        cout<<endl;
//    }

    cout<<z(1,2,3)<<endl;
    cout<<norm(a)<<endl;
    cout<<a(1,2,4)+3<<endl;
    cout << "Hello, World!" << endl;
    return 0;
}