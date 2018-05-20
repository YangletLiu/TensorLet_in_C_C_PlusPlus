#include <iostream>
#include "tensor.h"
using namespace std;

int main() {
    Tensor a(5,5,5), b(2,3,4);

    cout<<a(1,2,4)<<endl;

//    z=z.zeros(2,3,4);   //为什么会被析构了。。。
 //   cout<<z(0,1,2)<<endl;
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
    cout<<norm(a)<<endl;
    cout<<a(1,2,4)+3<<endl;
    cout << "Hello, World!" << endl;
    return 0;
}