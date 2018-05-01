#include <iostream>
#include "tensor.h"
using namespace std;

int main() {
    Tensor a(5,5,5),b(2,3,4), c(4,3,2), z(1,2,3);
    z=z.zeros(2,3,4);  //零张量
    b=b.Identity(3,3,3); //单位张量
    b=Transpose(b); //转置
    b(1,2,2)=2;
    c=b;    //类的相等
    double x=dotProduct(b,c); //张量内积
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
