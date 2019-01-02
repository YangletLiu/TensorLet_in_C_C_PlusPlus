#include "tensor.h"
#include "runningtime.h"

#include "ten2mat.cpp"
#include "cpgen.cpp"
#include "cp_als.cpp"
#include "tucker_hosvd.cpp"
#include "tensor_hooi.cpp"
#include "t_svd.cpp"


int main(){
    int n1,n2,n3;
    double t0,t1;
    int I=100;
    int R=10;
    //    bool cc= true;

    n1=I; n2=2*I; n3=I;

    int iter=10;
    double ccc =0.05;
    t0=gettime();

    t1=gettime();
    cout << "time:" <<t1-t0 <<endl;

    return 0;
}
