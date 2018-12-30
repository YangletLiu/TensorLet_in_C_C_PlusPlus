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

    cube a(n1,n2,n3);
    a.randn();
    cout << a.slice_memptr(0) << endl;

//    t0=gettime();
//    cube b = cpgen<double>(n1,n2,n3,R);
//    t1=gettime();
//    cout << "time :" << t1-t0 << endl;


    int iter=10;
    double ccc =0.05;
    t0=gettime();
    cp_mats<double> B;
    B = cp_als(a, R,iter,ccc);
    t1=gettime();
    cout << "time:" <<t1-t0 <<endl;

//    R=1;
//    t0=gettime();
//    tucker_core<double> result_tucker;
//    result_tucker = hosvd(a,R,R,R);
//    t1=gettime();
//    cout << "time:" << t1-t0 << endl;


//    t0=gettime();
//    tucker_core<double> result_tucker;
//    result_tucker = hosvd(a,R,R,R);
//    t1=gettime();
//    cout << "time:" <<t1-t0 <<endl;

//    t0=gettime();
//    tsvd_core<double> result_tucker;
//    result_tucker = tsvd(a);
//    t1=gettime();
//    cout << "time:" <<t1-t0 <<endl;

//    double t3,t4;
//    mat b=randn(500,500);
//    t3=gettime();
//    for (int i=0;i<500;i++){
//        mat c = b*b.t();
//    }
//    t4=gettime();
//    cout << "time:" <<t4-t3 <<endl;

    return 0;
}
