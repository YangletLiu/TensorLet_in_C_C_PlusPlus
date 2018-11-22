#include "tensor.h"

#include "cpgen.cpp"
#include "cp_als.cpp"
#include "tucker_hosvd.cpp"
#include "tensor_hooi.cpp"
#include "tsvd.cpp"

double gettime(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return (tv.tv_sec*1000+tv.tv_usec/1000.0)/1000.0; //time:s
};

template <class T>
int s(Cube<T> &a) {
    return a.n_rows;
}

int main(){
    int n1,n2,n3;
    double t0,t1;
    int I=800;
    int R=0.2*I;
    //    bool cc= true;

    n1=I; n2=I; n3=I;

    cube a = randu<cube>(n1,n2,n3);
    cout << a.slice_memptr(0) << endl;
//    cout << a.subcube(0,0,1,2,2,3);

//    t0=gettime();
//    cube b = cpgen<double>(n1,n2,n3,R);
//    t1=gettime();
//    cout << "time :" << t1-t0 << endl;

//    mat tes=zeros(200,200);
//    cout << pinv(tes,'dc');

//    int r=R;
//    mat A = randn<mat>(n1,r); mat B=randn<mat>(n2,r); mat C=randn<mat>(n3,r);
//    for(int i=0;i<n1;i++){
//        for(int j=0;j<n2;j++){
//            for(int k=0;k<n3;k++){
//                double tmp=0.0; //类型错误...
//                for(int r0=0; r0<R;r0++){
//                    tmp = tmp + A(i,r0) * B(j,r0) * C(k,r0);
//                }
//                a(i,j,k) = tmp;
//            }
//        }
//    }
//    cout << a << endl;

//    cout << a << endl;
//    inplace_trans(a.slice(0));
//    cout << a.slice(0)<<endl;

//    t0=gettime();
//    cpals(a,R);
//    t1=gettime();
//    cout << "time:" <<t1-t0 <<endl;

//    int iter=1;
//    t0=gettime();
//    cp_mats<double> B;
//    B = cp_als(a, R,iter);
//    t1=gettime();
//    cout << "time:" <<t1-t0 <<endl;

//    t0=gettime();
//    tucker_core<double> result_tucker;
//    result_tucker = hosvd(a,R,R,R);
//    t1=gettime();
//    cout << "time:" <<t1-t0 <<endl;

    t0=gettime();
    tucker_core1<double> result_tucker;
    result_tucker = hooi(a,R,R,R);
    t1=gettime();
    cout << "time:" <<t1-t0 <<endl;

//    t0=gettime();
//    tsvd_core<double> result_tucker;
//    result_tucker = tsvd(a);
//    t1=gettime();
//    cout << "time:" <<t1-t0 <<endl;


    return 0;
}

//    cube * m;
//    m = &a;
//    b = *m;
//    mat * ccc;
//    mat cb;
//    ccc = &a.slice(0);
//    cb = *ccc;
//    cout << cb << endl;
//    cout << b << endl;
//    *m = a;
//    cout << m << endl;
//    cout << sizeof(m) << endl;
//    cout << m[0] << endl;

//    cout << a << endl;

//    vec x = vectorise(a);
////    cout << reshape(x,n1,n2*n3);
//
//    double *m[n1];
//
//    t0=gettime();
//    cube b(a);
//        a.reshape(n1,n2*n3,1);
//        a1 = a.slice(0);
//    b=a;
//    t1=gettime();
//    cout << "time:" <<t1-t0 <<endl;


    //    I1 = k*n1; J1 = k*n1+n1-1;
//    a1.cols(I1,J1) = a.slice(k);
//    cout << a1.cols(I1,J1)<< endl;
//        cout << a1 << endl;



//    t0=gettime();
//    int I1,J1;
//    mat a1(n1,n2*n3);
//    double *m[n1];
//    for(int k=0;k<n3;k++){
//        for(int i=0;i<n1;i++){
//            m[i] = a.slice(k).colptr(i);
//        }
//    }
//
//    //    I1 = k*n1; J1 = k*n1+n1-1;
////    a1.cols(I1,J1) = a.slice(k);
////    cout << a1.cols(I1,J1)<< endl;
////        cout << a1 << endl;
//    t1=gettime();
//    cout << "time:" <<t1-t0 <<endl;

//    cout << a1.colptr(1) << endl;
//    cout << a.slice(1).colptr(1) << endl;


////mode-1
//    t0=gettime();
//        mat a1(n1,n2*n3);
//        mat *a1_tmp;
//        a.reshape(n1,n2*n3,1);
//        a1_tmp = &a.slice(0);
////        a1 = *a1_tmp;
//    t1=gettime();
//    cout << "time:" <<t1-t0 <<endl;
//    mat b = randu(I,I);
//
//    t0=gettime();
//    for (int i=0;i<1000;i++)
//        b * (*a1_tmp);
//    t1=gettime();
//    cout << "time:" <<t1-t0 <<endl;
//
//    t0=gettime();
//    for (int i=0;i<1000;i++)
//        b * (a.slice(0));
//    t1=gettime();
//    cout << "time:" <<t1-t0 <<endl;
////    cout << b * (*a1_tmp) << endl;


//    t0=gettime();
//    mat a1(n1,n2*n3);
//    a.reshape(n1,n2*n3,1);
//    a1 = a.slice(0);
//    t1=gettime();
//    cout << "time:" <<t1-t0 <<endl;

//    cout << a1 << endl;
//    a.reshape(n1,n2,n3);
//    cout << a << endl;
//    cout << a1 << endl;

//    a1.set_size(n2,n1*n3);
//    //mode-2
//    t0=gettime();
//    mat tmp;
//    int I1,J1;
//    for(int k=0; k<n3;k++){
//        I1 = k*n1; J1 = k*n1+n1-1;
//        tmp = a1.cols(I1,J1);
//        a1.cols(I1,J1) = tmp.t();
//    }
//    t1=gettime();
//    cout << "time:" <<t1-t0 <<endl;
//
//    t0=gettime();
//    mat x1(1,n2*n3);
//    for(int i=0; i<n1; i++){
//        x1 = a1.row(i);
//        x1.reshape(n2,n3);
//    }
//    t1=gettime();
//    cout << "time:" <<t1-t0 <<endl;

//mode-3
//    t0=gettime();
//    mat x2 =  vectorise(a1,1);
//
////    cout << x1;
//    t1=gettime();
//    x2 = reshape(x2,n3,n1*n2);

//    mat test(n1,n2);
//    test.memptr() = a.slice(0).memptr();
//
//    cout << a.slice(0)(0,99);
//    cout << "pointer" << a.slice(0).memptr() << endl;


//    a.reshape(n2,1,n1*n3);
//    mat a2 = a.col(0);
//    cout << a2 << endl;
//
//    a.reshape(n3,n1*n2,1);
//    mat a3 = a.slice(0);
//    cout << a3 << endl;



//arma_rng::set_seed(1);

