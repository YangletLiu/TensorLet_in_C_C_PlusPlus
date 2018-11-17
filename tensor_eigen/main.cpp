#include <iostream>

#include <time.h>
#include <sys/time.h>

#include <mkl.h>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;
using Eigen::Tensor;

double gettime(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec+tv.tv_usec/1000000.0;
};

template <class T>
T firs(Tensor<T,3> a){
    T b;
    b=a(1,1,1);
    int c = sizeof(T);
    cout << "size " << c << endl;
}

int main() {
    int n1,n2,n3;
    n1=n2=n3=3;
    Eigen::Tensor<double,3> a(n1,n2,n3);
    Tensor<double,3> * pointer = &a;
    a.setRandom();

    a.data();
    double * c = a.data();
    cout << c[26] << endl;

    cout << "data " << *(a.data()+26) << endl;
    cout << pointer << endl;
    cout << *pointer << endl;
    Map<Tensor<double,3>>
//    MatrixXcd d=c*c;

    cout << sizeof(double) << endl;

    double t1,t2;
    t1=gettime();
    Eigen::array<long,3> startIdx = {0,0,0};
    Eigen::array<long, 3> endIdx_1 = {n1, n2, n3};  //mode-1

    a.slice(startIdx, endIdx_1);

    t2=gettime();
    cout << "time:" << t2-t1 << endl;

    Eigen::array<long,3> endIdx_2 = {1,n2,n3};  //mode-2
    Eigen::array<long,3> endIdx_3 = {n1,1,n3};  //mode-3

    Eigen::Tensor<double,3> slice_2 = a.slice(startIdx,endIdx_2);
    Eigen::Tensor<double,3> slice_3 = a.slice(startIdx,endIdx_3);


//    cout << sliced << endl;
//    cout << a << endl;


    double T =1;
    T = firs(a);

    std::cout << "Hello, World!" << std::endl;
    return 0;
}

//int n1;
//cin >> n1;
//Eigen::Tensor<double,3> b(n1,3,3);
//b.resize(2,2,2);
//cout << b << endl << endl;
//
//Eigen::TensorMap<double> c(b.data(),2,2);

//std::vector<int> v(27);
//std::iota(v.begin(),v.end(),1);
//Eigen::TensorMap<Eigen::Tensor<int,3>> mapped(v.data(), 3, 3, 3 );
//cout << mapped << endl;
//
//Eigen::array<long,3> startIdx = {0,0,0};       //Start at top left corner
//Eigen::array<long,3> extentt = {2,2,2};       // take 2 x 2 x 2 elements
//
//Eigen::Tensor<int,3> sliced = mapped.slice(startIdx,extentt);
//
//std::cout << sliced << std::endl;

//b slice
//Eigen::Tensor<int, 2> b(4, 3);
//b.setValues({{0, 100, 200}, {300, 400, 500},
//{600, 700, 800}, {900, 1000, 1100}});
//
//Eigen::array<int, 2> offsets = {1, 0};
//Eigen::array<int, 2> extents = {2, 2};
//Eigen::Tensor<int, 2> slice = b.slice(offsets, extents);
//
//cout << "b" << endl << b << endl;
//cout << "slice" << endl << slice << endl;
//
//MatrixXf M1 = MatrixXf::Random(3,8);
//
//cout << "Column major input:" << endl << M1 << "\n";
//
//cout << "M1.outerStride() = " << M1.outerStride() << endl;
//cout << "M1.innerStride() = " << M1.innerStride() << endl;
//
//Map<MatrixXf,0,OuterStride<> > M2(
//        M1.data(), M1.rows(), (M1.cols()+2)/3, OuterStride<>(M1.outerStride()*3));
//
//cout << "1 column over 3:" << endl << M2 << "\n";
