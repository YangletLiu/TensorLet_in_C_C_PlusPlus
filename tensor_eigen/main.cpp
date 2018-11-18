#include <iostream>
#define EIGEN_USE_MKL_ALL
#define EIGEN_USE_BLAS

#include <mkl.h>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <tensor.h>

using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;
using Eigen::Tensor;

template <class T>
T firs(Tensor<T,3> a){
    T b;
    b=a(1,1,1);
    int c = sizeof(T);
//    cout << sizeof(double) << endl;
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
    Map<MatrixXd>  eigMat1(a.data(), n1,n2*n3);
    Map<MatrixXd,0,InnerStride<1>>  eigMat2(a.data(), n1,n2*n3);

    Map<MatrixXd,0,OuterStride<>>  eigMat3(a.data(), n1,n2*n3,OuterStride<>(2));
    cout << "eigMat3: \n" << eigMat3 << endl;
    MatrixXd ccc =eigMat1;

/*Stride 的第一个参数为OuterStride，对于按列优先存储的矩阵来说，就是 列与列之间指针的差值；
第二个参数为InnerStride，即两个相邻元素 指针之间的差值 */

// 矩阵乘法
    MatrixXd cc(n2,n1),d;
    d = cc* eigMat1;
    cout << d << endl;

//    Map<Matrix<double,n1,n2*n3,RowMajor>>(a.data());

    cout << "eigMat1: \n" << eigMat1 << endl;
    cout << "eigMat2: \n" << eigMat2 << endl;

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

    double T =1;
    T = firs(a);

    std::cout << "Hello, World!" << std::endl;

    return 0;
}

//可以根据输入确定大小
//int n1; cin >> n1;
//Eigen::Tensor<double,3> b(n1,3,3);
//b.resize(2,2,2);
//cout << b << endl << endl;
//Eigen::TensorMap<double> c(b.data(),2,2);

//Map类作用
//std::vector<int> v(27);
//std::iota(v.begin(),v.end(),1);
//std::iota :用顺序递增的值赋值指定范围内的元素;从1开始，1～27
//Eigen::TensorMap<Eigen::Tensor<int,3>> mapped(v.data(), 3, 3, 3 );
//cout << mapped << endl;
//
//Eigen::array<long,3> startIdx = {0,0,0};       //Start at top left corner
//Eigen::array<long,3> extentt = {2,2,2};       // take 2 x 2 x 2 elements
//Eigen::Tensor<int,3> sliced = mapped.slice(startIdx,extentt);
//std::cout << sliced << std::endl;

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

//隔几列取元素
//MatrixXf M1 = MatrixXf::Random(3,8);
//
//cout << "Column major input:" << endl << M1 << "\n";
//cout << "M1.outerStride() = " << M1.outerStride() << endl;
//cout << "M1.innerStride() = " << M1.innerStride() << endl;
//
//Map<MatrixXf,0,OuterStride<> > M2(
//        M1.data(), M1.rows(), (M1.cols()+2)/3, OuterStride<>(M1.outerStride()*3));
//cout << "1 column over 3:" << endl << M2 << "\n";

//数组转化为Eigen::Matrix
//int array[9];
//cout << "colMajor matrix = \n" << Map<Matrix3i>(array) << endl;                      // map a contiguous array as a column-major matrix
//cout << "rowMajor matrix = \n" << Map<Matrix<int, 3, 3, RowMajor>>(array) << endl;   // map a contiguous array as a row-major matrix
//
//Map<MatrixXi>  eigMat11(array, 3, 3);                     // eigMat1和array指向的是同一个内存空间，是绑定在一起的
//MatrixXi       eigMat21 = Map<MatrixXi>(array, 3, 3);    //  eigMat1和array指向不同的内存空间，互不影响

//Eigen::Matrix转化为数组
//    Matrix3d eigMat;
//    double* eigMatptr = eigMat.data();
//    double* eigMatptrnew = new double[eigMat.size()];
//    Map<MatrixXd>(eigMatptrnew, eigMat.rows(), eigMat.cols()) = eigMat;
//    cout << "eigMat: \n" << eigMatptrnew << endl;