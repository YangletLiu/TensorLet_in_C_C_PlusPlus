#include "tensor.h"

#include "train.h"
using namespace yph;
double gettime()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000 + tv.tv_usec / 1000.0) / 1000.0; //time:s
};

int main()
{
    int n1, n2, n3;
    double t0, t1;
    int I = 3;
    int R = 0.2 * I;
    //    bool cc= true;

    n1 = I;
    n2 = I;
    n3 = I;

    cube a = randu<cube>(n1, n2, n3);
    a.print("at the very beginning tha test tensor is");
    Mat<double> test = ten2mat(a, n1 * n2, n3);
    test.print("tensor to mat");

    Mat<double> U,V;
    Col<double> s;
    svd(U,s,V,test);
    U.cols(1,3).print("U");
    s.print("s");
    V.print("V");
    test = U.cols(1, 3) * diagmat(s) * V.t();
    test.print("svd");

    mat2ten(test,a,n1,n2,n3);
    a.print("reshape from a matrix");

    TensorTrain<double> tensorTrain(a, 0.01);
    cout << "the tt-rank is " << tensorTrain.getTTRank(0)
         << " " << tensorTrain.getTTRank(1) << " " << tensorTrain.getTTRank(2)
         << " " << tensorTrain.getTTRank(3) << endl;
    cout << "the cores are"<<endl;;
    tensorTrain.getCores(0).print("core 1");
    tensorTrain.getCores(1).print("core 2");
    tensorTrain.getCores(2).print("core 3");

    return 0;
}
