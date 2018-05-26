#include <iostream>
#include <vector>

#include <Eigen/SVD>
#include <Eigen/Dense>

//using Eigen::MatrixXf;
using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

int main()
{
//-------------------------------svd测试    eigen
    int m=3;int n=3;

    float **p=new float*[m];
    for(int i=0; i<m; i++)
        p[i]=new float[n];
    cout<<p[0][0]+2<<endl;

    Matrix3f A;
    A(0,0)=1,A(0,1)=p[0][0]+10,A(0,2)=1;
    A(1,0)=2,A(1,1)=1,A(1,2)=1;
    A(2,0)=3,A(2,1)=0,A(2,2)=0;
    cout<<A.row(0)<<endl;
    cout<<A.col(0)<<endl;
    JacobiSVD<Eigen::MatrixXf> svd(A, ComputeThinU | ComputeThinV );
    clock_t time_stt = clock();
    Matrix3f V = svd.matrixV(), U = svd.matrixU();
    Matrix3f  S = U.inverse() * A * V.transpose().inverse(); // S = U^-1 * A * VT * -1
    std::cout<<"A :\n"<<A<<std::endl;
    std::cout<<"U :\n"<<U<<std::endl;
    std::cout<<"S :\n"<<S<<std::endl;
    std::cout<<"V :\n"<<V<<std::endl;
    std::cout<<"U * S * VT :\n"<<U * S * V.transpose()<<std::endl;
    //-------------------------------svd测试    eigen
    return 0;
}
