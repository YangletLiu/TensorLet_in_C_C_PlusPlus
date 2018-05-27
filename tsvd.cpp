#include <complex>
#include <fftw3.h>
#include <math.h>
#include <iostream>

#include <iostream>
#include <vector>
#include <Eigen/SVD>
#include <Eigen/Dense>

#include "time.h"

//using Eigen::MatrixXf;
using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;


#define N 3
using namespace std;

int main(int argc, char * argv[]){
    MatrixXf m= MatrixXf::Zero(3,3);
    MatrixXcf fm=MatrixXcf::Random(3,3);
    m<<1,2,3,4,5,6,7,8,9;
    cout<<m<<endl;

    clock_t start,end,time;
    fftw_complex in[N], out[N];
    fftw_plan p;
    p=fftw_plan_dft_1d(N,in,out,FFTW_FORWARD,FFTW_MEASURE);
//实部和虚部指定
    start=clock();
    for (int j=0;j<N;j++){
        for(int i=0;i <N;i ++) {
            in[i][0]=m(i,j);
            in[i][1]=0.0;
        }

        for(int i=0;i <N;i ++){
            cout<<in[i][0]<<" "<<in[i][1]<<endl;
        }
//执行变换？
        fftw_execute(p);


        for(int i=0;i <N;i ++){
            cout<<out[i][0]<<" "<<out[i][1]<<endl;
            fm(i,j).real(out[i][0]);
            fm(i,j).imag(out[i][1]);
        }
    }
    cout<<fm<<endl;
    end=clock();

    cout<<double(end-start)/1.0<<endl;
//    cout<<CLOCKS_PER_SEC<<endl;
//    time=double((end-start))/CLOCKS_PER_SEC*1000;
    cout<<"time:"<< time<<endl;

//    cout<<"Hello World"<<endl;
//    complex<double> temp = 0.0;
//
//    for(int k =0; k < N; k ++){
//        double pi = 4*atan(1.0);
//        temp += exp(complex<double>(0.0,-2.0*pi*3*k/N))*complex<double>(in[k][0],in[k][1]);
//    }
//    cout<<"out[3] is "<<temp<<endl;

    fftw_complex out1[N];
//逆变换
    fftw_plan p1;

    p1=fftw_plan_dft_1d(N,out1,in,FFTW_BACKWARD,FFTW_MEASURE);

    for(int i=0;i <N;i ++){
        out1[i][0]=out[i][0];
        out1[i][1]=out[i][1];
    }
    start=clock();
    for(int i=0;i <N;i ++){
        out1[i][0]=i;
        out1[i][1]=0;
    }

    fftw_execute(p1);
    end=clock();

    for(int i=0;i <N;i ++){
        cout<<in[i][0]<<" "<<in[i][1]<<endl;
    }
    cout<<end-start<<endl;

//    MatrixXcf A=MatrixXcf::Random(3,3);
    JacobiSVD<Eigen::MatrixXcf> svd(fm, ComputeThinU | ComputeThinV );
    clock_t time_stt = clock();
    MatrixXcf V = svd.matrixV(), U = svd.matrixU();
    MatrixXcf  S = U.inverse() * fm * V.transpose().inverse(); // S = U^-1 * A * VT * -1
    std::cout<<"fm :\n"<<fm<<std::endl;
    std::cout<<"U :\n"<<U<<std::endl;
    std::cout<<"S :\n"<<S<<std::endl;
    std::cout<<"V :\n"<<V<<std::endl;
    std::cout<<"U * S * VT :\n"<<U * S * V.transpose()<<std::endl;

    fftw_destroy_plan(p);
    fftw_destroy_plan(p1);

    return 0;
}
