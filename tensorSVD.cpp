#include <iostream>
#include <fstream>
#include<iomanip>

#include <fftw3.h>
#include <cstdlib>
#include <stdio.h>
#include <string>
#include <vector>

#include <time.h>
#include <sys/time.h>

#include <Eigen/SVD>
#include <Eigen/Dense>
//using Eigen::MatrixXf;
using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

//timeval结构定义为:
double gettime(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*1000+tv.tv_usec/1000.0;
};

double *** al(int &n1,int &n2,int &n3){
    double ***p;
    p=new double**[n1];
    for (int i=0;i<n1;i++){
        p[i] = new double *[n2];
        for (int j=0;j<n2;j++) {
            p[i][j] = new double[n3];
        }
    }
    return p;
}

int main()
{
    double start,end,timeuse;
    int count = 0;
    FILE* fp;
    char str[100];

    int n1=144,n2=256,n3=40; // 2000-2000-200

    string ***p0 =new string**[n1];
    for (int i=0; i<n1;i++) {
        p0[i] = new string *[n2];
        for (int j=0;j<n2;j++){
            p0[i][j]=new string [n3];
        }
    }

    fp = fopen("/home/jcfei/Documents/MATLAB/data/VIDEO.txt", "r");

    while (fscanf(fp, "%s", str) != EOF)
    {
        int NUM=count;
        int k=NUM%(n1*n2);
        k=(NUM-k)/(n1*n2);
        int j=(NUM-k*n1*n2)%n2;
        int i=(NUM-k*n1*n2)/n2;
        p0[i][j][k]=str;
        count++;
        if(count==n1*n2*n3){break;}
    }

    fclose(fp);

    double ***v =new double**[n1];
    for (int i=0; i<n1;i++) {
        v[i] = new double *[n2];
        for (int j=0;j<n2;j++){
            v[i][j]=new double [n3];
        }
    }

    string tmp;
    for(int i=0; i<n1; i++) {
        for (int j = 0; j < n2; j++) {
            for(int k=0; k<n3; k++){
                tmp=p0[i][j][k];
                v[i][j][k]=atof(tmp.c_str());
            }
        }
    }

    double ***v_t = al(n1,n2,n3);
    double ***v_t1 = al(n1,n2,n3);

    start=gettime();

    fftw_complex in[n3], out[n3]; //fftw_alloc_real()
    fftw_plan p;
    p=fftw_plan_dft_1d(n3,in,out,FFTW_FORWARD,FFTW_MEASURE);
    for (int i=0;i<n1;i++){
        for (int j=0;j<n2;j++){
            for (int k=0;k<n3;k++){
                in[k][0]=v[i][j][k];
                in[k][1]=0;
            }
            fftw_execute(p);
            for (int k=0;k<n3;k++){
                v_t[i][j][k]=out[k][0];
                v_t1[i][j][k]=out[k][1];
            }
        }
    }
    fftw_destroy_plan(p);

    double ***uf = al(n1,n1,n3);
    double ***uf1 = al(n1,n1,n3);
    double ***theta = al(n1,n2,n3);
    double ***vf = al(n2,n2,n3);
    double ***vf1 = al(n2,n2,n3);

    double ***U = al(n1,n1,n3);
    double ***V = al(n2,n2,n3);
    double ***Theta = al(n1,n2,n3);


    MatrixXcf TMP=MatrixXcf::Random(n1,n2);
    MatrixXcf TMPU=MatrixXcf::Random(n1,n1);
    MatrixXcf TMPV=MatrixXcf::Random(n2,n2);
    MatrixXcf TMPT=MatrixXcf::Random(n1,n2);

    for (int k=0;k<n3;k++){
        for (int i=0;i<n1;i++){
            for (int j=0;j<n2;j++){
                TMP(i,j).real(v_t[i][j][k]);
                TMP(i,j).imag(v_t1[i][j][k]);
            }
        }
        BDCSVD<MatrixXcf> svd(TMP, ComputeFullU | ComputeFullV );
        TMPU=svd.matrixU();
        TMPV=svd.matrixV();
        TMPT=TMPU.conjugate().transpose()*TMP*TMPV;

        for (int i=0;i<n1;i++){
            for (int j=0;j<n1;j++){
                uf[i][j][k]=TMPU(i,j).real();
                uf1[i][j][k]=TMPU(i,j).imag();
//                vf[i][j][k]=TMPV(i,j).real();
//                vf1[i][j][k]=TMPV(i,j).imag();
//                theta[i][j][k]=TMPT(i,j).real();
            }
        }
        for (int i=0;i<n2;i++){
            for (int j=0;j<n2;j++){
                vf[i][j][k]=TMPV(i,j).real();
                vf1[i][j][k]=TMPV(i,j).imag();
            }
        }
        for (int i=0;i<n1;i++){
            for (int j=0;j<n2;j++){
                theta[i][j][k]=TMPT(i,j).real();
            }
        }
    }

    fftw_plan p1;
    p1=fftw_plan_dft_1d(n3,in,out,FFTW_BACKWARD,FFTW_MEASURE);
    for (int i=0;i<n1;i++) {
        for (int j = 0; j < n1; j++) {
            for (int k = 0; k < n3; k++) {
                in[k][0] = uf[i][j][k];
                in[k][1] = uf1[i][j][k];
            }
            fftw_execute(p1);
            for (int k = 0; k < n3; k++) {
                U[i][j][k] = out[k][0];
            }
        }
    }
    for (int i=0;i<n2;i++) {
        for (int j = 0; j < n2; j++) {
            for (int k = 0; k < n3; k++) {
                in[k][0] = vf[i][j][k];
                in[k][1] = vf1[i][j][k];
            }
            fftw_execute(p1);
            for (int k = 0; k < n3; k++) {
                V[i][j][k] = out[k][0];
            }
        }
    }
    for (int i=0;i<n1;i++) {
        for (int j = 0; j < n2; j++) {
            for (int k=0;k<n3;k++){
                in[k][0]=theta[i][j][k];
                in[k][1]=0;
            }
            fftw_execute(p1);
            for (int k=0;k<n3;k++){
                Theta[i][j][k]=out[k][0];
            }
        }
    }
    fftw_destroy_plan(p1);

    //fft transform result write to txt
//    ofstream a;
//    a.open("/home/jcfei/Desktop/TensorC++/txtToarray/a3456.txt");
//    for (int k=0; k<n3;k++){
//        for (int i=0; i<n1;i++){
//            for(int j=0; j<n2;j++){
//                a<<setiosflags(ios::right)<<setw(15)<<uf[i][j][k]<<"  ";
//                a<<setiosflags(ios::right)<<setw(15)<<uf1[i][j][k]<<"  ";
//
//            }
//            a<<endl;
//        }
//    }

    end=gettime();
    timeuse=end-start;
    cout<<"Time: "<<timeuse<<endl;

    cout<<"Time: "<<endl;

    return 0;

}