#include <iostream>
#include <fstream>
#include<iomanip>

#include <cstdlib>
#include <stdio.h>
#include <string>
#include <vector>

#include <time.h>
#include <sys/time.h>

#include "mkl.h"
#include <omp.h>
//#include "mkl_service.h"
#include <fftw3.h>
#include <armadillo>
using namespace std;
using namespace arma;

//timeval结构定义为:
double gettime(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*1000+tv.tv_usec/1000.0;
};

float *** al(int &n1,int &n2,int &n3){
    float ***p;
    p=new float**[n1];
    for (int i=0;i<n1;i++){
        p[i] = new float *[n2];
        for (int j=0;j<n2;j++) {
            p[i][j] = new float[n3];
        }
    }
    return p;
}

//template <typename T>
void del(float *** p, int &n1,int &n2){
    for(int i=0; i<n1; i++) {
        for (int j = 0; j < n2; j++) {
            delete [] p[i][j];
        }
        delete [] p[i];
    }
    delete  [] p;
    p=NULL;
}

float *** loadfile(int &n1, int &n2, int &n3, char *path){

    int count = 0;
    FILE* fp;
    char str[100];

    float ***v = al(n1,n2,n3);

    fp = fopen(path,"r");

    string tmp;
    while (fscanf(fp, "%s", str) != EOF)
    {
        int NUM=count;
        int k=NUM%(n1*n2);
        k=(NUM-k)/(n1*n2);
        int j=(NUM-k*n1*n2)%n2;
        int i=(NUM-k*n1*n2)/n2;
        tmp=str;
        v[i][j][k]=(float)atof(tmp.c_str());
        count++;
        if(count==n1*n2*n3){break;}
    }

    fclose(fp);

//释放空间 内存泄漏。。。还会影响运行时间
    return v;
}

void display(float *** v, int &n1, int &n2, int &n3){

    cout << "Tensor: " << endl;

    for (int k = 0; k < n3; k++) {
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                cout << v[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

//inline arma::svd(cx_mat,vec,cx_mat,cx_mat); 如何声明？

int main()
{
    double t0, t4;
    int turn=1500;
    int n1 = turn, n2 = turn, n3 = 256; // 2000-2000-200
    int N0 = floor(n3/2.0) + 1; //    cout << "N0 " << N0 <<endl;

//    float ***M = loadfile(n1, n2, n3, path);
    float ***M = al(n1, n2, n3);

    for (int k = 0; k < n3; k++) {
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                M[i][j][k] = (float)randu();
            }
        }
    }

    cout << "loadfile" << endl;

    float ***v_t = al(n1, n2, N0);
    float ***v_t1 = al(n1, n2, N0);

    t0 = gettime();
    fftwf_complex out[N0]; //fftw_alloc_real()
    float *in = fftwf_alloc_real(n3);

    fftwf_plan p_fft;
    p_fft = fftwf_plan_dft_r2c_1d(n3, in, out, FFTW_ESTIMATE);
//    p=fftw_plan_dft_1d(n3,in,out,FFTW_FORWARD,FFTW_MEASURE);

//#pragma omp parallel for num_threads(8)
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            in = M[i][j];   //第三维赋值
            fftwf_execute_dft_r2c(p_fft, in, out);
            for (int k = 0; k < N0; k++) {
                v_t[i][j][k] = out[k][0];
                v_t1[i][j][k] = out[k][1];
            }               //局部变量归零了？？
        }
    }
//    del(M,n1,n2);

//    t1=gettime();
//    cout<<"fft time: "<<t1-t0<<endl;

    float ***uf = al(n1, n1, N0);
    float ***uf1 = al(n1, n1, N0);
    float ***theta = al(n1, n2, N0);
    float ***vf = al(n2, n2, N0);
    float ***vf1 = al(n2, n2, N0); //合并成一个矩阵

    cx_fmat TMP = zeros<cx_fmat>(n1, n2);
    cx_fmat TMPU = zeros<cx_fmat>(n1, n1);
    cx_fmat TMPV = zeros<cx_fmat>(n2, n2);
    fcolvec TMPT;

//    t2=gettime();
//    cout<<"alloc space: "<<t2-t1<<endl;

//#pragma omp parallel for num_threads(8) //如何写多线程的svd 会不会覆盖同一个svd，的结果？？
    for (int k = 0; k < N0; k++) {
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                TMP(i, j).real(v_t[i][j][k]);
                TMP(i, j).imag(v_t1[i][j][k]);
            }
        }

        svd(TMPU, TMPT, TMPV, TMP, "dc");
//        svd(TMPU,TMPT,TMPV,TMP,"std");

        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n1; j++) {
                uf[i][j][k] = TMPU(i, j).real();
                uf1[i][j][k] = TMPU(i, j).imag();
            }
        }
        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < n2; j++) {
                vf[i][j][k] = TMPV(i, j).real();
                vf1[i][j][k] = TMPV(i, j).imag();
            }
        }
        if (n1 <= n2) {
            for (int i = 0; i < n1; i++) {
                theta[i][i][k] = TMPT(i);
            }
        } else {
            for (int i = 0; i < n2; i++) {
                theta[i][i][k] = TMPT(i);
            }
        }
    }

//    del(M,n1,n2);  //这个删除的位置怎么那么奇怪。。。为什么不能删除？ 是不是之后用不到自动删除了？？
    del(v_t, n1, n2);
    del(v_t1, n1, n2);

    fftwf_complex out1[N0]; //fftw_alloc_real()
    float *in1 = fftwf_alloc_real(n3);
    p_fft = fftwf_plan_dft_c2r_1d(n3, out1, in1, FFTW_ESTIMATE);

//    #pragma omp parallel for num_threads(8)
//    #pragma omp parallel for num_threads(2)

    float ***U = al(n1, n1, n3);

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n1; j++) {
            for (int k = 0; k < N0; k++) {
                out1[k][0] = uf[i][j][k];
                out1[k][1] = uf1[i][j][k];
            }
            fftwf_execute_dft_c2r(p_fft, out1, in1); //为什么执行不了？？和申请Theta有什么关系？？

            for (int k = 0; k < n3; k++) {
                U[i][j][k] = 1.0 / n3 * in1[k];
            }
//            U[i][j] = in;  为何不能直接传递指针呢？？
        }
    }

    del(uf, n1, n1);
    del(uf1, n1, n1);

    float ***V = al(n2, n2, n3);
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n2; j++) {
            for (int k = 0; k < N0; k++) {
                out1[k][0] = vf[i][j][k];
                out1[k][1] = vf1[i][j][k];
            }
            fftwf_execute_dft_c2r(p_fft, out1, in1);
            for (int k = 0; k < n3; k++) {
                V[i][j][k] = 1.0 / n3 * in1[k];
            }
        }
    }
    del(vf, n2, n2);
    del(vf1, n2, n2);

    float ***Theta = al(n1, n2, n3);
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            for (int k = 0; k < N0; k++) {
                out1[k][0] = theta[i][j][k];
                out1[k][1] = 0;
            }
            fftwf_execute_dft_c2r(p_fft, out1, in1);
            for (int k = 0; k < n3; k++) {
                Theta[i][j][k] = 1.0 / n3 * in1[k];
            }
        }
    }
    del(theta, n1, n2);

    fftwf_destroy_plan(p_fft);

    t4 = gettime();
//    cout<<"ifft time: "<<t4-t3<<endl;
    cout << "Total time: " << t4 - t0 << endl;


    //fft transform result write to txt
//    ofstream a;
//    a.open("/home/jcfei/Desktop/TensorC++/txtToarray/Theta-video.txt");
//    for (int k=0; k<n3;k++){
//        for (int i=0; i<n1;i++){
//            for(int j=0; j<n2;j++){
//                a<<setiosflags(ios::right)<<setw(15)<<Theta[i][j][k]<<"  ";
//            }
//            a<<endl;
//        }
//    }
//    a.close();
//
//    ofstream b;
//    b.open("/home/jcfei/Desktop/TensorC++/txtToarray/U-video.txt");
//    for (int k=0; k<n3;k++){
//        for (int i=0; i<n1;i++){
//            for(int j=0; j<n1;j++){
//                b<<setiosflags(ios::right)<<setw(10)<<U[i][j][k]<<"  ";
//            }
//            b<<endl;
//        }
//    }
//    b.close();
//
//    ofstream c;
//    c.open("/home/jcfei/Desktop/TensorC++/txtToarray/V-video.txt");
//    for (int k=0; k<n3;k++){
//        for (int i=0; i<n2;i++){
//            for(int j=0; j<n2;j++){
//                c<<setiosflags(ios::right)<<setw(10)<<V[i][j][k]<<"  ";
//            }
//            c<<endl;
//        }
//    }
//    c.close();
    return 0;
}