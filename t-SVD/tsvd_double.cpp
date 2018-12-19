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

//template <typename T>
void del(double *** p, int &n1,int &n2){
    for(int i=0; i<n1; i++) {
        for (int j = 0; j < n2; j++) {
            delete [] p[i][j];
        }
        delete [] p[i];
    }
    delete  [] p;
    p=NULL;
}

double *** loadfile(int &n1, int &n2, int &n3, char *path){

    int count = 0;
    FILE* fp;
    char str[100];

    double ***v = al(n1,n2,n3);

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
        v[i][j][k]=(double)atof(tmp.c_str());
        count++;
        if(count==n1*n2*n3){break;}
    }

    fclose(fp);

    return v;
}

void display(double *** v, int &n1, int &n2, int &n3){

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

int main()
{
    double t0, t1, t2, t3, t4;
    int turn=2000;
    int n1 = turn, n2 = turn, n3 = 110;
    int N0 = floor(n3/2.0) + 1; 

    char path[1000] = "/home/jcfei/Documents/MATLAB/data/a2000.txt";

    double ***M = loadfile(n1, n2, n3, path);
    cout << "loadfile" << endl;

    double ***v_t = al(n1, n2, N0);
    double ***v_t1 = al(n1, n2, N0);

    t0 = gettime();
    fftw_complex out[N0]; //fftw_alloc_real()
    double *in = fftw_alloc_real(n3);

    fftw_plan p_fft;
    p_fft = fftw_plan_dft_r2c_1d(n3, in, out, FFTW_ESTIMATE);
//    p=fftw_plan_dft_1d(n3,in,out,FFTW_FORWARD,FFTW_MEASURE);

//#pragma omp parallel for num_threads(8)
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            in = M[i][j];   
            fftw_execute_dft_r2c(p_fft, in, out);
            for (int k = 0; k < N0; k++) {
                v_t[i][j][k] = out[k][0];
                v_t1[i][j][k] = out[k][1];
            }             
        }
    }
    del(M,n1,n2);
//    t1=gettime();
//    cout<<"fft time: "<<t1-t0<<endl;

    double ***uf = al(n1, n1, N0);
    double ***uf1 = al(n1, n1, N0);
    double ***theta = al(n1, n2, N0);
    double ***vf = al(n2, n2, N0);
    double ***vf1 = al(n2, n2, N0);

    cx_mat TMP = zeros<cx_mat>(n1, n2);
    cx_mat TMPU = zeros<cx_mat>(n1, n1);
    cx_mat TMPV = zeros<cx_mat>(n2, n2);
    colvec TMPT;

//    t2=gettime();
//    cout<<"alloc space: "<<t2-t1<<endl;

//#pragma omp parallel for num_threads(8) 
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

    del(v_t, n1, n2);
    del(v_t1, n1, n2);

    fftw_complex out1[N0]; //fftw_alloc_real()
    double *in1 = fftw_alloc_real(n3);
    p_fft = fftw_plan_dft_c2r_1d(n3, out1, in1, FFTW_ESTIMATE);

//    #pragma omp parallel for num_threads(8)
//    #pragma omp parallel for num_threads(2)

    double ***U = al(n1, n1, n3);

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n1; j++) {
            for (int k = 0; k < N0; k++) {
                out1[k][0] = uf[i][j][k];
                out1[k][1] = uf1[i][j][k];
            }
            fftw_execute_dft_c2r(p_fft, out1, in1);

            for (int k = 0; k < n3; k++) {
                U[i][j][k] = 1.0 / n3 * in1[k];
            }
        }
    }

    del(uf, n1, n1);
    del(uf1, n1, n1);

    double ***V = al(n2, n2, n3);
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n2; j++) {
            for (int k = 0; k < N0; k++) {
                out1[k][0] = vf[i][j][k];
                out1[k][1] = vf1[i][j][k];
            }
            fftw_execute_dft_c2r(p_fft, out1, in1);
            for (int k = 0; k < n3; k++) {
                V[i][j][k] = 1.0 / n3 * in1[k];
            }
        }
    }
    del(vf, n2, n2);
    del(vf1, n2, n2);

    double ***Theta = al(n1, n2, n3);
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            for (int k = 0; k < N0; k++) {
                out1[k][0] = theta[i][j][k];
                out1[k][1] = 0;
            }
            fftw_execute_dft_c2r(p_fft, out1, in1);
            for (int k = 0; k < n3; k++) {
                Theta[i][j][k] = 1.0 / n3 * in1[k];
            }
        }
    }
    del(theta, n1, n2);

    fftw_destroy_plan(p_fft);

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
