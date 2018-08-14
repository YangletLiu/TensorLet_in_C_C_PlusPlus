#include <complex>
#include <fftw3.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <mkl.h>
#include <omp.h>

using namespace std;

double gettime(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*1000+tv.tv_usec/1000.0;
};

inline double *** al(int &n1,int &n2,int &n3){
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

double *** loadfile(int &n1, int &n2, int &n3, char *path){

    int count = 0;
    FILE* fp;
    char str[100];

    string ***p0 =new string**[n1];
    for (int i=0; i<n1;i++) {
        p0[i] = new string *[n2];
        for (int j=0;j<n2;j++){
            p0[i][j]=new string [n3];
        }
    }

    fp = fopen(path,"r");

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

    double ***v = al(n1,n2,n3);

    string tmp;
    for(int i=0; i<n1; i++) {
        for (int j = 0; j < n2; j++) {
            for(int k=0; k<n3; k++){
                tmp=p0[i][j][k];
                v[i][j][k]=(double) atof(tmp.c_str());
            }
        }
    }

    for(int i=0; i<n1; i++) {
        for (int j = 0; j < n2; j++) {
            delete [] p0[i][j];
        }
        delete [] p0[i];
    }
    delete p0;

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

int main(int argc, char * argv[]){

    double t0,t1,t2;

    int n1=144,n2=256,n3=1000; // 2000-2000-200

    char path[1000] = "/home/jcfei/Documents/MATLAB/data/a1442561000.txt";

    double ***v = loadfile(n1,n2,n3,path);

//    display(v, n1, n2, n3);

    int N0=floor(n3/2.0)+1;
    cout << "N0 " << N0 <<endl;

    fftw_complex out[N0]; //fftw_alloc_real()
//    fftw_complex v_f[n1][n2][N0];

    double *in = fftw_alloc_real(n3);

    t0 = gettime();

    fftw_plan p_fft;

    p_fft=fftw_plan_dft_r2c_1d(n3, in, out, FFTW_ESTIMATE);

    double ***v_t = al(n1,n2,N0);
    double ***v_t1 = al(n1,n2,N0);

//    p=fftw_plan_dft_r2c_1d(N, in, out, FFTW_MEASURE);

#pragma omp parallel for num_threads(4)
    for (int i=0; i<n1;i++) {
        for (int j = 0; j < n2; j++) {
            in = v[i][j];
            fftw_execute_dft_r2c(p_fft,in,out);
            for(int k=0;k <N0; k++){
                v_t[i][j][k]=out[k][0];
                v_t1[i][j][k]=out[k][1];
            }              
        }
    }

    t1=gettime();
    cout<<"fft time: "<<t1-t0<<endl;

//    display(v_t,n1,n2,N0);
//    display(v_t1,n1,n2,N0);

//    for (int k=0; k<2; k++){
//        for (int i=0; i<n1;i++) {
//            for (int j = 0; j < n2; j++) {
//                cout << v_t[i][j][k] << " " << v_t1[i][j][k] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }

    t1=gettime();

    p_fft=fftw_plan_dft_c2r_1d(n3,out,in,FFTW_ESTIMATE);   

    //fftw_execute(p_fft);

    double ***v_o=al(n1,n2,n3);

    for (int i=0; i<n1;i++) {
        for (int j = 0; j < n2; j++) {
            for (int k = 0; k< N0; k++) {
                out[k][0] = v_t[i][j][k];
                out[k][1] = v_t1[i][j][k];
            }
            fftw_execute_dft_c2r(p_fft,out,in);
//            v_o[i][j] = in;   
            for(int k=0;k <n3; k++){
                v_o[i][j][k] = 1.0/n3*in[k];
            }             
        }
    }
    t2=gettime();

    cout << "ifft time: " << t2-t1 << endl;

    fftw_destroy_plan(p_fft);
    return 0;
}
