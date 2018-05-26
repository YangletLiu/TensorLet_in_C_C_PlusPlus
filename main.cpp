#include <complex>
#include <fftw3.h>
#include <math.h>
#include <iostream>

#define N 8
using namespace std;

int main(int argc, char * argv[]){

    fftw_complex in[N], out[N];
    fftw_plan p;

    p=fftw_plan_dft_1d(N,in,out,FFTW_FORWARD,FFTW_MEASURE);
    for(int i=0;i <N;i ++) {
        in[i][0]=i;
        in[i][1]=0.0;
    }


    fftw_execute(p);

    for(int i=0;i <N;i ++){
        cout<<out[i][0]<<" "<<out[i][1]<<endl;
    }

    complex<double> temp = 0.0;
    for(int k =0; k < N; k ++){
        double pi = 4*atan(1.0);
        temp += exp(complex<double>(0.0,-2.0*pi*3*k/N))*complex<double>(in[k][0],in[k][1]);
    }
    cout<<"out[3] is "<<temp<<endl;

    fftw_complex out1[N];

    fftw_plan p1;

    p1=fftw_plan_dft_1d(N,out1,in,FFTW_BACKWARD,FFTW_MEASURE);

    for(int i=0;i <N;i ++){
        out1[i][0]=out[i][0];
        out1[i][1]=out[i][1];
    }

    fftw_execute(p1);

    for(int i=0;i <N;i ++){
        cout<<in[i][0]<<" "<<in[i][1]<<endl;
    }



    fftw_destroy_plan(p);
    fftw_destroy_plan(p1);
    return 1;
}
