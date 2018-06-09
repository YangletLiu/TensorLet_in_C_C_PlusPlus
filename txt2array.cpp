#include<iomanip>
#include <iostream>
#include <fstream>

#include <cstdlib>
#include <stdio.h>
#include <string.h>

using namespace std;

int main(int argc, char* argv[])
{
    int count = 0;
    FILE* fp;
    char str[100];

    int n1=4,n2=5,n3=6; // 2000-2000-200

    string ***p =new string**[n1];
    for (int i=0; i<n1;i++) {
        p[i] = new string *[n2];
        for (int j=0;j<n2;j++){
            p[i][j]=new string [n3];
        }
    }

    fp = fopen("/home/jcfei/Documents/MATLAB/data/b456.txt", "r");

    while (fscanf(fp, "%s", str) != EOF)
        {
        int NUM=count;
        int k=NUM%(n1*n2);
        k=(NUM-k)/(n1*n2);
        int j=(NUM-k*n1*n2)%n2;
        int i=(NUM-k*n1*n2)/n2;
        p[i][j][k]=str;
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
    int count_0=0;
    for(int i=0; i<n1; i++) {
        for (int j = 0; j < n2; j++) {
            for(int k=0; k<n3; k++){
                tmp=p[i][j][k];
                v[i][j][k]=atof(tmp.c_str());
                cout<<i<<" "<<j<<" "<<k<<endl;
                cout<<v[i][j][k]<<endl;
                count_0=count_0+1;
            }
        }
    }

cout<<count_0<<endl;
//    //dispaly order by the third dimension k
//    cout<<"Hello World"<<endl;
//    for (int k=0; k<n3;k++){
//        for (int i=0; i<n1;i++){
//            for(int j=0; j<n2;j++){
//                cout<<v[i][j][k]<<" ";
//            }
//            cout<<endl;
//        }
//        cout<<k<<endl;
//    }

    //write to txt
    ofstream a;
    a.open("/home/jcfei/Desktop/TensorC++/txtToarray/a3456.txt");
        for (int k=0; k<n3;k++){
            for (int i=0; i<n1;i++){
                for(int j=0; j<n2;j++){
                    a<<setiosflags(ios::right)<<setw(15)<<v[i][j][k]<<"  ";
                }
                a<<endl;
            }
        }
    return 0;
}