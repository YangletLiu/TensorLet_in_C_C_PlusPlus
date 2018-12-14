#include "tensor.h"
#include "runningtime.h"

#include "ten2mat.cpp"
#include "cp_gen.cpp"
#include "cp_als.cpp"
#include "tucker_hosvd.cpp"
#include "tensor_hooi.cpp"
#include "tsvd.cpp"


//int main(){
//    int n1,n2,n3;
//    double t0,t1;
//    int I=100;
//    int R=10;
//
//    int iter=10;
//    double ccc =0.05;
//    t0=gettime();
//
//    cout << "hello world" << endl;
//    t1=gettime();
//
//    cout << "time:" <<t1-t0 <<endl;
//
//    return 0;
//}

#include <stdlib.h>
#include <stdio.h>
int main()
{
    int n1,n2,n3;
    int ***array;
    int i,j,k;
    puts("输入一维长度:");
    scanf("%d",&n1);
    puts("输入二维长度:");
    scanf("%d",&n2);
    puts("输入三维长度:");
    scanf("%d",&n3);
    array=(int***)malloc(n1*sizeof(int**));//第一维
    for(i=0; i<n1; i++)
    {
        array[i]=(int**)malloc(n2*sizeof(int*)); //第二维
        for(j=0;j<n2;j++)
        {
            array[i][j]=(int*)malloc(n3*sizeof(int)); //第三维
            for(k=0;k<n3;k++)
            {
                array[i][j][k]=i+n2*j+n3*k+1;
                printf("%d\t",array[i][j][k]);
            }
            puts("");
        }
        puts("");
    }
    for(i=0;i<n1;i++)
    {
        for(j=0;j<n2;j++)
        {
            free(array[i][j]);//释放第三维指针
        }
    }
    for(i=0;i<n1;i++)
    {
        free(array[i]);//释放第二维指针
    }
    free(array);//释放第一维指针

    return 0;
}