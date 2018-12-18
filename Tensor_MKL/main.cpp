#include<stdio.h>
#include<stdlib.h>

#include<mkl.h>

int main()
{
    float *A, *B, *C;
    int m = 100, n = 100, k = 100;//A维度2*3,B维度2*3(计算时候转置),C维度2*2
    int a = 1, b = 1;//缩放因子
    A = (float *)mkl_malloc(m*n*sizeof(float), 64);
    B = (float *)mkl_malloc(k*n*sizeof(float), 64);
    C = (float *)mkl_malloc(m*k*sizeof(float), 64);

    printf("矩阵1为\n");
    for (int i = 0; i < m*n; i++)
    {
        if (i != 0 && i%n == 0)
            printf("\n");
        A[i] = i + 1;
        printf("%2.0f", A[i]);
    }
    printf("\n");

    printf("矩阵2为\n");
    for (int i = 0; i < k*n; i++)
    {
        if (i != 0 && i%n == 0)
            printf("\n");
        B[i] = 1;
        printf("%2.0f", B[i]);
    }
    printf("\n");

    printf("矩阵3为\n");
    for (int i = 0; i < m*k; i++)
    {
        if (i != 0 && i%k == 0)
            printf("\n");
        C[i] = i;
        printf("%2.0f", C[i]);
    }
    printf("\n");

    printf("结果矩阵\n");

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, k, n, a, A, n, B, n, b, C, k);//注意mkn的顺序☆
    for (int i = 0; i < m*k; i++)
    {
        if (i != 0 && i%k == 0)
            printf("\n");
        printf("%2.0f", C[i]);
    }
    printf("\n");

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    getchar();
    return 0;

}
