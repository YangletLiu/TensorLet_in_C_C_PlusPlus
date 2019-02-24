//
// Created by jcfei on 19-2-24.
//



//cout << "dynamic thread " << mkl_get_dynamic() << endl;
//cout << "max thread " << mkl_get_max_threads() << endl;


//随机数生成
//t0=gettime();
//VSLStreamStatePtr stream;
//vslNewStream(&stream,VSL_BRNG_MCG31, 1);
//vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,10000000000,a.pointer,0,1);
//cout << a.pointer[999999999] << endl;

//    for (int i=1;i<2;i++){
//        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,100,r,0,1);
////        cout << r[0] << endl;
//        printf("%e \n",r[0]);
//    }
//vslDeleteStream(&stream);
//t1=gettime();
//cout << "time:" <<t1-t0 <<endl;

//how to calloc memory

//    double *cc = (double*)mkl_calloc(10000,sizeof(double),64);  //返回成功为1
//    cout << sizeof(cc) << endl;
//    cout << cc << endl;
//    mkl_free(cc);

// how to free memory
//    t0=gettime();
//    double * p = (double*)mkl_calloc(1000000,sizeof(double),64);
//    p = a.tens2mat(p,1);  //函数内声明后 调用
//    t1=gettime();
//    cout << "time:" <<t1-t0 <<endl;
//    mkl_free(p);   //必须 mkl_free


//    MKLVersion Version;
//    mkl_get_version(&Version);
//    printf("Major version: %d\n",Version.MajorVersion);
//    printf("Minor version: %d\n",Version.MinorVersion);
//    printf("Update version: %d\n",Version.UpdateVersion);
//    printf("Product status: %s\n",Version.ProductStatus);
//    printf("Build: %s\n",Version.Build);
//    printf("Platform: %s\n",Version.Platform);
//    printf("Processor optimization: %s\n",Version.Processor);
//    printf("================================================================\n");
//    printf("\n");

/*******************************
//basic operation test
*******************************/
//MKL_INT n1,n2,n3;
//n1=n2=n3=100;
//
//double t0,t1;
//t0=gettime();
//Tensor3D<double> a(n1,n2,n3); //element
//t1=gettime();
//cout << "Initialize time:" <<t1-t0 <<endl;
//cout << "size of class: " << sizeof(a) << endl;
//
//int *p1;
//p1 = a.getsize();
//cout << "shape: " << p1[2] << endl;
//cout << "element a: " << a(2,2,2) << endl;
//cout << "norm: " << a.frobenius_norm() << endl;
//
//MKL_INT aa[3]={10,10,10};
//Tensor3D<double> b(aa); // int array
////    b = a;
////    b += a;
//cout << "shape: " << b.getsize()[2] << '\n';
//b.random_tensor();
//for (int i = 0; i<1000; i++){
//cout << b.pointer[i] << endl;
//}
//
//Tensor3D<double> vd(a);
//cout << "shape: " << vd.getsize()[1] << endl;
//
//MKL_INT k = 2;
////    b = k*b;
////    b = a*b;
////    b = a+b+a;
//
//t0=gettime();
//a.random_tensor();
//t1=gettime();
//cout << "random time:" <<t1-t0 <<endl;
//cout << a.pointer[0] << endl;
//cout << a.pointer[1] << endl;
//cout << a.pointer[997002998] << endl;
//cout << a.pointer[999999999] << endl;
//
//cout << "norm: " << a.frobenius_norm() << endl;

//    for (int i = 0; i<n1*n2*n3; i++){
//        cout << a.pointer[i] << endl;
//    }

// 对称乘积
//double* tmp = (double*)mkl_malloc(n1*n1*sizeof(double),64);
//t0=gettime();
////    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n1,n1,n2*n3,1,a.pointer,n1,a.pointer,n1,0,tmp,n1);
//cblas_dsyrk(CblasColMajor,CblasUpper,CblasNoTrans,n1,n2*n3,1,a.pointer,n1,0,tmp,n1);
//t1=gettime();
//    cout << tmp[0] << endl;
//    cout << tmp[1] << endl;
//    cout << tmp[2] << endl;
//    cout << tmp[n1] << endl;
//    cout << tmp[n1+1] << endl;
//cout << "Product time:" <<t1-t0 <<endl;


// test x1 * x1t
//    t0=gettime();
//    double* X1_times_X1T = (double*)mkl_malloc(n1*n1*sizeof(double),64);
//    double* X2_times_X2T = (double*)mkl_malloc(n1*n1*sizeof(double),64);
//    double* X3_times_X3T = (double*)mkl_malloc(n1*n1*sizeof(double),64);
//
//    cblas_dsyrk(CblasColMajor,CblasUpper,CblasNoTrans,n1,n2*n3,1,a.pointer,n1,0,X1_times_X1T,n1);  //x1*x1^t
//    for(MKL_INT i=0;i<n3;i++){
//        cblas_dsyrk(CblasColMajor,CblasUpper,CblasTrans, n2, n1, 1, a.pointer+i*n1*n2, n1, 1, X2_times_X2T,n2);  // X(2) * X2^t rank update
//    }
//    cblas_dsyrk(CblasRowMajor,CblasUpper,CblasNoTrans,n3,n1*n2,1,a.pointer,n1*n2,0,X3_times_X3T,n3);  //x3*x3^t
//
//    t1=gettime();
//    cout << "Tucker time:" <<t1-t0 <<endl;


//    t0=gettime();
//    cblas_dsyrk( CblasColMajor, CblasUpper, CblasNoTrans,
//                 n1, n2 * n3, 1, a.pointer, n1,
//                 0, X1_times_X1T, n1 );  //x1*x1^t
//    t1=gettime();
//    cout << "X1X1T: " << t1 - t0 << endl;
//
//    t0=gettime();
//    for(MKL_INT i = 0; i < n3; i++){
//        cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
//                    n2, n1, 1, a.pointer + i * n1 * n2, n1,
//                    1, X2_times_X2T, n2);  // X(2) * X2^t rank update
////        cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
////                    n2, n1, 1, a.pointer + i * n1 * n2, n1,
////                    1, X1_times_X1T, n2);  // X(2) * X2^t rank update
//    }
//    t1=gettime();
//    cout << "X2X2T: " << t1 - t0 << endl;


/*******************************
/mode n product
*******************************/

//    VSLStreamStatePtr stream;
//
//// The seed need to test
//    srand((unsigned)time(NULL));
//    MKL_INT SEED = rand();
//    MKL_INT status[3];
//    vslNewStream(&stream,VSL_BRNG_MCG31, SEED);
//
////    double* X1_times_X1T = (double*)mkl_malloc(n1 * n1 * sizeof(double), 64);
////    status[0] = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n1 * n1, X1_times_X1T, 0, 1);
//
//    double* X2_times_X2T = (double*)mkl_malloc(n2 * n2 * sizeof(double), 64);
//    status[1] = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n2 * n2, X2_times_X2T, 0, 1);
//
////    status[2] = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n3 * r, C, 0, 1);
//
//    vslDeleteStream(&stream);

//    double* u1t_times_x1 = (double*)mkl_malloc(n1 * n2 * n3 * sizeof(double), 64);
//
//    t0=gettime();
//    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, n2 * n3, n1,
//                1, X1_times_X1T , n1, a.pointer, n1,
//                0, u1t_times_x1, n1); // U1^t * X(1)
//    t1=gettime();
//    cout << "U1X1: " << t1 - t0 << endl;
//
//    MKL_free(u1t_times_x1);


//    double* u2t_times_u1t_times_x1 = (double*)mkl_malloc(n1 * n2 * n3 * sizeof(double), 64);
//
//    t0=gettime();
//    for(MKL_INT i = 0; i < n3; i++){
//        cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, n2, n1, n2,
//                    1, X2_times_X2T, n2, a.pointer + i * n1 * n2, n2,
//                    0, u2t_times_u1t_times_x1 + i * n1 * n2, n1);  // U2^t * ( U1^
//                    // t * X(1) )_(2)
//    }
//    t1=gettime();
//    cout << "U2X2: " << t1 - t0 << endl;
//    MKL_free(u2t_times_u1t_times_x1);
//
//    double norm_a = cblas_dnrm2(n1 * n2 * n3, a.pointer, 1);
//    double norm_u = cblas_dnrm2(n1 * n2 , X2_times_X2T, 1);
//
//    cout << norm_a << " " << norm_u << endl;

//    a.mode_n_product(X2_times_X2T,u2t_times_u1t_times_x1,2);

//    mode_n_product(a,X2_times_X2T,u2t_times_u1t_times_x1,2);
