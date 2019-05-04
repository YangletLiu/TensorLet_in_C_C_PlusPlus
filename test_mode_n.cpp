//
// Created by jcfei on 19-5-4.
//

/*******************************
        mode n product
*******************************/
//    VSLStreamStatePtr stream;
//
//    srand((unsigned)time(NULL));
//    MKL_INT SEED = rand();
//    MKL_INT status[3];
//    vslNewStream(&stream,VSL_BRNG_MCG31, SEED);
//
//    double* X1_times_X1T = (double*)mkl_malloc(n1 * n1 * sizeof(double), 64);
//    double* X2_times_X2T = (double*)mkl_malloc(n2 * n2 * sizeof(double), 64);
//    double* X3_times_X3T = (double*)mkl_malloc(n3 * n3 * sizeof(double), 64);
//
//    status[0] = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n1 * n1, X1_times_X1T, 0, 1);
//    status[1] = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n2 * n2, X2_times_X2T, 0, 1);
//    status[2] = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n3 * n3, X3_times_X3T, 0, 1);
//
//    vslDeleteStream(&stream);
//
//    double* u1t_times_x1 = (double*)mkl_malloc(r1 * n2 * n3 * sizeof(double), 64);
//
//    for(int i = 0; i < 27; i++){
//        cout << a.pointer[i] << " " ;
//    }
//    cout << endl;
//
//    for(int i = 0; i < 9; i++){
//        cout << X1_times_X1T[i] << " " ;
//    }
//    cout << endl;
//
//    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, r1, n2 * n3, n1,
//                1, X1_times_X1T + (n1 - r1) * n1, n1, a.pointer, n1,
//                0, u1t_times_x1, r1); // U1^t * X(1)
//
//    for(int i = 0; i < 18; i++){
//        cout << u1t_times_x1[i] << " " ;
//    }
//    cout << endl;
//
//    for(int i = 0; i < 9; i++){
//        cout << X2_times_X2T[i] << " " ;
//    }
//    cout << endl;
//
//    double* u2t_times_u1t_times_x1 = (double*)mkl_malloc(r1 * r2 * n3 * sizeof(double), 64);
//
//    for(MKL_INT i = 0; i < n3; i++){
//        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, r1, r2, n2,
//                    1, u1t_times_x1 + i * r1 * n2, r1, X2_times_X2T + (n2 - r2) * n2, n2,
//                    0, u2t_times_u1t_times_x1 + i * r1 * r2, r1);  // U2^t * ( U1^t * X(1) )_(2)
//    }
//
//    MKL_free(u1t_times_x1);
//
//    for(int i = 0; i < 12; i++){
//        cout << u2t_times_u1t_times_x1[i] << " " ;
//    }
//    cout << endl;
//
//
//    for(int i = 0; i < 9; i++){
//        cout << X3_times_X3T[i] << " " ;
//    }
//    cout << endl;
//
//    double* g = (double*)mkl_malloc(r1 * r2 * r3 * sizeof(double), 64);
//
//    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, r1 * r2, r3, n3,
//                1, u2t_times_u1t_times_x1, r1 * r2, X3_times_X3T + (n3 - r3) * n3, n3,
//                0, g, r1 * r2); // g
//
//    for(int i = 0; i < 8; i++){
//        cout << g[i] << " " ;
//    }
//    cout << endl;


//mode-n-product.cpp test
//

//    t0=gettime();
//    a.mode_n_product(a.pointer, u1t_times_x1, 1);
//    t1=gettime();
//    cout << "U1X1 time: " << t1 - t0 << endl;
//    MKL_free(u1t_times_x1);
//
//
//    double* u2t_times_u1t_times_x1 = (double*)mkl_malloc(n1 * n2 * n3 * sizeof(double), 64);
//
//    t0=gettime();
//    a.mode_n_product(X2_times_X2T,u2t_times_u1t_times_x1,2);
//    mode_n_product(a,X2_times_X2T,u2t_times_u1t_times_x1,2);
//    t1=gettime();
//    cout << "U2X2 time: " << t1 - t0 << endl;
//
//    MKL_free(u2t_times_u1t_times_x1);
//
//    double* u3t_times_u1t_times_x1 = (double*)mkl_malloc(n1 * n2 * n3 * sizeof(double), 64);
//    t0=gettime();
//    a.mode_n_product(X3_times_X3T,u3t_times_u1t_times_x1,2);
//    mode_n_product(a,X3_times_X3T,u3t_times_u1t_times_x1,2);
//    t1=gettime();
//    cout << "U3X3 time: " << t1 - t0 << endl;
//
//    double norm_a = cblas_dnrm2(n1 * n2 * n3, a.pointer, 1);
//    double norm_u = cblas_dnrm2(n1 * n2 , X2_times_X2T, 1);
//    cout << norm_a << " " << norm_u << endl;
