//
// Created by jcfei on 19-5-6.
//

// ifft

//        for(int i=0;i<32;i++){
//            x_in[i] = 1000;
//        }
//
//        status_test = DftiCommitDescriptor( my_desc1_handle );
//        status_test = DftiComputeBackward( my_desc1_handle, x_in);
//
//        status_test = DftiFreeDescriptor(&my_desc1_handle);

//        DFTI_DESCRIPTOR_HANDLE my_desc2_handle;
//        status_test = DftiCreateDescriptor( &my_desc2_handle, DFTI_SINGLE,
//                                            DFTI_REAL, 1, 3);
//        status_test = DftiSetValue( my_desc2_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
//        status_test = DftiSetValue(my_desc2_handle,
//                                   DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
//        status_test = DftiSetValue(my_desc2_handle,
//                              DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT);
//        DftiSetValue(my_desc2_handle,  DFTI_INPUT_STRIDES,  strides_test);
//        DftiSetValue(my_desc2_handle,  DFTI_OUTPUT_STRIDES,  strides_test);
//
//        status_test = DftiCommitDescriptor( my_desc2_handle );
//        status_test = DftiComputeBackward( my_desc2_handle, y_out, x_in);
//        status_test = DftiFreeDescriptor(&my_desc2_handle);
//        for(int i=0;i<17;i++){
//            cout << x_in[i] << endl;
//        }

//3d fft cce format
//        float x[10][11][12];
//        for(int i=0;i<10;i++){
//            for(int j=0;j<11;j++){
//                for(int k=0;k<12;k++){
//                    x[i][j][k] = 1;
//                }
//            }
//        }
//
//        MKL_Complex8 y[10][11][12];
//        DFTI_DESCRIPTOR_HANDLE my_desc_handle;
//        MKL_LONG status, l[3];
//        MKL_LONG strides_out[4];
//
//        l[0] = 10; l[1] = 11; l[2] = 12;
//
//        strides_out[0] = 0; strides_out[1] = 132;
//        strides_out[2] = 12; strides_out[3] = 1;
//
//        status = DftiCreateDescriptor( &my_desc_handle,
//                DFTI_SINGLE, DFTI_REAL, 3, l );
//
//        status = DftiSetValue( my_desc_handle,
//                DFTI_PLACEMENT, DFTI_NOT_INPLACE );
//
//        status = DftiSetValue(my_desc_handle,
//                DFTI_OUTPUT_STRIDES, strides_out);
//
//        status = DftiSetValue(my_desc_handle,
//                DFTI_INPUT_STRIDES, strides_out);
//
//        status = DftiCommitDescriptor(my_desc_handle);
//
//        status = DftiComputeForward(my_desc_handle, x, y);
//
//        status = DftiFreeDescriptor(&my_desc_handle);
//
//        for(int i=0;i<10;i++){
//            for(int j=0;j<11;j++){
//                for(int k=0;k<12;k++){
//                    cout << y[i][j][k].real << " " << y[i][j][k].imag << " ";
//                }
//                cout << endl;
//            }
//            cout << endl;
//        }


//一维fft变换
//        float x_in[35];
//        for(int i=0;i<35;i++){
//            x_in[i] = i;
//        }
//        MKL_Complex8 y_out[35];
//
//        DFTI_DESCRIPTOR_HANDLE my_desc1_handle;
//
//        MKL_LONG strides_test[] = {0, 5};
//        MKL_LONG status_test;
//        status_test = DftiCreateDescriptor( &my_desc1_handle,
//                DFTI_SINGLE, DFTI_REAL, 1, 7);
//
//        status_test = DftiSetValue( my_desc1_handle,
//                DFTI_PLACEMENT, DFTI_NOT_INPLACE);
//
//        status_test = DftiSetValue(my_desc1_handle,
//                DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
//
//        DftiSetValue(my_desc1_handle,  DFTI_INPUT_STRIDES,  strides_test);
//        DftiSetValue(my_desc1_handle,  DFTI_OUTPUT_STRIDES,  strides_test);
//
//        status_test = DftiCommitDescriptor( my_desc1_handle );
//        for(int i=0; i < 5; i++){
//            status_test = DftiComputeForward( my_desc1_handle, x_in + i, y_out+i);
//        }
//        status_test = DftiFreeDescriptor( &my_desc1_handle );
//
//        for(int i=0;i<35;i++){
//            cout << y_out[i].real << " " << y_out[i].imag << endl;
//        }


//  1d fftw
//        fftw_complex out1[N0]; //fftw_alloc_real()
//        double *in = fftw_alloc_real(n3);
//        for(int i=0;i<n3;i++){
//            in[i] = 1;
//        }
//        fftw_plan p_fft1;
//        p_fft1 = fftw_plan_dft_r2c_1d(n3, in, out1, FFTW_ESTIMATE);
//        fftw_execute_dft_r2c(p_fft1, in, out1);
//        for(int i=0;i<N0;i++){
//            cout << out1[i][0] << " " << out1[i][1] << endl;
//        }
//        fftw_destroy_plan(p_fft1);

// many fftws
//  fftw_complex* fft_a  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n1*n2*N0 );
//        int rank = 1;
//        int * n = &n3;
//        int howmany = n1*n2;
//        int idist = 1;
//        int odist = 1;
//        int istride = n1*n2/2;
//        int ostride = n1*n2/2;
////        int * N0_p = &N0;
//
//        int *inembed = n, *onembed = n;
//        fftw_plan p_fft;
//        p_fft = fftw_plan_many_dft_r2c(rank, n, howmany, a.pointer, inembed, istride, idist,
//                fft_a, onembed, ostride, odist, FFTW_ESTIMATE);
//        fftw_execute(p_fft);
//
////        for(int i=0; i< n1*n2+10; i++){
////            cout << i << " "  << fft_a[i][0] << " " << fft_a[i][1] << endl;
////        }
//
//        fftw_destroy_plan(p_fft);

// initialization
//        VSLStreamStatePtr stream;
//        vslNewStream(&stream,VSL_BRNG_MCG59, 1);
//        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,n1*n2*n3,a.pointer,0,1);

//        for(int i=0;i<n1*n2*n3;i++){
//            a.pointer[i] = i;
//        }

// many dft_r2c
//        fftw_complex* fft_a  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n1*n2*N0 );
//        int rank = 1;
//        int * n = &n3;
//        int howmany = n1*n2;
//        int idist = 1;
//        int odist = 1;
//        int istride = n1*n2/2;
//        int ostride = n1*n2/2;
////        int * N0_p = &N0;
//
//        int *inembed = n, *onembed = n;
//        fftw_plan p_fft;
//        p_fft = fftw_plan_many_dft_r2c(rank, n, howmany, a.pointer, inembed, istride, idist,
//                fft_a, onembed, ostride, odist, FFTW_ESTIMATE);
//        fftw_execute(p_fft);

//        for(int i=0; i< n1*n2+10; i++){
//            cout << i << " "  << fft_a[i][0] << " " << fft_a[i][1] << endl;
//        }

//        fftw_destroy_plan(p_fft);


// many  ifft x
//        double* ifft_x  = (double*) fftw_malloc(sizeof(double) * n1*n2*n3 );
//        int rank = 1;
//        int * n = &n3;
//        int howmany = n1*n2;
//        int idist = 1;
//        int odist = 1;
//        int istride = n1*n2;
//        int ostride = n1*n2;
//
//        fftw_complex* in =reinterpret_cast<fftw_complex *> (fft_x);  // right
//
//        int *inembed = n, *onembed = n;
//
//        fftw_plan p_fft;
//        p_fft = fftw_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride, idist,
//                                       ifft_x, onembed, ostride, odist, FFTW_ESTIMATE);
//        fftw_execute(p_fft);
//
//        for(int i=0; i< 100; i++){
//            cout << i << " "  << ifft_x[i] << endl;
//        }
//
//        fftw_destroy_plan(p_fft);


// gesvd
//        const MKL_INT M = 3;
//        const MKL_INT N = 4;
////        M = 3; N = 4;
//
//        /* Locals */
//        MKL_INT m = M, n = N, lda = M, ldu = M, ldvt = N, info;
//        /* Local arrays */
//        double s[M];
//        double superb[min(M,N)-1];
//        MKL_Complex16 u[M*M], vt[N*N];
//        MKL_Complex16 x[M*N] = {
//                { 5.91, -5.69}, {-3.15, -4.08}, {-4.89,  4.20},
//                { 7.09,  2.72}, {-1.89,  3.27}, { 4.10, -6.70},
//                { 7.78, -4.06}, { 4.57, -2.07}, { 3.28, -3.84},
//                {-0.79, -7.21}, {-3.88, -3.30}, { 3.84,  1.19}
//        };
//        cout << x[0].imag << " " << x[0].real << endl;
//
//        /* Compute SVD */
//        info = LAPACKE_zgesvd( LAPACK_COL_MAJOR, 'A', 'A', m, n, x, lda, s,
//                               u, ldu, vt, ldvt, superb );
//
//        cout << x[0].imag << " " << x[0].real << endl;