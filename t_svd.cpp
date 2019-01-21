//
// Created by jcfei on 18-9-17.
//

#include "tensor.h"
#include "Tensor3D.h"

template<class datatype>
class tsvd_format{
    Tensor3D<datatype>* U;
    Tensor3D<datatype>* Theta;
    Tensor3D<datatype>* V;
};

namespace TensorLet_decomposition {

    template<class datatype>
    tsvd_format<datatype> tsvd(Tensor3D<datatype>& a) {
        int *shape = a.getsize();  //dimension
        int n1 = shape[0]; int n2 =shape[1]; int n3 = shape[2];

        int N0 = floor(n3/2.0)+1;
        cout << n3 << n2 << n1 << " " << N0 <<endl;

// fft(a,[],3)  mkl fft r2c
      for(int i=0;i<n1*n2*n3;i++){
            a.pointer[i] = 1;
        }

        MKL_Complex16* fft_x = (MKL_Complex16*)MKL_malloc(n1*n2*N0*sizeof(MKL_Complex16),64);

        DFTI_DESCRIPTOR_HANDLE desc_z;

        MKL_LONG strides_z[] = {0, n1*n2};
        MKL_LONG status;

        status = DftiCreateDescriptor( &desc_z,
                DFTI_DOUBLE, DFTI_REAL, 1, n3);

        status = DftiSetValue( desc_z,
                DFTI_PLACEMENT, DFTI_NOT_INPLACE);

        status = DftiSetValue( desc_z,
                DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);

        status = DftiSetValue(desc_z,
                DFTI_INPUT_STRIDES,  strides_z);
        status = DftiSetValue(desc_z,
                DFTI_OUTPUT_STRIDES,  strides_z);

        status = DftiCommitDescriptor( desc_z );

        for(int i=0; i < n1*n2; i++){
            status = DftiComputeForward( desc_z, a.pointer+i, fft_x+i);
        }

//        for(int i=0; i< n1*n2+10; i++){
//            cout << i << " "  << fft_x[i].real << " " << fft_x[i].imag << endl;
//        }

        status = DftiFreeDescriptor( &desc_z );


// many  ifft x
//        double* ifft_x  = (double*) fftw_malloc(sizeof(double) * n1*n2*N0 );
//        int rank = 1;
//        int * n = &N0;
//        int howmany = n1*n2;
//        int idist = 1;
//        int odist = 1;
//        int istride = n1*n2;
//        int ostride = n1*n2;
////        int * N0_p = &N0;
//        fftw_complex* in =reinterpret_cast<fftw_complex *> (fft_x);
//        int *inembed = n, *onembed = n;
//        fftw_plan p_fft;
//        p_fft = fftw_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride, idist,
//                                       ifft_x, onembed, ostride, odist, FFTW_ESTIMATE);
//        fftw_execute(p_fft);
//
//        for(int i=0; i< n1*n2+10; i++){
//            cout << i << " "  << ifft_x[i] << endl;
//        }
//
//        fftw_destroy_plan(p_fft);


//        #define M 3
//        #define N 4
//        #define LDA M
//        #define LDU M
//        #define LDVT N
//        /* Locals */
//        MKL_INT m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info;
//        /* Local arrays */
//        double s[M];
//        double superb[min(M,N)-1];
//        MKL_Complex16 u[LDU*M], vt[LDVT*N];
//        MKL_Complex16 x[LDA*N] = {
//                { 5.91, -5.69}, {-3.15, -4.08}, {-4.89,  4.20},
//                { 7.09,  2.72}, {-1.89,  3.27}, { 4.10, -6.70},
//                { 7.78, -4.06}, { 4.57, -2.07}, { 3.28, -3.84},
//                {-0.79, -7.21}, {-3.88, -3.30}, { 3.84,  1.19}
//        };
//        cout << x[0].imag << " " << x[0].real << endl;
//
//        /* Executable statements */
//        printf( "LAPACKE_zgesvd (column-major, high-level) Example Program Results\n" );
//        /* Compute SVD */
//        info = LAPACKE_zgesvd( LAPACK_COL_MAJOR, 'A', 'A', m, n, x, lda, s,
//                               u, ldu, vt, ldvt, superb );
//
////        cout << x[0].imag << " " << x[0].real << endl;
////        cout << s[0] << endl;
////        cout << u[0].imag << " " << u[0].real << endl;
//
//        /* Check for convergence */
//        if( info > 0 ) {
//            printf( "The algorithm computing SVD failed to converge.\n" );
//            exit( 1 );
//        }




//    Tensor3D<datatype> uf(n1,n1,N0),uf1(n1,n1,N0);
//    Tensor3D<datatype> theta(n1,n2,N0);
//    Tensor3D<datatype> vf(n2,n2,N0), vf1(n2,n2,N0);
//
////    cx_mat TMP = zeros<cx_mat>(n1, n2);
////    cx_mat TMPU = zeros<cx_mat>(n1, n1);
////    cx_mat TMPV = zeros<cx_mat>(n2, n2);
////    colvec TMPT;
//
////    for (int k = 0; k < N0; k++) {
////        for (int i = 0; i < n1; i++) {
////            for (int j = 0; j < n2; j++) {
////                TMP(i, j).real(v_t(i,j,k));
////                TMP(i, j).imag(v_t(i,j,N0+k));
////            }
////        }
////
////        svd(TMPU, TMPT, TMPV, TMP, "dc");
////        svd(TMPU,TMPT,TMPV,TMP,"std");
////
////        for (int i = 0; i < n1; i++) {
////            for (int j = 0; j < n1; j++) {
////                uf(i,j,k) = TMPU(i, j).real();
////                uf1(i,j,k) = TMPU(i, j).imag();
////            }
////        }
////        for (int i = 0; i < n2; i++) {
////            for (int j = 0; j < n2; j++) {
////                vf(i,j,k) = TMPV(i, j).real();
////                vf1(i,j,k) = TMPV(i, j).imag();
////            }
////        }
////        if (n1 <= n2) {
////            for (int i = 0; i < n1; i++) {
////                theta(i,i,k) = TMPT(i);
////            }
////        } else {
////            for (int i = 0; i < n2; i++) {
////                theta(i,i,k) = TMPT(i);
////            }
////        }
////    }
//    v_t.reset();
//
//
//    fftw_complex out1[N0]; //fftw_alloc_real()
//    double *in1 = fftw_alloc_real(n3);
//    p_fft = fftw_plan_dft_c2r_1d(n3, out1, in1, FFTW_ESTIMATE);
//
//    Tensor3D<datatype> U(n1,n1,n3);
//    for (int i = 0; i < n1; i++) {
//        for (int j = 0; j < n1; j++) {
//            for (int k = 0; k < N0; k++) {
//                out1[k][0] = uf(i,j,k);
//                out1[k][1] = uf1(i,j,k);
//            }
//            fftw_execute_dft_c2r(p_fft, out1, in1);
//            for (int k = 0; k < n3; k++) {
//                U(i,j,k) = 1.0 / n3 * in1[k];
//            }
//        }
//    }
//    uf.reset();
//    uf1.reset();
//
//    Tensor3D<datatype> V(n1,n1,n3);
//    for (int i = 0; i < n1; i++) {
//        for (int j = 0; j < n1; j++) {
//            for (int k = 0; k < N0; k++) {
//                out1[k][0] = vf(i,j,k);
//                out1[k][1] = vf1(i,j,k);
//            }
//            fftw_execute_dft_c2r(p_fft, out1, in1);
//            for (int k = 0; k < n3; k++) {
//                V(i,j,k) = 1.0 / n3 * in1[k];
//            }
//        }
//    }
//    uf.reset();
//    uf1.reset();
//
//    Tensor3D<datatype> Theta(n1,n2,n3);
//    for (int i = 0; i < n1; i++) {
//        for (int j = 0; j < n2; j++) {
//            for (int k = 0; k < N0; k++) {
//                out1[k][0] = theta(i,j,k);
//                out1[k][1] = 0;
//            }
//            fftw_execute_dft_c2r(p_fft, out1, in1);
//            for (int k = 0; k < n3; k++) {
//                Theta(i,j,k) = 1.0 / n3 * in1[k];
//            }
//        }
//    }
//    theta.reset();
//    fftw_destroy_plan(p_fft);
//
//    tsvd_format<datatype> c;
//    c.U = U;
//    c.Theta = Theta;
//    c.V = V;
//
//    return c;
    }

}

//        Tensor3D<datatype> v_t(n1,n2,2*N0);
//        fftw_complex out[N0]; //fftw_alloc_real()
//        double *in = fftw_alloc_real(n3);
//        fftw_plan p_fft;
//        p_fft = fftw_plan_dft_r2c_1d(n3, in, out, FFTW_ESTIMATE);
//        p_fft=fftw_plan_dft_r2c_1d(n3,in,out,FFTW_MEASURE);
//
//
//        MKL_Complex16* fft_x = (MKL_Complex16*)mkl_malloc(n1*n2*(n3/2+1)*sizeof(MKL_Complex16),64);
//

//        int count=0;
//        for(int i=0;i<32;i++){
//            for(int j=0;j<100;j++){
//                for(int k=0;k<10;k++){
//                    if(y[i][j][k] == 0) {count++;}
//                    cout << y[i][j][k];
//                }
//                cout << endl;
//            }
//        }
//        cout << count << endl;

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