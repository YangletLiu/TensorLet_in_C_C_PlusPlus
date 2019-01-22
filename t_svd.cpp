//
// Created by jcfei on 18-9-17.
//

#include "tensor.h"
#include "Tensor3D.h"

template<class datatype>
class tsvd_format{
public:
    datatype* U;
    datatype* Theta;
    datatype* V;
};

namespace TensorLet_decomposition {

    template<class datatype>
    tsvd_format<datatype> tsvd(Tensor3D<datatype>& a) {
        int *shape = a.getsize();  //dimension

        int n1 = shape[0]; int n2 =shape[1]; int n3 = shape[2];

        int N0 = floor(n3/2.0)+1;
//        cout << n3 << n2 << n1 << " " << N0 <<endl;

// fft(a,[],3)  mkl fft r2c
//        for(int i=0;i<n1*n2*n3;i++){
//            a.pointer[i] = i;
//        }

        MKL_Complex16* fft_x = (MKL_Complex16*)MKL_malloc(n1 * n2 * N0 * sizeof(MKL_Complex16), 64);

        DFTI_DESCRIPTOR_HANDLE desc_z;

        MKL_LONG strides_z[] = {0, n1 * n2};
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

        for(int i = 0; i < n1 * n2; i++){
            status = DftiComputeForward( desc_z, a.pointer+i, fft_x+i);
        }

//        for(int i=0; i< 100; i++){
//            cout << i << " "  << fft_x[i].real << " " << fft_x[i].imag << endl;
//        }

        status = DftiFreeDescriptor( &desc_z );

        /* malloc memory */
        MKL_Complex16* fft_u = (MKL_Complex16*)MKL_malloc( n1 * n1 * N0 * sizeof(MKL_Complex16), 64);
        MKL_Complex16* fft_vt = (MKL_Complex16*)MKL_malloc(n2 * n2 * N0 * sizeof(MKL_Complex16), 64);

        MKL_INT min_n1_n2 = min(n1, n2);
        double* fft_s = (double*)MKL_malloc(min_n1_n2 * N0 * sizeof(double), 64);

        double super[min_n1_n2-1];

        /* Compute SVD */

        MKL_INT info;
        for(int i = 0; i < N0; i++){
            info = LAPACKE_zgesvd( LAPACK_COL_MAJOR, 'A', 'A', n1, n2, fft_x + i * n1 * n2, n1, fft_s + i * min_n1_n2,
                                   fft_u + i * n1 * n1 , n1, fft_vt + i * n2 * n2, n2, super );
        }

        /* Check for convergence */

        if( info > 0 ) {
            printf( "The algorithm computing SVD failed to converge.\n" );
            exit( 1 );
        }

        /* cast mkl_complex to fftw_complex */

        fftw_complex* in_fft_u =reinterpret_cast<fftw_complex *> (fft_u);  // right
        fftw_complex* in_fft_vt =reinterpret_cast<fftw_complex *> (fft_vt);  // right


// many  ifft_u
        int rank = 1;
        int *n = &n3;
        int howmany = n1*n1;
        int idist = 1;
        int odist = 1;
        int istride = n1*n1;
        int ostride = n1*n1;
        int *inembed = n, *onembed = n;

        double* u = (double*)MKL_malloc(n1*n1*n3*sizeof(double),64);

        fftw_plan p_fft;
        p_fft = fftw_plan_many_dft_c2r(rank, n, howmany, in_fft_u, inembed, istride, idist,
                                       u, onembed, ostride, odist, FFTW_ESTIMATE);
        fftw_execute(p_fft);

// many  ifft_u

        double* vt = (double*)MKL_malloc(n2*n2*n3*sizeof(double),64);
        howmany = n2*n2;
        istride = n2*n2;
        ostride = n2*n2;
        p_fft = fftw_plan_many_dft_c2r(rank, n, howmany, in_fft_vt, inembed, istride, idist,
                                       vt, onembed, ostride, odist, FFTW_ESTIMATE);
        fftw_execute(p_fft);



// many  ifft_s

        fftw_complex* fft_s_complex = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * min_n1_n2 * N0);  // right
        double* s = (double*)MKL_malloc(min_n1_n2*n3*sizeof(double),64);


        for(int i=0; i< min_n1_n2 * N0; i++){
            fft_s_complex[i][0] = fft_s[i];
            fft_s_complex[i][1] = 0;
        }
        MKL_free(fft_s);

//        for(int i=0; i< min_n1_n2 * N0; i++){
//            cout << fft_s_complex[i][0] << " ";
//            cout << fft_s_complex[i][1] << endl;
//        }

        howmany = min_n1_n2;
        istride = n3;
        ostride = n3;

        p_fft = fftw_plan_many_dft_c2r(rank, n, howmany, fft_s_complex, inembed, istride, idist,
                                       s, onembed, ostride, odist, FFTW_ESTIMATE);

        fftw_execute(p_fft);
        fftw_destroy_plan(p_fft);
        MKL_free(fft_vt);
        MKL_free(fft_u);
        fftw_free(fft_s_complex);


//        for(int i=0; i< n1; i++){
//            cout << i << " " << s[i] << endl;
//        }
//        MKL_INT u_dimension[] = {n1,n1,n3};
//        MKL_INT v_dimension[] = {n2,n2,n3};

        tsvd_format<datatype> result;
        result.U = u;
        result.V = vt;
        result.Theta = s;
        return result;

    }

}

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