//
// Created by jcfei on 18-9-17.
//

#include "tensor.h"
#include "Tensor3D.h"
#include "tsvd.h"

namespace TensorLet_decomposition {

    template<class datatype>
    tsvd_format<datatype> tsvd(Tensor3D<datatype>& a) {

        int *shape = a.getsize();  //dimension

        int n1 = shape[0]; int n2 =shape[1]; int n3 = shape[2];

        int N0 = floor(n3/2.0)+1;

// fft(a,[],3)  mkl fft r2c
//        for(int i=0;i<n1*n2*n3;++i){
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

        for(int i = 0; i < n1 * n2; ++i){
            status = DftiComputeForward( desc_z, a.pointer+i, fft_x+i);
        }

//        for(int i=0; i< 100; ++i){
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

        for(int i = 0; i < N0; ++i){
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


        for(int i=0; i< min_n1_n2 * N0; ++i){
            fft_s_complex[i][0] = fft_s[i];
            fft_s_complex[i][1] = 0;
        }
        MKL_free(fft_s);

//        for(int i=0; i< min_n1_n2 * N0; ++i){
//            cout << fft_s_complex[i][0] << " ";
//            cout << fft_s_complex[i][1] << endl;
//        }

        howmany = min_n1_n2;
        istride = n3;
        ostride = n3;

        p_fft = fftw_plan_many_dft_c2r(rank, n, howmany, fft_s_complex, inembed, istride, idist,
                                       s, onembed, ostride, odist, FFTW_ESTIMATE);


        fftw_execute(p_fft);
//        cout << n3 << n2 << n1 << " " << N0 <<endl;
        MKL_free(fft_vt);
        MKL_free(fft_u);

//        fftw_destroy_plan(p_fft);
//        fftw_free(fft_s_complex);


        tsvd_format<datatype> result;
        result.U = u;
        result.V = vt;
        result.Theta = s;
        return result;

    }

}