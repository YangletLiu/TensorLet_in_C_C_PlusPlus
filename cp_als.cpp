//
// Created by jcfei on 18-9-9.
//

#include "tensor.h"
#include "Tensor3D.h"


template <class datatype>
class cp_format{
public:
    datatype* cp_A;
    datatype* cp_B;
    datatype* cp_C;
};
namespace TensorLet_decomposition{

    template<class datatype>
    cp_format<datatype> cp_als( Tensor3D<datatype> &a, int r, int max_iter = 1, datatype tolerance = 1e-6) {

        if( r == 0 ){
            printf("CP decomposition rank cannot be zero.");
            exit(1);
        }

        MKL_INT *shape = a.getsize();  //dimension

        MKL_INT n1 = shape[0]; MKL_INT n2 =shape[1]; MKL_INT n3 = shape[2];

/*****************************
******* Allocate memory ******
******************************/

        datatype* A = (datatype*)mkl_malloc(n1 * r * sizeof(datatype), 64);
        datatype* B = (datatype*)mkl_malloc(n2 * r * sizeof(datatype), 64);
        datatype* C = (datatype*)mkl_malloc(n3 * r * sizeof(datatype), 64);


        if( A == NULL || B == NULL || C == NULL ){
            printf("Cannot allocate enough memory for A, B, C.");
            exit(1);
        }

/****************************
******* randomization *******
*****************************/

        MKL_INT status[3]; // random state

        VSLStreamStatePtr stream;

// The seed need to test
        srand((unsigned)time(NULL));
        MKL_INT SEED = rand();

        vslNewStream(&stream,VSL_BRNG_MCG31, SEED);
        status[0] = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n1 * r, A, 0, 1);

//        srand((unsigned)time(NULL));
//        SEED = rand();
        status[1] = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n2 * r, B, 0, 1);

//        srand((unsigned)time(NULL));
//        SEED = rand();
        status[2] = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n3 * r, C, 0, 1);

        vslDeleteStream(&stream);

        if( status[0] + status[1] +status[2] != 0){
            printf("Random initialization failed for A, B, C.");
            exit(1);
        }

//        for(int i = 0; i < 10; i++){
//            cout << A[i] << " ";
//        }
//        cout << endl;

//        for(int i = 0; i < 10; i++){
//            cout << B[i] << " ";
//        }
//        cout << endl;

//        for(int i = 0; i < 10; i++){
//            cout << C[i] << " ";
//        }
//        cout << endl;

/************************
******* update A ********
************************/

        datatype* c_kr_b = (datatype*)mkl_malloc(n2 * n3 * r * sizeof(datatype), 64); // kr(c,b)
        datatype* x1_times_c_kr_b = (datatype*)mkl_malloc(n1 * r * sizeof(datatype), 64);  // X(1) * kr(c,b)

        if( c_kr_b == NULL || x1_times_c_kr_b == NULL ){
            printf("Cannot allocate enough memory for kr product.");
            exit(1);
        }

        // c_kr_b tested: right
        for(MKL_INT i = 0; i < r; i++){
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        n2, n3, 1, 1, B + i * n2, n2, C + i * n3, n3,
                        0, c_kr_b + i * n2 * n3, n2);
        }
//        for(int i = 0; i < 6; i++){
//            cout << C[i] << " ";
//        }
//        cout << endl;
//        for(int i = 0; i < 6; i++){
//            cout << B[i] << " ";
//        }
//        cout << endl;
//        for(int i = 0; i < 18; i++){
//            cout << c_kr_b[i] << " ";
//        }
//        cout << endl;
//
//        for(int i = 0; i < 27; i++){
//            cout << a.pointer[i] << " ";
//        }
//        cout << endl;

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n1, r, n2*n3, 1, a.pointer, n1, c_kr_b, n2 * n3,
                    0, x1_times_c_kr_b, n1); // X(1) * kr(c,b)

//        for(int i = 0; i < 6; i++){
//            cout << x1_times_c_kr_b[i] << " ";
//        }

        datatype* at_times_a = (datatype*)mkl_malloc(r * r * sizeof(datatype), 64); //A^t * A
        datatype* bt_times_b = (datatype*)mkl_malloc(r * r * sizeof(datatype), 64); //B^t * B
        datatype* ct_times_c = (datatype*)mkl_malloc(r * r * sizeof(datatype), 64); //C^t * C

        datatype* ct_times_c_times_bt_times_b = (datatype*)mkl_malloc(r * r * sizeof(datatype), 64); // c^t * c * b^t *b
        datatype* ct_times_c_times_at_times_a = (datatype*)mkl_malloc(r * r * sizeof(datatype), 64); // c^t * c * a^t *a
        datatype* bt_times_b_times_at_times_a = (datatype*)mkl_malloc(r * r * sizeof(datatype), 64); // b^t * b * a^t *a

        if( at_times_a == NULL || bt_times_b == NULL || ct_times_c == NULL
                || ct_times_c_times_at_times_a == NULL || ct_times_c_times_bt_times_b == NULL
                || bt_times_b_times_at_times_a == NULL ){
            printf("Cannot allocate enough memory for A * A^t.");
            exit(1);
        }

        cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                    r, r, n3, C, n3,
                    0, ct_times_c, r);

        cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                    r, r, n2, B, n2,
                    0, bt_times_b, r);

        vdMul(r * r, ct_times_c, bt_times_b, ct_times_c_times_bt_times_b);
        //pinv need test
        int info = -1;
        MKL_INT* ivpv=(MKL_INT*)mkl_malloc(r * sizeof(MKL_INT), 64);
        datatype* work=(datatype*)mkl_malloc(r * sizeof(datatype),64);

        MKL_INT order = r;
        dsytrf("U", &order, ct_times_c_times_bt_times_b, &r, ivpv, work, &r, &info);

        dsytri("U", &order, ct_times_c_times_bt_times_b, &r, ivpv, work, &info);

//    cblas_dsymm(CblasColMajor, CblasRight, CblasUpper, n1, r, 1, cal_a, r, c_times_ct_times_b_times_bt,n1,0,A,n1);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n1, r, r, 1, x1_times_c_kr_b, n1, ct_times_c_times_bt_times_b, r,
                    0, A, n1);

        MKL_free(c_kr_b);
        MKL_free(x1_times_c_kr_b);

/************************
******* update B ********
************************/

        datatype* c_kr_a = (datatype*)mkl_malloc(n3 * n1 * r * sizeof(datatype), 64);
        datatype* x2_times_c_kr_a = (datatype*)mkl_malloc(n2 * r * sizeof(datatype), 64);

        if( c_kr_a == NULL || x2_times_c_kr_a == NULL ){
            printf("Cannot allocate enough memory for kr product.");
            exit(1);
        }

        for(MKL_INT i = 0; i < r; i++){
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        n1, n3, 1, 1, A + i * n1, n1, C + i * n3, n3,
                        0, c_kr_a + i * n3 * n1, n1);  // kr(c,a)
        }

        for(MKL_INT i = 0; i < n3; i++){
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        n2, r, n1, 1, a.pointer + i * n1 * n2, n1, c_kr_a + i * n2, n1,
                        1, x2_times_c_kr_a, n2);  // X(2) * kr(c,a) rank update
        }

        cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                    r, r, n1, A, n3,
                    0, at_times_a, r);

        vdMul(r * r, ct_times_c, at_times_a, ct_times_c_times_at_times_a);

        dsytrf( "U", &r, ct_times_c_times_at_times_a, &r, ivpv, work, &r, &info );
        dsytri( "U", &r, ct_times_c_times_at_times_a, &r, ivpv, work, &info );

        cblas_dsymm( CblasColMajor, CblasRight, CblasUpper,
                     n2, r, 1, x2_times_c_kr_a, r, ct_times_c_times_at_times_a, n2,
                     0, B, n2 );

        MKL_free( c_kr_a );
        MKL_free( x2_times_c_kr_a );

/************************
******* update C ********
************************/

        datatype* b_kr_a = ( datatype* )mkl_malloc( n1 * n2 * r * sizeof( datatype ), 64 );
        datatype* x3_times_b_kr_a = ( datatype* )mkl_malloc( n3 * r * sizeof(datatype), 64 );

        if( b_kr_a == NULL || x3_times_b_kr_a == NULL ){
            printf("Cannot allocate enough memory for kr product.");
            exit(1);
        }

        for( MKL_INT i = 0; i < r; i++ ){
            cblas_dgemm( CblasColMajor, CblasNoTrans, CblasTrans,
                         n1, n2, 1, 1, A + i * n1, n1, B + i * n2, n2,
                         0, b_kr_a + i * n1 * n2, n1 );  // kr(b,a)
//        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n2,n1,1,1,B+i*n2,n2,A+i*n1,n1,0,b_kr_a+i*n1*n2,n2);  // kr(a,b)
        }


        cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasTrans,
                     n3, r, n1 * n2, 1, a.pointer, n1 * n2, b_kr_a, n1 * n2,
                     0, x3_times_b_kr_a, r ); //  X(3) * kr(b,a) CblasRowMajor


        cblas_dsyrk( CblasColMajor, CblasUpper, CblasTrans,
                     r, r, n2, B, n2,
                     0, bt_times_b, r);

        vdMul( r * r, bt_times_b, at_times_a, bt_times_b_times_at_times_a );

        dsytrf( "U", &r, bt_times_b_times_at_times_a, &r, ivpv, work, &r, &info );
        dsytri( "U", &r, bt_times_b_times_at_times_a, &r, ivpv, work, &info );
        cblas_dsymm( CblasRowMajor, CblasRight, CblasUpper,
                     n3, r, 1, x3_times_b_kr_a, r, bt_times_b_times_at_times_a, r,
                     0, C, r );


        /* clean up */
        MKL_free(b_kr_a);
        MKL_free( x3_times_b_kr_a );

        MKL_free( ivpv );
        MKL_free( work );
        MKL_free( at_times_a );
        MKL_free( bt_times_b );
        MKL_free( ct_times_c );
        MKL_free( ct_times_c_times_bt_times_b );
        MKL_free( ct_times_c_times_at_times_a );
        MKL_free( bt_times_b_times_at_times_a );

        cp_format<datatype> result;
        result.cp_A = A;
        result.cp_B = B;
        result.cp_C = C;

        return result;

    }

    //    template<class datatype>
//    cp_format<datatype> cp_als( Tensor3D<datatype> &a, int &r, int max_iter = 1, datatype tolerance = 1e-6) {
//
//        MKL_INT *shape = a.getsize();  //dimension
//        MKL_INT n1 = shape[0]; MKL_INT n2 =shape[1]; MKL_INT n3 = shape[2];
//
////random A,B,C
//
//        datatype* A = (datatype*)mkl_malloc(n1 * r * sizeof(datatype), 64);
//        datatype* B = (datatype*)mkl_malloc(n2 * r * sizeof(datatype), 64);
//        datatype* C = (datatype*)mkl_malloc(n3 * r * sizeof(datatype), 64);
//
//
//        if( A == NULL || B == NULL || C == NULL ){
//            printf("Cannot allocate enough memory.");
//            exit(1);
//        }
//
//
///******randomization *****/
//
//        MKL_INT status[3]; // random state
//
//        VSLStreamStatePtr stream;
//
//// The seed need to test
//        srand((unsigned)time(NULL));
//        MKL_INT SEED = rand();
//
//        vslNewStream(&stream,VSL_BRNG_MCG31, SEED);
//        status[0] = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n1 * r, A, 0, 1);
//
////        srand((unsigned)time(NULL));
////        SEED = rand();
//        status[1] = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n2 * r, B, 0, 1);
//
////        srand((unsigned)time(NULL));
////        SEED = rand();
//        status[2] = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n3 * r, C, 0, 1);
//
//        vslDeleteStream(&stream);
//
//        if( status[0] + status[1] +status[2] != 0){
//            printf("Random initialization failed.");
//            exit(1);
//        }
//
////        for(int i = 0; i < 10; i++){
////            cout << A[i] << " ";
////        }
////        cout << endl;
//
////        for(int i = 0; i < 10; i++){
////            cout << B[i] << " ";
////        }
////        cout << endl;
//
////        for(int i = 0; i < 10; i++){
////            cout << C[i] << " ";
////        }
////        cout << endl;
//
///***********************
//*******update A********
//***********************/
//
//        datatype* c_kr_b = (datatype*)mkl_malloc(n2 * n3 * r * sizeof(datatype), 64); // kr(c,b)
//        datatype* x1_times_c_kr_b = (datatype*)mkl_malloc(n1 * r * sizeof(datatype), 64);  // X(1) * kr(c,b)
//
//        if( c_kr_b == NULL || x1_times_c_kr_b == NULL ){
//            printf("Cannot allocate enough memory.");
//            exit(1);
//        }
//
//        // c_kr_b tested: right
//        for(MKL_INT i = 0; i < r; i++){
//            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
//                        n2, n3, 1, 1, B + i * n2, n2, C + i * n3, n3,
//                        0, c_kr_b + i * n2 * n3, n2);
//        }
////        for(int i = 0; i < 6; i++){
////            cout << C[i] << " ";
////        }
////        cout << endl;
////        for(int i = 0; i < 6; i++){
////            cout << B[i] << " ";
////        }
////        cout << endl;
////        for(int i = 0; i < 18; i++){
////            cout << c_kr_b[i] << " ";
////        }
////        cout << endl;
////
////        for(int i = 0; i < 27; i++){
////            cout << a.pointer[i] << " ";
////        }
////        cout << endl;
//
//        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
//                    n1, r, n2*n3, 1, a.pointer, n1, c_kr_b, n2 * n3,
//                    0, x1_times_c_kr_b, n1); // X(1) * kr(c,b)
//
////        for(int i = 0; i < 6; i++){
////            cout << x1_times_c_kr_b[i] << " ";
////        }
//
//        datatype* at_times_a = (datatype*)mkl_malloc(r * r * sizeof(datatype), 64); //A^t * A
//        datatype* bt_times_b = (datatype*)mkl_malloc(r * r * sizeof(datatype), 64); //B^t * B
//        datatype* ct_times_c = (datatype*)mkl_malloc(r * r * sizeof(datatype), 64); //C^t * C
//
//        datatype* ct_times_c_times_bt_times_b = (datatype*)mkl_malloc(r * r * sizeof(datatype), 64); // c^t * c * b^t *b
//        datatype* ct_times_c_times_at_times_a = (datatype*)mkl_malloc(r * r * sizeof(datatype), 64); // c^t * c * a^t *a
//        datatype* bt_times_b_times_at_times_a = (datatype*)mkl_malloc(r * r * sizeof(datatype), 64); // b^t * b * a^t *a
//
//        if( at_times_a == NULL || bt_times_b == NULL || ct_times_c == NULL
//            || ct_times_c_times_at_times_a == NULL || ct_times_c_times_bt_times_b == NULL
//            || bt_times_b_times_at_times_a == NULL ){
//            printf("Cannot allocate enough memory.");
//            exit(1);
//        }
//
//        cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
//                    r, r, n3, C, n3,
//                    0, ct_times_c, r);
//
//        cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
//                    r, r, n2, B, n2,
//                    0, bt_times_b, r);
//
//        vdMul(r * r, ct_times_c, bt_times_b, ct_times_c_times_bt_times_b);
//        //pinv need test
//        int info = -1;
//        MKL_INT* ivpv=(MKL_INT*)mkl_malloc(r * sizeof(MKL_INT), 64);
//        datatype* work=(datatype*)mkl_malloc(r * sizeof(datatype),64);
//
//        MKL_INT order = r;
//        dsytrf("U", &order, ct_times_c_times_bt_times_b, &r, ivpv, work, &r, &info);
//
//        dsytri("U", &order, ct_times_c_times_bt_times_b, &r, ivpv, work, &info);
//
////    cblas_dsymm(CblasColMajor, CblasRight, CblasUpper, n1, r, 1, cal_a, r, c_times_ct_times_b_times_bt,n1,0,A,n1);
//        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
//                    n1, r, r, 1, x1_times_c_kr_b, n1, ct_times_c_times_bt_times_b, r,
//                    0, A, n1);
//
//        MKL_free(c_kr_b);
//        MKL_free(x1_times_c_kr_b);
//
///************************
//********update B*********
//************************/
//
//        datatype* c_kr_a = (datatype*)mkl_malloc(n3 * n1 * r * sizeof(datatype), 64);
//        datatype* x2_times_c_kr_a = (datatype*)mkl_malloc(n2 * r * sizeof(datatype), 64);
//
//        if( c_kr_a == NULL || x2_times_c_kr_a == NULL ){
//            printf("Cannot allocate enough memory.");
//            exit(1);
//        }
//
//        for(MKL_INT i = 0; i < r; i++){
//            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
//                        n1, n3, 1, 1, A + i * n1, n1, C + i * n3, n3,
//                        0, c_kr_a + i * n3 * n1, n1);  // kr(c,a)
//        }
//
//        for(MKL_INT i = 0; i < n3; i++){
//            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
//                        n2, r, n1, 1, a.pointer + i * n1 * n2, n1, c_kr_a + i * n2, n1,
//                        1, x2_times_c_kr_a, n2);  // X(2) * kr(c,a) rank update
//        }
//
//        cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
//                    r, r, n1, A, n3,
//                    0, at_times_a, r);
//
//        vdMul(r * r, ct_times_c, at_times_a, ct_times_c_times_at_times_a);
//
//        dsytrf( "U", &r, ct_times_c_times_at_times_a, &r, ivpv, work, &r, &info );
//        dsytri( "U", &r, ct_times_c_times_at_times_a, &r, ivpv, work, &info );
//
//        cblas_dsymm( CblasColMajor, CblasRight, CblasUpper,
//                     n2, r, 1, x2_times_c_kr_a, r, ct_times_c_times_at_times_a, n2,
//                     0, B, n2 );
//
//        MKL_free( c_kr_a );
//        MKL_free( x2_times_c_kr_a );
//
///************************
//******* update C ********
//************************/
//
//        datatype* b_kr_a = ( datatype* )mkl_malloc( n1 * n2 * r * sizeof( datatype ), 64 );
//        datatype* x3_times_b_kr_a = ( datatype* )mkl_malloc( n3 * r * sizeof(datatype), 64 );
//
//        for( MKL_INT i = 0; i < r; i++ ){
//            cblas_dgemm( CblasColMajor, CblasNoTrans, CblasTrans,
//                         n1, n2, 1, 1, A + i * n1, n1, B + i * n2, n2,
//                         0, b_kr_a + i * n1 * n2, n1 );  // kr(b,a)
////        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n2,n1,1,1,B+i*n2,n2,A+i*n1,n1,0,b_kr_a+i*n1*n2,n2);  // kr(a,b)
//        }
//
//
//        cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasTrans,
//                     n3, r, n1 * n2, 1, a.pointer, n1 * n2, b_kr_a, n1 * n2,
//                     0, x3_times_b_kr_a, r ); //  X(3) * kr(b,a) CblasRowMajor
//
//
//        cblas_dsyrk( CblasColMajor, CblasUpper, CblasTrans,
//                     r, r, n2, B, n2,
//                     0, bt_times_b, r);
//
//        vdMul( r * r, bt_times_b, at_times_a, bt_times_b_times_at_times_a );
//
//        dsytrf( "U", &r, bt_times_b_times_at_times_a, &r, ivpv, work, &r, &info );
//        dsytri( "U", &r, bt_times_b_times_at_times_a, &r, ivpv, work, &info );
//        cblas_dsymm( CblasRowMajor, CblasRight, CblasUpper,
//                     n3, r, 1, x3_times_b_kr_a, r, bt_times_b_times_at_times_a, r,
//                     0, C, r );
//
//
//        MKL_free(b_kr_a);
//        MKL_free( x3_times_b_kr_a );
//
//        MKL_free( ivpv );
//        MKL_free( work );
//        MKL_free( at_times_a );
//        MKL_free( bt_times_b );
//        MKL_free( ct_times_c );
//        MKL_free( ct_times_c_times_bt_times_b );
//        MKL_free( ct_times_c_times_at_times_a );
//        MKL_free( bt_times_b_times_at_times_a );
//
//        cp_format<datatype> result;
//        result.cp_A = A;
//        result.cp_B = B;
//        result.cp_C = C;
//
//        return result;
//
//    }

}