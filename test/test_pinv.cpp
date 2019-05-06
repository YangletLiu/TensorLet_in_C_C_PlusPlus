//
// Created by jcfei on 19-5-6.
//

//            //pinv need test
//            int info[] = {-1, -1};
//
//            MKL_INT* ivpv = (MKL_INT*)mkl_malloc(r * sizeof(MKL_INT), 64);
//            datatype* work = (datatype*)mkl_malloc(r * sizeof(datatype),64);
//
//            MKL_INT order = r;
//
//            // 计算对称矩阵的伪逆矩阵， 仅计算上半部分 //不稳定。。。 修改
//            dsytrf("U", &order, ct_times_c_times_bt_times_b, &r, ivpv, work, &r, &info[0]);
//            dsytri("U", &order, ct_times_c_times_bt_times_b, &r, ivpv, work, &info[1]);
//
//            if(info[0] + info[1] != 0){
//                printf("Compute inverse failed during update A.");
//            }
//
//            cblas_dsymm(CblasColMajor, CblasRight, CblasUpper,
//                    n1, r, 1, ct_times_c_times_bt_times_b, r, x1_times_c_kr_b, n1,
//                    0, A, n1);


//            dsytrf( "U", &r, ct_times_c_times_at_times_a, &r, ivpv, work, &r, &info[0] );
//            dsytri( "U", &r, ct_times_c_times_at_times_a, &r, ivpv, work, &info[1] );
//
//            if(info[0] + info[1] != 0){
//                printf("Compute inverse failed during update B.");
//            }

//dsytrf( "U", &r, bt_times_b_times_at_times_a, &r, ivpv, work, &r, &info[0] );
//            dsytri( "U", &r, bt_times_b_times_at_times_a, &r, ivpv, work, &info[1] );
//            if(info[0] + info[1] != 0){
//                printf("Compute inverse failed during update C.");
//            }


//            MKL_free( ivpv );
//            MKL_free( work );