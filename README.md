# A C++ library for popular tensor decompositions

## Documentation
All template functions are implemented in tensor2.0 directory. T-SVD algorithm is independently implemented in T-SVD directory ;  

You can implement CP decomposition using cp_als function:    
cp_als(Tensor a, rank r);  

You can implement Tucker decomposition using HOSVD function:    
HOSVD(Tensor a, r1, r2, r3);  

You can simply run main function in T-SVD directory to test T-SVD algorithm.  

For more API details, please refer to the tensor.h file.  

For the float preccision version, you can go to float directory;    

mat-Py directory is the corresponding Python and MATLAB implementation.    


### References
[1] Xiao-Yang Liu and Xiaodong Wang. Fourth-order Tensors with Multidimensional Discrete Transforms, 2017. https://arxiv.org/abs/1705.01576

[2] Kilmer, M. E., Braman, K., Hao, N., & Hoover, R. C. (2013). Third-order tensors as operators on matrices: A theoretical and computational framework with applications in imaging. SIAM Journal on Matrix Analysis and Applications, 34(1), 148-172.

[3] Kjolstad, Fredrik, Shoaib Kamil, Stephen Chou, David Lugato, and Saman Amarasinghe. "The tensor algebra compiler." Proceedings of the ACM on Programming Languages 1, no. OOPSLA (2017): 77.

[4] De Lathauwer L, De Moor B, Vandewalle J. A multilinear singular value decomposition[J]. SIAM journal on Matrix Analysis and Applications, 2000, 21(4): 1253-1278.

[5] Kolda T G, Bader B W. Tensor decompositions and applications[J]. SIAM review, 2009, 51(3): 455-500.

[6] Papalexakis E E, Faloutsos C, Sidiropoulos N D. Tensors for data mining and data fusion: Models, applications, and scalable algorithms[J]. ACM Transactions on Intelligent Systems and Technology (TIST), 2017, 8(2): 16.

[7] Liavas A P, Sidiropoulos N D. Parallel algorithms for constrained tensor factorization via alternating direction method of multipliers[J]. IEEE Transactions on Signal Processing, 2015, 63(20): 5450-5463.

[8] Ravindran N, Sidiropoulos N D, Smith S, et al. Memory-efficient parallel computation of tensor and matrix products for big tensor decomposition[C]//Signals, Systems and Computers, 2014 48th Asilomar Conference on. IEEE, 2014: 581-585.
