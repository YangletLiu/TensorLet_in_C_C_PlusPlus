# A C++ library for popular tensor decompositions

## Pre-requisite
Users need to install the following packages and add them to specific paths according to your CMakeLists.txt file.

1. Basic matrix library, Eigen: https://eigen.tuxfamily.org/dox/   
2. Intel Math Kernel Library (MKL): https://software.intel.com/en-us/mkl  
3. cmake version 3.12 or greater: https://cmake.org/  
4. Fastest Fourier Transform in the West (FFTW): http://www.fftw.org/  
(Optional) OpenMP   

We tested our code on Ubuntu Linux. 
                
## CANDECOMP/PARAFAC decomposition  
CP decomposition via alternating least squares (ALS), which is realized in cp_als.cpp file.    

The struct type is defined as:  
>template\<class T\>  
>struct cp_mats{  
>&emsp;&emsp;    Mat\<T\> A,B,C;  
>};  
where matrix A,B and C are the corresponding factors.   

You can call cp_als function like:   

        cp_mats<T> cp_als(Cube<T> &a, int r,int max_iter， T tol);    
	
The type \<T\> can be float and double, you can run the test.cpp file to test the algorithm.

### Tucker decomposition
Tucker decomposition via Higher Order SVD (HOSVD), which is realized in tucker_hosvd.cpp file.  
Tucker decomposition via Higher Order Orthogonal Iteration (HOOI), which is realized in tucker_hooi.cpp file.    

The struct type tucker_core is defined as:  
>template\<class T\>    
>struct tucker_core{  
>&emsp;&emsp;  Cube<T> g, Mat\<T\> u1, u2, u3;  
>};  

You can call hosvd function like: 

        tucker_core A = hosvd(Cube<T> &a, int r1, int r2, int r3);    
	
You can call hooi function like:   

        tucker_core A = hooi(Cube<T> &a, int r1, int r2, int r3, T tol);      

The type \<T\> can be float and double, you can run the test.cpp file to test the algorithm.

### T-SVD decomposition
You can simply run main function in T-SVD directory to test T-SVD algorithm. The cpp file for T-SVD, like cp_als.cpp, will be provided in the near furture.

### API Reference
For more API details, please refer to the tensor.h file, where all definitations and corresponding illustrations is provied therein. The corresponding functions is realized in tensor.cpp file.

cp_mats<T> cp_als(Cube<T> &a, int r,int max_iter, T tol);       
### Parameters: 
	Cube<T>: tensor; 
	rank, int r: Number of components.   
	max_iter : int: Maximum number of iteration   
	tol : float, optional  
	(Default: 1e-6) Relative reconstruction error tolerance. The algorithm is considered to have found the global minimum when the reconstruction error is less than tol.

### References
[1] Xiao-Yang Liu and Xiaodong Wang. Fourth-order Tensors with Multidimensional Discrete Transforms, 2017. https://arxiv.org/abs/1705.01576

[2] Kilmer, M. E., Braman, K., Hao, N., & Hoover, R. C. (2013). Third-order tensors as operators on matrices: A theoretical and computational framework with applications in imaging. SIAM Journal on Matrix Analysis and Applications, 34(1), 148-172.

[3] Kjolstad, Fredrik, Shoaib Kamil, Stephen Chou, David Lugato, and Saman Amarasinghe. "The tensor algebra compiler." Proceedings of the ACM on Programming Languages 1, no. OOPSLA (2017): 77.

[4] De Lathauwer L, De Moor B, Vandewalle J. A multilinear singular value decomposition[J]. SIAM journal on Matrix Analysis and Applications, 2000, 21(4): 1253-1278.

[5] Kolda T G, Bader B W. Tensor decompositions and applications[J]. SIAM review, 2009, 51(3): 455-500.

[6] Papalexakis E E, Faloutsos C, Sidiropoulos N D. Tensors for data mining and data fusion: Models, applications, and scalable algorithms[J]. ACM Transactions on Intelligent Systems and Technology (TIST), 2017, 8(2): 16.

[7] Liavas A P, Sidiropoulos N D. Parallel algorithms for constrained tensor factorization via alternating direction method of multipliers[J]. IEEE Transactions on Signal Processing, 2015, 63(20): 5450-5463.

[8] Ravindran N, Sidiropoulos N D, Smith S, et al. Memory-efficient parallel computation of tensor and matrix products for big tensor decomposition[C]//Signals, Systems and Computers, 2014 48th Asilomar Conference on. IEEE, 2014: 581-585.

