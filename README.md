## TensorLet: A C++ library for popular tensor decompositions

## Pre-requisite
<details>	
<summary> Users need to install the following packages and add them to specific paths according to your CMakeLists.txt file. </summary>

1. Basic matrix library, Eigen: https://eigen.tuxfamily.org/dox/   

2. Intel Math Kernel Library (MKL): https://software.intel.com/en-us/mkl  

3. (Optional) OpenMP: https://www.openmp.org/  

Intel Math Kernel Library (Intel MKL) is a library which is hand-optimized specifically for Intel processors. Core math functions include BLAS, LAPACK, ScaLAPACK, sparse solvers, fast Fourier transforms, and vector math. 

We use MKL as basic matrix library for high performance and test our code on Ubuntu Linux. 

<!--
3. cmake version 3.12 or greater: https://cmake.org/  

4. Fastest Fourier Transform in the West (FFTW): http://www.fftw.org/  
-->
[//]: # (This may be the most platform independent comment)

</details>

## User guide

<details>	
<summary> CANDECOMP/PARAFAC decomposition </summary>

CP decomposition via alternating least squares (ALS), which is realized in cp_als.cpp file.    

The decomposition components of CP is defined as:  
>template\<class type\>  
>class cp_decomposition{  
>&emsp;&emsp;    Mat\<type\> A,B,C;  
>};  
where Mat\<type\> is dense matrix class and matrix A,B and C are the corresponding factors.   

You can call cp_als function like:   

        cp_decomposition<double> cp_decomposition(Tensor3D<double> &a, int rank, int max_iter，double tol);    
	
The type double can replace with float, you can run the test.cpp file to test the algorithm.
</details>

<details>	
<summary> Tucker decomposition </summary>

Tucker decomposition via Higher Order SVD (HOSVD), which is realized in tucker_hosvd.cpp file.  
Tucker decomposition via Higher Order Orthogonal Iteration (HOOI), which is realized in tucker_hooi.cpp file.    

The decomposition components of tucker is defined as:  
>template\<class type\>    
>class tucker_decomposition{  
>&emsp;&emsp;  Tensor3D\<type\> g, Mat\<type\> u1, u2, u3;  
>};  

You can call hosvd function like: 

        tucker_decomposition<double> A = tucker_hosvd(Tensor3D<double> &a, int ranks[3]);    
	
You can call hooi function like:   

        tucker_decomposition<double> A = tucker_hooi(Tensor3D<double> &a, int ranks[3], double tol);      

The type double can replace with float, you can run the test.cpp file to test the algorithm.
</details>

<details>	
<summary> tSVD decomposition </summary>

tSVD algorithm is implemented in tsvd.cpp file.

The decomposition components of tSVD is defined as:  
>template\<class type\>    
>class tsvd_decomposition{  
>&emsp;&emsp;  Tensor3D\<type\> U, Sigma, V;  
>};  

You can call tsvd function like:   
	
        tsvd_decomposition<double> A = tsvd_decomposition(Tensor3D<double> &a);      
</details>

<details>	
<summary> Tensor Train decomposition  </summary>

Tensor Train decomposition via alternating least squares (ALS), which is realized in the Tensor-Train directory.        

You can find TensorTrain class in train.h file in the Tensor-Train directory.    

The decomposition components of tensortrain is defined as:    

You can call cp_als function like:     

       tensortrain_decomposition<double> A = tensortrain_decomposition(Tensor3D<double> &a, tol);      
</details>

## API Reference

<details>	
<summary> CANDECOMP/PARAFAC decomposition via alternating least squares (ALS) </summary>

### cp_decomposition\<type\> cp_decomposition(Tensor3D\<type\>& tensor, int rank, int max_iter，type tol);    
### Source: cp_als.cpp  
### Parameters: 
	tensor: the address of tensor; 
	rank: int, number of components;   
	max_iter: int, maximum number of iteration;   
	tol: float, optional  
	(Default: 1e-6) Relative reconstruction error tolerance. The algorithm is considered to have found the global minimum when the reconstruction error is less than tol.  
### Output:
	template<class type>  
	class cp_decomposition{  
	    Mat<type> A,B,C;  
	};  
	where matrix A,B and C are the corresponding factors.   
</details>

<details>	
<summary> Tucker decomposition via High Order SVD (HOSVD) and High-Order Orthogonal Iteration (HOOI) </summary>
	
### tucker_decomposition\<type\> tucker_hosvd(Tensor3D\<type\> &tensor, int ranks[3]);      
### Source: tucker_hosvd.cpp  
### Parameters:	
	tensor: the address of tensor; 
	ranks: int array; size of the core tensor, (len(ranks) == tensor.ndim);  
	
### tucker_decomposition\<type\> tucker_hooi(Tensor3D\<type\> &a, int ranks[3], T tol);  
### Source: tucker_hooi.cpp  
### Parameters:	
	Tensor3D<type>: tensor;  
	int ranks[3]: size of the core tensor, (len(ranks) == tensor.ndim);  
	init : {‘svd’, ‘random’}, optional;  
	tol : float, optional  
	tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance  

### Output:
	template<class type>    
	class tucker_decomposition{  
	   Tensor3D<type> g; Mat<type> u1, u2, u3;  
	};  
</details>

<details>	
<summary> tSVD decomposition API </summary>
	
### tsvd_decomposition\<type\> tSVD(Tensor3D\<type\> &a);      
### Source: tsvd.cpp  
### Parameters:	
	Tensor3D<type>: tensor;  
	
### Output:
	class tsvd_decomposition{  
	   Tensor3D<type> U, Sigma, V;  
	};  	

For more details, please refer to the corresponding source files, where all definitations and corresponding illustrations is provied therein.
</details>

<details>	
<summary> Tensor Train decomposition  </summary>
	
### Tensor Train decomposition 
### Source: Tensor-Train/train.h    
### Parameters:	
	Tensor3D<type>: tensor;  
### Output:
	class TensorTrain<type> 
	
</details>

## References
<details>	
<summary>
Main references
</summary>
[1] Xiao-Yang Liu and Xiaodong Wang. Fourth-order Tensors with Multidimensional Discrete Transforms, 2017. https://arxiv.org/abs/1705.01576

[2] Kilmer, M. E., Braman, K., Hao, N., & Hoover, R. C. (2013). Third-order tensors as operators on matrices: A theoretical and computational framework with applications in imaging. SIAM Journal on Matrix Analysis and Applications, 34(1), 148-172.

[3] Kjolstad, Fredrik, Shoaib Kamil, Stephen Chou, David Lugato, and Saman Amarasinghe. "The tensor algebra compiler." Proceedings of the ACM on Programming Languages 1, no. OOPSLA (2017): 77.

[4] De Lathauwer L, De Moor B, Vandewalle J. A multilinear singular value decomposition[J]. SIAM journal on Matrix Analysis and Applications, 2000, 21(4): 1253-1278.

[5] Kolda T G, Bader B W. Tensor decompositions and applications[J]. SIAM review, 2009, 51(3): 455-500.

[6] Papalexakis E E, Faloutsos C, Sidiropoulos N D. Tensors for data mining and data fusion: Models, applications, and scalable algorithms[J]. ACM Transactions on Intelligent Systems and Technology (TIST), 2017, 8(2): 16.

[7] Liavas A P, Sidiropoulos N D. Parallel algorithms for constrained tensor factorization via alternating direction method of multipliers[J]. IEEE Transactions on Signal Processing, 2015, 63(20): 5450-5463.

[8] Ravindran N, Sidiropoulos N D, Smith S, et al. Memory-efficient parallel computation of tensor and matrix products for big tensor decomposition[C]//Signals, Systems and Computers, 2014 48th Asilomar Conference on. IEEE, 2014: 581-585.

</details>


	<div id="NavbarMenu" class="navbar-menu">
		<div class="navbar-start">

			<a class="navbar-item" href="../installation.html">
				Install
			</a>
			<a class="navbar-item" href="index.html">
				User guide
			</a>
			<a class="navbar-item" href="../modules/api.html">
				API
			</a>
			<a class="navbar-item" href="../auto_examples/index.html">
				Examples
			</a>
			<a class="navbar-item" href="../authors.html">
				People
			</a>
			<a class="navbar-item" href="https://github.com/JeanKossaifi/tensorly-notebooks" target="_blank">
				Notebooks
			</a>

		</div>

		<div class="navbar-end">
			<a class="navbar-item is-tab tooltip is-hidden-touch" href="../index.html">
				<i class="fa fa-home" aria-hidden="true"></i>
				<span class="tooltiptext">Home page</span>
			</a>

			<a class="navbar-item is-tab tooltip is-hidden-touch" href="https://github.com/tensorly/tensorly" target="_blank">
				<span class="tooltiptext">Open project on Github</span>
				<span class="icon"><i class="fa fa-github"></i></span>
			</a>

		</div>
    </div>
