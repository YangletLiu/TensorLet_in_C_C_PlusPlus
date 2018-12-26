## TensorLet: A C++ library for popular tensor decompositions

* [TensorLet](#readme)
	* [Installation](#Installation)
	* [User guide](#user-guide)
	* [API Reference](#api-reference)
	* [Class List](#class-list)
	* [References](#references)
	
## Installation
<details>	
<summary> Pre-requisite </summary>
Users need the following packages:   
	
1. Basic matrix library, Armadillo: http://arma.sourceforge.net/  

2. Intel Math Kernel Library (MKL): https://software.intel.com/en-us/mkl  

3. (Optional) OpenMP: https://www.openmp.org/  

We use MKL as basic matrix library for high performance and test our code on Ubuntu.  

Intel Math Kernel Library (Intel MKL) is a library which is hand-optimized specifically for Intel processors. Core math functions include BLAS, LAPACK, ScaLAPACK, sparse solvers, fast Fourier transforms, and vector math.    
</details>

<details>	
<summary> Instructions </summary>
We recommend users use TensorLet on Ubuntu and you can refer to the installation instructions in Installation folder.   

You need to add them to specific paths according to your CMakeLists.txt file. For example, you can link MKL in  CMakeLists.txt file,     
"include_directories(/opt/intel/mkl/include)  
link_directories(/opt/intel/mkl/lib/intel64)  
link_libraries(libmkl_core.a libmkl_blas95_ilp64.a libmkl_rt.so)"  

<!--
3. cmake version 3.12 or greater: https://cmake.org/    

4. Fastest Fourier Transform in the West (FFTW): http://www.fftw.org/    
-->
[//]: # (This may be the most platform independent comment)  

</details>

## User guide
In TensorLet, all third order tensors are objects of the Tensor3D template class. You can refer to Class list for more details.

<details>	
<summary> Tensor basics </summary>

#### Creating a tensor
	Cube<double> tensor = random(10,10,10);    
#### Unfolding
	Mat<double> A = tensor.unfold(1) // mode-1 unfolding  
	Mat<double> B = tensor.unfold(2) // mode-2 unfolding  
	Mat<double> C = tensor.unfold(3) // mode-3 unfolding  

</details>

<details>	
<summary> CANDECOMP/PARAFAC decomposition </summary>

CP decomposition via alternating least squares (ALS), which is realized in cp_als.cpp.    

The decomposition components of CP is defined as:  
>template\<class type\>  
>class cp_decomposition{  
>&emsp;&emsp;    Mat\<type\> factor[3];  
>};  
where, Mat\<type\> is dense matrix class provided by the third party library Eigen/MKL;      
The template parameter <type> represents the data type of tensor and be <double> and <float>;  
The factor is the matrix list of the corresponding CP decomposition.   

You can call cp_als function like:   

	Cube<double> tensor = random(10,10,10);  
	cp_decomposition<double> A = cp_decomposition(tensor, int rank = 3, int max_iter = 1，double tol = 1e-6);    

where Tensor3D\<type\> represents the third-order tensor class.
</details>

<details>	
<summary> Tucker decomposition </summary>

Tucker decomposition via Higher Order SVD (HOSVD), which is realized in tucker_hosvd.cpp.  
Tucker decomposition via Higher Order Orthogonal Iteration (HOOI), which is realized in tucker_hooi.cpp.    

The decomposition components of tucker is defined as:  
>template\<class type\>    
>class tucker_decomposition{  
>&emsp;&emsp;  Cube\<type\> core, Mat\<type\> factor[3];   
>};  
where factor is the matrix list of the corresponding Tucker decomposition.   

You can call hosvd function like: 

        tucker_decomposition<double> A = tucker_hosvd(Cube<double> &tensor, int ranks[3]);    
	
You can call hooi function like:   

        tucker_decomposition<double> A = tucker_hooi(Cube<double> &tensor, int ranks[3], double tol);      

</details>

<details>	
<summary> t-SVD decomposition </summary>

t-SVD algorithm is implemented in t-SVD.cpp.

The decomposition components of t-SVD is defined as:  
>template\<class type\>    
>class tsvd_decomposition{  
>&emsp;&emsp;  Cube\<type\> U, Sigma, V;  
>};  

You can call tsvd function like:   
	
        tsvd_decomposition<double> A = tsvd_decomposition(Cube<double> &tensor);      
</details>

<details>	
<summary> Tensor Train decomposition  </summary>

Tensor Train decomposition via alternating least squares (ALS), which is realized in the Tensor-Train directory.        

You can find TensorTrain class in train.h file in the Tensor-Train directory.    

The decomposition components of tensortrain is defined as:    
>template\<class type\>    
>class tensortrain_decomposition{  
>&emsp;&emsp;  Cube\<type\> U;  
>&emsp;&emsp;  Mat<type> G1,G2;  
>};  

You can call cp_als function like:     

       tensortrain_decomposition<double> A = tensortrain_decomposition(Cube<double> &tensor, tol);      

</details>

## API Reference

<details>	
<summary> CANDECOMP/PARAFAC decomposition via alternating least squares (ALS) </summary>

### cp_decomposition\<type\> cp_decomposition(Cube\<type\>& tensor, int rank, int max_iter，type tol);    
#### Source: CP decomposition is realized in cp_als.cpp.    
### Parameters: 
	tensor: the address of tensor; 
	rank: int, number of components;   
	max_iter: int, maximum number of iteration;   
	tol: float, optional  
	(Default: 1e-6) Relative reconstruction error tolerance. The algorithm is considered to have found the global minimum when the reconstruction error is less than tol.  
### Returns:
	cp_decomposition<type>: abstract data type（ADT） for the CP decomposition result.    
	template<class type>  
	class cp_decomposition{  
	    Mat<type> factor[3];  
	};  
	where factor is the matrix list of the corresponding CP decomposition.   

</details>

<details>	
<summary> Tucker decomposition via High Order SVD (HOSVD) and High-Order Orthogonal Iteration (HOOI) </summary>
	
### tucker_decomposition\<type\> tucker_hosvd(Cube\<type\> &tensor, int ranks[3]);      
#### Source: Tucker decomposition is realized in tucker_hosvd.cpp and tucker_hooi.cpp.     

### Parameters:	
	tensor: the address of tensor; 
	ranks: int array; size of the core tensor, (len(ranks) == tensor.ndim);  
	
### tucker_decomposition\<type\> tucker_hooi(Cube\<type\> &tensor, int ranks[3], int max_iter, T tol);  
### Parameters:	
	tensor: the address of tensor; 
	int ranks[3]: size of the core tensor, (len(ranks) == tensor.ndim);  
	init : {‘svd’, ‘random’}, optional;  
	tol : float, optional  
	tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance  

### Returns:
	tucker_decomposition<type>: abstract data type（ADT） for the Tucker decomposition result.    
	template<class type>    
	class tucker_decomposition{  
	   Cube<type> core; Mat<type> factor[3];   
	};  
</details>

<details>	
<summary> t-SVD decomposition API </summary>
	
### tsvd_decomposition\<type\> tSVD(Cube\<type\> &tensor);      
#### Source: t-SVD is realized in t-SVD.cpp.     

### Parameters:	
	tensor: the address of tensor; 
	
### Returns:
	tsvd_decomposition<type>: abstract data type（ADT） for the t-SVD decomposition result.    
	class tsvd_decomposition{  
	   Cube<type> U, Sigma, V;  
	};  	

For more details, please refer to the corresponding source files, where all definitations and corresponding illustrations is provied therein.
</details>

<details>	
<summary> Tensor Train decomposition  </summary>
	
### tensortrain_decomposition\<type\> tensortrain_decomposition(Cube\<type\> &tensor, tol);      

#### Source: Tensor Train decomposition is realized in Tensor-Train/train.h.    

### Parameters:	
	tensor: the address of tensor; 
### Returns:
	tensortrain_decomposition<type>: abstract data type（ADT） for the Tensor Train decomposition result.    
	class tensortrain_decomposition{  
	   Cube<type> U;    
	   Mat<type> G1,G2;  
	};  	

</details>

## Class List
In TensorLet_Armadillo, all third order tensors are objects of the Cube template class. You can refer to Armadillo library.



## References
<details>	
<summary>
Main references
</summary>
[1] Kolda T G, Bader B W. Tensor decompositions and applications[J]. SIAM review, 2009, 51(3): 455-500.

[2] Kilmer, M. E., Braman, K., Hao, N., & Hoover, R. C. (2013). Third-order tensors as operators on matrices: A theoretical and computational framework with applications in imaging. SIAM Journal on Matrix Analysis and Applications, 34(1), 148-172.

[3] Kjolstad, Fredrik, Shoaib Kamil, Stephen Chou, David Lugato, and Saman Amarasinghe. "The tensor algebra compiler." Proceedings of the ACM on Programming Languages 1, no. OOPSLA (2017): 77.

[4] De Lathauwer L, De Moor B, Vandewalle J. A multilinear singular value decomposition[J]. SIAM journal on Matrix Analysis and Applications, 2000, 21(4): 1253-1278.

[5] Xiao-Yang Liu and Xiaodong Wang. Fourth-order Tensors with Multidimensional Discrete Transforms, 2017. https://arxiv.org/abs/1705.01576

[6] Papalexakis E E, Faloutsos C, Sidiropoulos N D. Tensors for data mining and data fusion: Models, applications, and scalable algorithms[J]. ACM Transactions on Intelligent Systems and Technology (TIST), 2017, 8(2): 16.

[7] Liavas A P, Sidiropoulos N D. Parallel algorithms for constrained tensor factorization via alternating direction method of multipliers[J]. IEEE Transactions on Signal Processing, 2015, 63(20): 5450-5463.

[8] Ravindran N, Sidiropoulos N D, Smith S, et al. Memory-efficient parallel computation of tensor and matrix products for big tensor decomposition[C]//Signals, Systems and Computers, 2014 48th Asilomar Conference on. IEEE, 2014: 581-585.

</details>

