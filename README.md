## TenDeC++: Tensor Decomposition Library in C++

#### In TenDeC++, we implement four popular tensor decomposition methods, CANDECOMP/PARAFAC (CP) decomposition, Tucker decomposition, t-SVD, and Tensor-Train (TT) decomposition.  

* [TenDeC++](#readme)
	* [Installation](#Installation)
	* [User guide](#user-guide)
	* [API reference](#api-reference)
	* [Class list](#class-list)
	* [References](#references)
	
## Installation
<details>	
<summary> Pre-requisite </summary>  

Users need the following packages:   
	
1. Intel Math Kernel Library (MKL): https://software.intel.com/en-us/mkl  

2. Fastest Fourier Transform in the West (FFTW): http://www.fftw.org/   

3. OpenMP: https://www.openmp.org/  

4. cmake version 3.12 or greater: https://cmake.org/     

</details>

<details>	
<summary> Instructions </summary>  

We recommend users use TenDeC++ on Ubuntu and you can refer to the installation instructions in TenDeC++_Installation folder.     

You need to add them to specific paths according to your CMakeLists.txt file.    
For example, you can link MKL in  CMakeLists.txt file like:      

	include_directories(/opt/intel/mkl/include)  
	link_directories(/opt/intel/mkl/lib/intel64)  
	link_libraries(libmkl_core.a libmkl_blas95_ilp64.a libmkl_rt.so)  

<!--
We use MKL as basic matrix library for high performance and test our code on Ubuntu.  
Intel Math Kernel Library (Intel MKL) is a library which is hand-optimized specifically for Intel processors. Core math functions include BLAS, LAPACK, ScaLAPACK, sparse solvers, fast Fourier transforms, and vector math.  
1. Basic matrix library, Eigen: https://eigen.tuxfamily.org/dox/   
-->

[//]: # (This may be the most platform independent comment)  

</details>

## User guide
<details>	
<summary> Tensor basics </summary>

##### TenDeC++ provides basic tensor algebraic operations, such as addition and different multiplication methods. In TenDeC++, all third order tensors are objects of the Tensor3D template class. You can refer to Class list for more details. All matrix and vectors operations are provided by the third party library MKL. 
	
#### Examples
	Tensor3D<double> X(10,10,10);	// Creating a tensor
	X.random();                        // Random initialization
	double* A = X.unfold(1);	// mode-1 unfolding  
	double* B = X.unfold(2);	// mode-2 unfolding  
	double* C = X.unfold(3);	// mode-3 unfolding  
</details>

<details>	
<summary> CANDECOMP/PARAFAC decomposition </summary>

##### CP decomposition via alternating least squares (ALS), which is realized in cp_als.cpp.    

The decomposition components of CP is defined as:  
>template\<class datatype\>  
>class cp_format{  
>&emsp;&emsp;    datatype* factor[3];  
>};  

The template parameter "datatype" represents the data type of tensor and be "double" and "float";  
The factor is the matrix list of the corresponding CP decomposition.   

You can call cp_als function like:   

	Tensor3D<double> X = random(10,10,10);  
	cp_format<double> A = cp_als(X, int rank = 3, int max_iter = 1，double tol = 1e-6);    

where Tensor3D\<datatype\> represents the third-order tensor class.
</details>

<details>	
<summary> Tucker decomposition </summary>

##### Tucker decomposition via Higher Order SVD (HOSVD), which is realized in tucker_hosvd.cpp.  
##### Tucker decomposition via Higher Order Orthogonal Iteration (HOOI), which is realized in tucker_hooi.cpp.    

The decomposition components of tucker is defined as:  
>template\<class datatype\>    
>class tucker_format{  
>&emsp;&emsp;  Tensor3D\<datatype\> core; datatype* factor[3];   
>};  
where factor is the matrix list of the corresponding Tucker decomposition.   

You can call hosvd function like: 
	
	Tensor3D<double> X = random(10,10,10);    
	tucker_format<double> A = tucker_hosvd(X, int ranks[3]);    
	
You can call hooi function like:   

	Tensor3D<double> X = random(10,10,10);    
	tucker_format<double> A = tucker_hooi(X, int ranks[3], double tol);      

</details>

<details>	
<summary> t-SVD decomposition </summary>

##### t-SVD algorithm is implemented in t-SVD.cpp.

The decomposition components of t-SVD is defined as:  
>template\<class datatype\>    
>class tsvd_format{  
>&emsp;&emsp;  Tensor3D\<datatype\> U, Sigma, V;  
>};  

You can call tsvd function like:   
	
	Tensor3D<double> X = random(10,10,10);  
	tsvd_format<double> A = tsvd_decomposition(X);      
</details>

<details>	
<summary> Tensor Train decomposition  </summary>

##### Tensor Train decomposition via alternating least squares (ALS), which is realized in train.h file in the Tensor-Train directory.        

The decomposition components of tensortrain is defined as:    
>template\<class type\>    
>class tensortrain_format{  
>&emsp;&emsp;  Tensor3D\<datatype\> U;  
>&emsp;&emsp;  datatype* G1; datatype* G2;  
>};  

You can call tensortrain decomposition like:     
	
	Tensor3D<double> X = random(10,10,10);  
	tensortrain_format<double> A = tensortrain_decomposition(X, tol);      

</details>

## API reference

<details>	
<summary> CANDECOMP/PARAFAC decomposition via alternating least squares (ALS) </summary>

#### cp_format\<datatype\> cp_decomposition(Tensor3D\<datatype\>& tensor, int rank, int max_iter, datatype tol);    
##### Source: CP decomposition is realized in cp_als.cpp.    
### Parameters: 
	tensor: the address of tensor; 
	rank: int, number of components;   
	max_iter: int, maximum number of iteration;   
	tol: float, optional  
	(Default: 1e-6) Relative reconstruction error tolerance. The algorithm is considered to have found the global minimum when the reconstruction error is less than tol.  
### Returns:
	cp_format<datatype>: abstract data type（ADT） for the CP decomposition result.    
	template<class datatype>  
	class cp_format{  
	    datatype* factor[3];  
	};  
	where factor is the matrix list of the corresponding CP decomposition.   

</details>

<details>	
<summary> Tucker decomposition via High Order SVD (HOSVD) and High-Order Orthogonal Iteration (HOOI) </summary>
	
#### tucker_format\<datatype\> tucker_hosvd(Tensor3D\<datatype\>& tensor, int ranks[3]);      
##### Source: Tucker decomposition is realized in tucker_hosvd.cpp and tucker_hooi.cpp.     

### Parameters:	
	tensor: the address of tensor; 
	ranks: int array; size of the core tensor, (len(ranks) == tensor.ndim);  
	
#### tucker_format\<datatype\> tucker_hooi(Tensor3D\<datatype\>& tensor, int ranks[3], int max_iter, datatype tol);  
### Parameters:	
	tensor: the address of tensor; 
	int ranks[3]: size of the core tensor, (len(ranks) == tensor.ndim);  
	init : {‘svd’, ‘random’}, optional;  
	tol : float, optional  
	tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance  

### Returns:
	tucker_format<datatype>: abstract data type（ADT） for the Tucker decomposition result.    
	template<class datatype>    
	class tucker_format{  
	   Tensor3D<datatype> core; datatype* factor[3];   
	};  
</details>

<details>	
<summary> t-SVD decomposition </summary>
	
#### tsvd_decomposition\<datatype\> tsvd(Tensor3D\<datatype\>& tensor);      
##### Source: t-SVD is realized in t-SVD.cpp.     

### Parameters:	
	tensor: the address of tensor; 
	
### Returns:
	tsvd_format<type>: abstract data type（ADT） for the t-SVD decomposition result.    
	class tsvd_format{  
	   Tensor3D<datatype> U, Sigma, V;  
	};  	

For more details, please refer to the corresponding source files, where all definitations and corresponding illustrations is provied therein.
</details>

<details>	
<summary> Tensor Train decomposition  </summary>
	
#### tensortrain_decomposition\<datatype\> tensortrain_decomposition(Tensor3D\<datatype\>& tensor, datatype tol);      

##### Source: Tensor Train decomposition is realized in Tensor-Train/train.h.    

### Parameters:	

	tensor: the address of tensor; 
	tol: tolerance;
### Returns:
	tensortrain_format<datatype>: abstract data type（ADT） for the Tensor Train decomposition result.    
	class tensortrain_format{  
	   Tensor3D<datatype> U;    
	   datatype* G1;
	   datatype* G2;  
	};  	

</details>

## Class list
Here are the classes, structs, unions and interfaces with brief descriptions:

<details>	
<summary>
Tensor3D<datatype>
</summary>
In TenDeC++, all third order tensors are objects of the Tensor3D template class. You can refer to Tensor3D.h file.
	
##### Data Members

int shape[3]; // the dimension of the third order tensor;  
datatype * p; // a pointer point to tensor.  

##### Public Member Functions
| Member Functions  | Description |
| ------------- | ------------- |
| frobenius_norm  | the Frobenius norm of tensors |
| size  | Get the dimension of tensor |
| slice  | Return specific slice of tensor |
| tens2mat  | Returns the mode-mode unfolding of tensor with modes starting at 0  |
| mat2tens  | Refolds the mode-mode unfolding into a tensor of shape shape  |
| tens2vec  | 	Vectorises a tensor    |
| vec2tens  | Folds a vectorised tensor back into a tensor of shape shape |

</details>

<details>	
<summary>
cp_format<datatype>
</summary>
	
##### Public Member Functions  
| Member Functions  | Description |
| ------------- | ------------- |
| cp_to_tensor  | Turns the Khatri-product of matrices into a full tensor |
| cp_to_unfolded  | Turns the khatri-product of matrices into an unfolded tensor|
| cp_to_vec  | Turns the khatri-product of matrices into a vector  |
| cp_gen  | Generate a r-rank CP tensor  |

</details>

<details>	
<summary>
tucker_format<datatype>
</summary>
	
##### Public Member Functions  
| Member Functions  | Description |
| ------------- | ------------- |
| tucker_to_tensor  | Converts the Tucker tensor into a full tensor |
| tucker_to_unfolded  | Converts the Tucker decomposition into an unfolded tensor |
| tucker_to_vec  | Converts a Tucker decomposition into a vectorised tensor |

</details>

<details>	
<summary>
tsvd_format<datatype>
</summary>
	
##### Public Member Functions   
| Member Functions  | Description |
| ------------- | ------------- |
| tsvd_to_tensor  | Converts the t-SVD tensor into a full tensor |
| tsvd_to_unfolded  | Converts the t-SVD decomposition into an unfolded tensor |
| tsvd_to_vec  | Converts a t-SVD decomposition into a vectorised tensor |

</details>

<details>	
<summary>
tensortrain_format<datatype>
</summary>
	
##### Public Member Functions  

| Member Functions  | Description |
| ------------- | ------------- |
| tt_to_tensor  | Converts the TT tensor into a full tensor |
| tt_to_unfolded  | Converts the TT decomposition into an unfolded tensor |
| tt_to_vec  | Converts a TT decomposition into a vectorised tensor |

</details>


<details>	
<summary>
Functions of Tensors
</summary>

| Functions  | Description |
| ------------- | ------------- |
| inner  | Generalised inner products between tensors |
|  element_wise | Generalised element-wise products between tensors |
| n_mode_prod  | n-mode product of a tensor and a matrix or vector at the specified mode |
| t_prod  | t-product between tensors |

</details>

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

[9] Oseledets, Ivan V. "Tensor-train decomposition." SIAM Journal on Scientific Computing 33.5 (2011): 2295-2317.  

</details>


