//
// Created by jcfei on 18-12-25.
//

#ifndef TENSOR_INTERFACE_H
#define TENSOR_INTERFACE_H


template <class datatype>
class Mat;

template <class datatype>
class Tensor3D {
private:
    int shape[3];
    datatype * pointer;
public:
    Tensor3D();
    Tensor3D(int *);
    Tensor3D(int, int, int);
    Tensor3D(const Tensor3D &);
    ~Tensor3D();

    Tensor3D& operator=(const Tensor3D&);
    Tensor3D& operator+=(const Tensor3D&);
    Tensor3D& operator-=(const Tensor3D&);
    Tensor3D& operator*=(const Tensor3D&);

    inline datatype& operator()(int i, int j, int k);

    Tensor3D& rand_tensor(int *);
//    Tensor3D& cp_tensor(int rank);
//    Tensor3D& tucker_tensor(int *);

    int * getsize(const Tensor3D<datatype> &a);

    double frobenius_norm(const Tensor3D<datatype>&);


    Mat<datatype>& tens2mat(const Tensor3D<datatype>&, int mode);
    Tensor3D<datatype>& mat2tens(const Mat<datatype>&, int mode);
//    Mat<datatype>& tens2vec(const Tensor3D<datatype>&, int mode);
//    Tensor3D<datatype>& vec2tens(const Mat<datatype>&, int mode);

};

template <class datatype>
double inner(const Tensor3D<datatype>&,const Tensor3D<datatype>&);

template <class datatype>
Mat<datatype>& n_mode_prod(const Tensor3D<datatype>&, Mat<datatype>&, int mode);

template <class datatype>
Tensor3D<datatype>& t_prod(const Tensor3D<datatype>&,const Tensor3D<datatype>&);


namespace tensorlet_decomposition{
    //CP 结构
    template <class datatype>
    class cp_decomposition{
        Mat<datatype> A,B,C;
    };


    //tucker 返回结构
    template<class datatype>
    class tucker_decomposition{
        Tensor3D<datatype> core;
        Mat<datatype> u1,u2,u3;
    };

    //t-SVD 返回结构
    template<class datatype>
    class tsvd_decomposition{
        Tensor3D<datatype> U,Theta,V;
    };

    //TT 返回结构
    template<class datatype>
    class tt_decomposition{
        Tensor3D<datatype> g;
        Mat<datatype> G1,G2;
    };

    template <class datatype>
    cp_decomposition<datatype> & cp_als(Tensor3D<datatype>& tensor, int rank, int max_iter, datatype tol);

    template <class datatype>
    tucker_decomposition<datatype> & tucker_hosvd(Tensor3D<datatype>& tensor, int* ranks);

    template <class datatype>
    tucker_decomposition<datatype> & tucker_hooi(Tensor3D<datatype>& tensor, int* ranks, int max_iter, datatype tol);

    template <class datatype>
    tsvd_decomposition<datatype> & tsvd(Tensor3D<datatype>& tensor);

    template <class datatype>
    tt_decomposition<datatype> & tt(Tensor3D<datatype>& tensor);

/******************************************************************************************************/
    template <class datatype>
    Tensor3D<datatype> & cp_gen(cp_decomposition<datatype>& a);

    template <class datatype>
    Tensor3D<datatype> & tucker_gen(tucker_decomposition<datatype>& a);

    template <class datatype>
    Tensor3D<datatype> & tsvd_gen(tsvd_decomposition<datatype>& a);

    template <class datatype>
    Tensor3D<datatype> & tt_gen(tt_decomposition<datatype>& a);
/******************************************************************************************************/


}   //namespace for decomposition

#endif //TENSOR_INTERFACE_H
