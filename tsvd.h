//
// Created by jcfei on 19-5-7.
//

#ifndef TENSOR_TSVD_H
#define TENSOR_TSVD_H

template<class datatype>
class tsvd_format{
public:
    datatype* U;
    datatype* Theta;
    datatype* V;
};


#endif //TENSOR_TSVD_H
