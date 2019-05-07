//
// Created by jcfei on 19-5-7.
//

#ifndef TENSOR_TUCKER_H
#define TENSOR_TUCKER_H

template<class datatype>
class tucker_format{
public:
    datatype* core;
    datatype* u1;
    datatype* u2;
    datatype* u3;
};

#endif //TENSOR_TUCKER_H
