//
// Created by jcfei on 18-12-4.
//

#ifndef TENSOR_FUNCTION_H
#define TENSOR_FUNCTION_H

#include <sys/time.h>
#include "time.h"

double gettime();

double gettime(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return (tv.tv_sec*1000+tv.tv_usec/1000.0)/1000.0; //time:s
};

#endif //TENSOR_FUNCTION_H
