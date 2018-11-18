//
// Created by jcfei on 18-11-18.
//


#include <time.h>
#include <sys/time.h>

double gettime() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec+tv.tv_usec/1000000.0;
};




