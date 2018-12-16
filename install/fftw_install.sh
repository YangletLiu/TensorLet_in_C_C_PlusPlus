#! /bin/bash

cd ~/Desktop/install

sudo tar -xzvf fftw-3.3.8.tar.gz fftw-3.3.8/

cd ./fftw-3.3.8/

./configure --enable-type-prefix --prefix=/home/jcfei/Desktop/install/ins --with-gcc --disable-fortran --enable-i386-hacks


make 

make install

make clean

./configure --enable-float --enable-type-prefix --prefix=/home/jcfei/Desktop/install/ins --with-gcc --disable-fortran --enable-i386-hacks

make install

make check



