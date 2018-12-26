#! /bin/bash

cd ~/Desktop/install

sudo tar -xzvf eigen-eigen-323c052e1731.tar.gz eigen-eigen-323c052e1731

cd ./eigen-eigen-323c052e1731

mkdir build

cmake .. 

make install

make check



