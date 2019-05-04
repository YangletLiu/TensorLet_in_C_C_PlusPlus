#! /bin/bash

cd ~/Desktop/install

sudo tar -xzvf cmake-3.13.2.tar.gz cmake-3.13.2/

cd cmake-3.13.2/

./bootstrap

make 

sudo make install
cmake --v
