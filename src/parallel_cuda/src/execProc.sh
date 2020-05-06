#!/bin/sh
rm info.txt
make clean
make all
a=0
while [ "$a" -lt 30 ]    # this is loop1
do
  ./compressor "$a"
  a=`expr $a + 1`
done
# g++ compare.cpp -std=c++11
# ./a.out
