#!/bin/sh
rm info.txt
g++ compression.cpp -std=c++11 -fopenmp
a=0
while [ "$a" -lt 1 ]    # this is loop1
do
  ./a.out "$a"
  a=`expr $a + 1`
done
g++ calculateTime.cpp
./a.out
