#!/bin/sh
rm info.txt
g++ compression.cpp -std=c++11
a=0
while [ "$a" -lt 30 ]
do
  ./a.out "$a"
  a=`expr $a + 1`
done
g++ calculateTime.cpp
./a.out
