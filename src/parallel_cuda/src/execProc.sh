#!/bin/sh
rm info.txt
make clean
make all
a=0
while [ "$a" -lt 30 ]  # do first 30 images from `images/`
do
  ./compressor "$a"
  a=`expr $a + 1`
done
