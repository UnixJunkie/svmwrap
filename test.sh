#!/usr/bin/env bash

make

# train a linear-SVR; C=0.005 is close to optimal for this dataset
./svmwrap -q -np 5 --NxCV 5 -c 0.005 --pairs -i data/CHEMBL2842_std.AP
