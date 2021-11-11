#!/usr/bin/env bash

set -x # DEBUG
set -u # strict

make

# train a SVR with a linear kernel; C=0.005 is close to optimal for this dataset; 5xCV
./svmwrap -q --pairs -np 5 --NxCV 5 -c 0.005  --feats 5183 -i data/CHEMBL2842_std.AP

# train a SVR with RBF kernel (default e, good C and gamma are known); 5xCV
./svmwrap -q --pairs -np 5 --NxCV 5 -c 10 -g 0.0005 --kernel RBF --feats 5183 \
          -i data/CHEMBL2842_std.AP
