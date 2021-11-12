#!/usr/bin/env bash

set -x # DEBUG
set -u # strict

NPROCS=`getconf _NPROCESSORS_ONLN`

make

# train a SVR with a linear kernel; C=0.005 is close to optimal for this dataset; 5xCV
./svmwrap -q --pairs -np ${NPROCS} --NxCV 5 -c 0.005  --feats 5183 -i data/CHEMBL2842_std.AP

# train a SVR with RBF kernel (default e, good C and gamma are known); 5xCV
./svmwrap -q --pairs -np ${NPROCS} --NxCV 5 -c 10 -g 0.0005 --kernel RBF --feats 5183 \
          -i data/CHEMBL2842_std.AP

# train a SVR with the sigmoid kernel
./svmwrap -q --pairs -np ${NPROCS} --NxCV 5 \
          -c 50 -g '2e-05' -r 0.0 --kernel Sig \
          --feats 5183 -i data/CHEMBL2842_std.AP
