#!/usr/bin/env bash

TF_USER_OPS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd -P )"
echo $TF_USER_OPS_DIR

mkdir -p $TF_USER_OPS_DIR/build
(cd $TF_USER_OPS_DIR/build && cmake -DCMAKE_BUILD_TYPE=Release ..)
(cd $TF_USER_OPS_DIR/build && make -j12 -l12)
