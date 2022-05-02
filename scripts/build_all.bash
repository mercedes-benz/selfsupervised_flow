#!/usr/bin/env bash
set -e

nvidia-smi
cd "${0%/*}"
script_dir=$(pwd)
tf_user_ops_dir="../unsup_flow/tf_user_ops"

echo "Building tf_user_ops!"
cd $script_dir
cd $tf_user_ops_dir
./build_cmake.bash
cd $script_dir

exit 0
