# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT

import os
from os.path import dirname as up

import tensorflow as tf

base_path = up(os.path.realpath(__file__))
points_to_voxel = tf.load_op_library(
    os.path.join(base_path, "build/src/points_to_voxel/libtfops_points-to-voxel.so")
).points_to_voxel
k_nearest_neighbor_op = tf.load_op_library(
    os.path.join(base_path, "build/src/tf_knn_op/libk_nearest_neighbor_op.so")
).k_nearest_neighbor
