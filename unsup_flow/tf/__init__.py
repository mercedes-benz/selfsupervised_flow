#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import functools

import tensorflow as tf

from .relaxed_tf_function import relaxed_tf_function  # noqa: F401


def shapes_broadcastable(shape_a, shape_b):
    return all(sa == 1 or sb == 1 or sa == sb for sa, sb in zip(shape_a, shape_b))


def make_str(tensor):
    return tf.as_string(tensor, shortest=True, precision=2)


def castf(tensor):
    if tensor.dtype not in {tf.float32, tf.float64}:
        tensor = tf.cast(tensor, tf.float32)
    return tensor


def cast32(tensor):
    if tensor.dtype in {tf.float32, tf.int32}:
        return tensor
    if tensor.dtype == tf.float64:
        return tf.cast(tensor, tf.float32)
    if tensor.dtype == tf.int64:
        return tf.cast(tensor, tf.int32)
    assert tensor.dtype in {tf.float32, tf.int32}, tensor.dtype


def cast64(tensor):
    if tensor.dtype in {tf.float64, tf.int64}:
        return tensor
    if tensor.dtype == tf.float32:
        return tf.cast(tensor, tf.float64)
    if tensor.dtype == tf.int32:
        return tf.cast(tensor, tf.int64)
    assert tensor.dtype in {tf.float64, tf.int64}, tensor.dtype


def func2namescope(function):
    @functools.wraps(function)
    def wrapped_function(*args, name=None, **kwargs):
        with tf.name_scope(name if name is not None else function.__name__):
            return function(*args, **kwargs)

    return wrapped_function


def rank(tensor) -> int:
    return len(tensor.shape)


@func2namescope
def shape(tensor: tf.Tensor):
    stat_shape = tensor.shape.as_list()
    dyn_shape = tf.shape(tensor)
    return [
        dyn_shape[i] if stat_shape[i] is None else stat_shape[i]
        for i in range(rank(tensor))
    ]


def max_pool_2d_flow_map(flow, kernel_stride=2):
    assert rank(flow) == 4  # NHWC
    assert flow.shape[-1] == 2

    if isinstance(kernel_stride, int):
        kernel_stride = (kernel_stride, kernel_stride)

    assert flow.shape[1] % kernel_stride[0] == 0
    assert flow.shape[2] % kernel_stride[1] == 0

    abs_flow = tf.linalg.norm(flow, axis=-1, keepdims=True)

    argmax_idxs = tf.nn.max_pool_with_argmax(
        input=abs_flow,
        ksize=kernel_stride,
        strides=kernel_stride,
        padding="VALID",
        data_format="NHWC",
        output_dtype=tf.dtypes.int64,
        include_batch_in_index=False,
    )[1][..., 0]

    assert rank(argmax_idxs) == 3
    B = tf.shape(flow)[0]
    max_pool_flow = tf.gather(tf.reshape(flow, [B, -1, 2]), argmax_idxs, batch_dims=1)
    assert rank(max_pool_flow) == 4
    assert max_pool_flow.shape[-1] == 2
    assert max_pool_flow.shape[1] == flow.shape[1] // kernel_stride[0]
    assert max_pool_flow.shape[2] == flow.shape[2] // kernel_stride[1]
    return max_pool_flow
