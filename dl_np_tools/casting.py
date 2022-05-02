#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


from .mod_available import np_available, tf_available, torch_available

try:
    import numpy as np
except ModuleNotFoundError:
    pass
try:
    import torch
except ModuleNotFoundError:
    pass
try:
    import tensorflow as tf
except ModuleNotFoundError:
    pass


def castf(array):
    if np_available() and isinstance(array, np.ndarray):
        return _np_castf(array)

    if tf_available() and isinstance(array, tf.Tensor):
        return _tf_castf(array)

    if torch_available() and isinstance(array, torch.Tensor):
        return _torch_castf(array)

    raise ValueError("Unknown array type: %s" % str(type(array)))


def _tf_castf(tensor):
    assert tensor.dtype in {tf.float32, tf.float64, tf.int32, tf.int64, tf.bool}
    if tensor.dtype not in {tf.float32, tf.float64}:
        tensor = tf.cast(tensor, tf.float32)
    assert tensor.dtype in {tf.float32, tf.float64}
    return tensor


def cast32(array):
    if np_available() and isinstance(array, np.ndarray):
        return _np_cast32(array)

    if tf_available() and isinstance(array, tf.Tensor):
        return _tf_cast32(array)

    if torch_available() and isinstance(array, torch.Tensor):
        return _torch_cast32(array)

    raise ValueError("Unknown array type: %s" % str(type(array)))


def _tf_cast32(tensor):
    if tensor.dtype in {tf.float32, tf.int32}:
        return tensor
    if tensor.dtype == tf.float64:
        return tf.cast(tensor, tf.float32)
    if tensor.dtype == tf.int64:
        return tf.cast(tensor, tf.int32)
    assert tensor.dtype in {tf.float32, tf.int32}, tensor.dtype
    return tensor


def cast64(array):
    if np_available() and isinstance(array, np.ndarray):
        return _np_cast64(array)

    if tf_available() and isinstance(array, tf.Tensor):
        return _tf_cast64(array)

    if torch_available() and isinstance(array, torch.Tensor):
        return _torch_cast64(array)

    raise ValueError("Unknown array type: %s" % str(type(array)))


def _tf_cast64(tensor):
    if tensor.dtype in {tf.float64, tf.int64}:
        return tensor
    if tensor.dtype == tf.float32:
        return tf.cast(tensor, tf.float64)
    if tensor.dtype == tf.int32:
        return tf.cast(tensor, tf.int64)
    assert tensor.dtype in {tf.float64, tf.int64}, tensor.dtype
    return tensor
