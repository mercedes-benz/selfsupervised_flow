# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT

import tensorflow as tf

from unsup_flow.tf import shape


def upflow_n(flow, n=8, mode="bilinear"):
    _, h, w, _ = shape(flow)
    new_size = (n * h, n * w)
    return n * tf.image.resize(flow, new_size, mode)


def uplogits_n(logits, n=8, mode="bilinear"):
    _, h, w, _ = shape(logits)
    new_size = (n * h, n * w)
    return tf.image.resize(logits, new_size, mode)
