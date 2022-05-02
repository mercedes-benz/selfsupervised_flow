# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT

import tensorflow as tf

from unsup_flow.tf import func2namescope

from .weighted_pc_alignment import weighted_pc_alignment  # noqa: F401


@func2namescope
def huber_delta(*, err=None, err_sqr=None, delta: float, mode: str = "large_grad_1"):
    assert mode in {"large_grad_1", "small_err_sqr"}

    if delta == 0.0:
        assert mode == "large_grad_1"
        if err is None:
            assert err_sqr is not None
            nonzero_mask_gradient_safe = ~tf.equal(err_sqr, 0.0)
            return tf.sqrt(
                tf.where(nonzero_mask_gradient_safe, err_sqr, tf.ones_like(err_sqr))
            ) * tf.cast(nonzero_mask_gradient_safe, tf.float32)
        else:
            assert err_sqr is None
            return tf.abs(err)

    assert delta > 0.0
    if err is None:
        assert err_sqr is not None
    else:
        assert err_sqr is None
        err_sqr = tf.square(err)

    if mode == "large_grad_1":
        delta_tensor = (
            tf.minimum(err_sqr, delta ** 2) / (2.0 * delta)
            + tf.sqrt(tf.maximum(err_sqr, delta ** 2))
            - delta
        )
    elif mode == "small_err_sqr":
        delta_tensor = (
            tf.minimum(err_sqr, delta ** 2)
            + tf.sqrt(tf.maximum(err_sqr, delta ** 2)) * (2 * delta)
            - 2 * delta ** 2
        )
    else:
        raise ValueError("Unknown huber mode %s" % mode)
    return delta_tensor


@func2namescope
def squared_sum(delta, axis: int = -1):
    return tf.reduce_sum(tf.square(delta), axis=axis)
