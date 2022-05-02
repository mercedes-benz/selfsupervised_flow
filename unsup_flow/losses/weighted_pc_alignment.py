#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import tensorflow as tf

from unsup_flow.tf import cast64, castf, func2namescope


def ragmul(x, y):
    return tf.ragged.map_flat_values(tf.math.multiply, x, y)


@func2namescope
def weighted_pc_alignment(
    cloud_t0: tf.RaggedTensor,
    cloud_t1: tf.RaggedTensor,
    weights: tf.RaggedTensor,
    use_epsilon_on_weights=False,
):
    dims = 3
    assert cloud_t0.shape.as_list()[2:] == [dims]
    assert cloud_t1.shape.as_list()[2:] == [dims]
    assert len(weights.shape) == 2

    tf.Assert(tf.reduce_all(weights.values >= 0.0), data=["negative weights found"])
    if use_epsilon_on_weights:
        weights += tf.keras.backend.epsilon()
        count_nonzero_weighted_points = tf.reduce_sum(
            tf.cast(weights > 0.0, dtype=tf.int32), axis=-1
        )
        not_enough_points = count_nonzero_weighted_points < 3
    else:
        count_nonzero_weighted_points = tf.reduce_sum(
            tf.cast(weights > 0.0, dtype=tf.int32), axis=-1
        )
        not_enough_points = count_nonzero_weighted_points < 3
        eps = castf(not_enough_points) * tf.keras.backend.epsilon()
        weights += eps[:, None]

    # m, n = cloud_t0.shape  # m = dims, n num points
    cum_wts = tf.reduce_sum(weights, axis=-1)

    X_wtd = ragmul(cloud_t0, weights[..., None])
    Y_wtd = ragmul(cloud_t1, weights[..., None])

    mx_wtd = tf.reduce_sum(X_wtd, axis=1) / cum_wts[:, None]
    my_wtd = tf.reduce_sum(Y_wtd, axis=1) / cum_wts[:, None]
    Xc = cloud_t0 - mx_wtd[:, None, :]
    Yc = cloud_t1 - my_wtd[:, None, :]
    Xc.values.set_shape(Xc.values.shape.as_list()[:-1] + mx_wtd.shape[-1:])
    Yc.values.set_shape(Yc.values.shape.as_list()[:-1] + my_wtd.shape[-1:])

    # sx = np.mean(np.sum(Xc * Yc, 0))

    # Sxy_wtd = (
    #     tf.einsum("bnc,bnd->bcd", Yc * weights[..., None], Xc) / cum_wts[:, None, None]
    # )
    Sxy_wtd = (
        tf.reduce_sum(
            ragmul(ragmul(Yc, weights[..., None])[:, :, :, None], Xc[:, :, None, :]),
            axis=1,
        )
        / cum_wts[:, None, None]
    )

    D, U, V = tf.linalg.svd(cast64(Sxy_wtd), full_matrices=True, compute_uv=True)
    R = tf.einsum("boc,bic->boi", U, V)

    # c = np.trace(np.dot(np.diag(D), S)) / sx
    # c = 1.0
    t = cast64(my_wtd) - tf.einsum("boc,bc->bo", R, cast64(mx_wtd))
    # t = my_wtd - c * np.dot(R, mx_wtd)

    # T = tf.eye(dims + 1, batch_shape=tf.shape(weights)[:1])
    R = tf.concat([R, tf.zeros_like(R[:, :1, :])], axis=1)
    t = tf.concat([t, tf.ones_like(t[:, :1])], axis=-1)
    T = tf.concat([R, t[:, :, None]], axis=-1)

    assert T.dtype == tf.float64

    return T, not_enough_points  # R, t
