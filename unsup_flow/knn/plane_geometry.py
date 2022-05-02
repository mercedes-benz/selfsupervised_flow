#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import tensorflow as tf


def squared_sum(delta, axis: int = -1):
    return tf.reduce_sum(tf.square(delta), axis=axis, keepdims=True)


def dot_product(a, b, axis: int = -1):
    return tf.reduce_sum(a * b, axis=axis, keepdims=True)


def plumb_line_point(point, threepointplane, dim: int = 3):
    # computes plumb line point onto plane defined by three points
    # in case of degenerate plane cases (line or single point) the plumb line is constructed
    # w.r.t. those degenerate plane objects as a natural extension
    # basically selecting from all possible planes those that maximize the plumb line distance

    assert point.shape[-1] == dim
    assert threepointplane.shape[-2:] == (dim, dim)
    batch_shape = point.shape[:-1].as_list()
    assert batch_shape == threepointplane.shape[:-2].as_list()
    return (
        plumb_line_point_helper(
            point - threepointplane[..., 0, :],
            threepointplane[..., 1:, :] - threepointplane[..., :1, :],
            dim=dim,
        )
        + threepointplane[..., 0, :]
    )


def angle_vector_plane(vector, threepointplane, dim: int = 3):
    assert vector.shape[-1] == dim
    assert threepointplane.shape[-2:] == (dim, dim)
    batch_shape = vector.shape[:-1].as_list()
    assert batch_shape == threepointplane.shape[:-2].as_list()
    plane_normal = plane_normal_helper(
        threepointplane[..., 1:, :] - threepointplane[..., :1, :],
        disambiguate=vector,
        dim=dim,
    )
    dot_prod = dot_product(vector, plane_normal)
    norm_factor = tf.sqrt(squared_sum(plane_normal) * squared_sum(vector))
    return tf.squeeze(tf.abs(tf.asin(dot_prod / norm_factor)), axis=-1)


def plumb_line_point_helper(point, origin_twopointplane, dim: int = 3):
    # computes plumb line point onto plane defined by origin and two span vectors
    # in case of degenerate plane cases (line or single point) the plumb line is constructed
    # w.r.t. those degenerate plane objects as a natural extension
    # basically selecting from all possible planes those that maximize the plumb line distance

    assert point.shape[-1] == dim
    assert origin_twopointplane.shape[-2:] == (dim - 1, dim)
    batch_shape = point.shape[:-1].as_list()
    assert batch_shape == origin_twopointplane.shape[:-2].as_list()
    plane_normal = plane_normal_helper(
        origin_twopointplane, disambiguate=point, dim=dim
    )
    norm_sqr_plane_normal = squared_sum(plane_normal)
    perp_point_comp = (
        plane_normal
        * dot_product(plane_normal, point)
        / tf.where(
            norm_sqr_plane_normal > 0,
            norm_sqr_plane_normal,
            tf.ones_like(norm_sqr_plane_normal),
        )
    )
    return point - perp_point_comp


def plane_normal_helper(origin_twopointplane, disambiguate=None, dim: int = 3):
    # dim=3:
    #   returns normal vector for each set of two 3d span vectors starting at origin
    # dim=2:
    #   returns normal vector to single 2d span vector starting at origin
    # disambiguate may specify a 2/3D vector to which the normal should point
    # in case of degenerate cases (span vectors lie on line in 3D or span vectors are zero in 2/3D)
    # if not specified, those cases return zero vector
    # in case disambiguate vector lies on line or origin itself, still returns zero vector

    assert origin_twopointplane.shape[-2:] == (dim - 1, dim)
    if disambiguate is not None:
        assert disambiguate.shape[-1] == dim
        batch_shape = origin_twopointplane.shape[:-2].as_list()
        assert batch_shape == disambiguate.shape[:-1].as_list()

    if dim == 2:
        result = tf.stack(
            [origin_twopointplane[..., 0, 1], -origin_twopointplane[..., 0, 0]], axis=-1
        )
        if disambiguate is not None:
            norm_res_sqr = squared_sum(result)
            result = tf.where(tf.equal(norm_res_sqr, 0.0), disambiguate, result)
        return result

    elif dim == 3:
        result = tf.linalg.cross(
            origin_twopointplane[..., 0, :], origin_twopointplane[..., 1, :]
        )
        if disambiguate is not None:
            norm_res_sqr = squared_sum(result)
            origin_twopointplane_norm_sqr = squared_sum(origin_twopointplane)
            line_vec = tf.where(
                (
                    origin_twopointplane_norm_sqr[..., 0, :]
                    < origin_twopointplane_norm_sqr[..., 1, :]
                ),
                origin_twopointplane[..., 1, :],
                origin_twopointplane[..., 0, :],
            )
            line_vec_norm_sqr = tf.reduce_max(origin_twopointplane_norm_sqr, axis=-2)
            cur_disambiguate = disambiguate - line_vec * dot_product(
                disambiguate, line_vec
            ) / tf.where(
                line_vec_norm_sqr > 0,
                line_vec_norm_sqr,
                tf.ones_like(line_vec_norm_sqr),
            )
            result = tf.where(tf.equal(norm_res_sqr, 0.0), cur_disambiguate, result)
        return result

    else:
        raise ValueError("dimension %d not supported" % dim)


if __name__ == "__main__":
    import numpy as np

    plane = tf.constant([[1.0, 1.0, 1.0], [2.0, 1.0, 2.0], [1.0, 2.0, 2.0]])
    point = tf.constant([1.0, 1.0, 2.0])
    assert np.allclose(plumb_line_point(point, plane).numpy(), [4 / 3, 4 / 3, 5 / 3])
    assert np.allclose(angle_vector_plane(tf.constant([1.0, 1.0, 2.0]), plane), 0.0)

    plane = tf.constant([[-1.0, -1.0], [1.0, 0.0]])
    point = tf.constant([-1.0, 1.5])
    assert np.allclose(plumb_line_point(point, plane, dim=2), [0, -1 / 2])
    assert np.allclose(
        angle_vector_plane(point - plane[0], plane, dim=2), np.arctan(2.0)
    )

    # edge case three points lie on line
    plane = tf.constant([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
    point = tf.constant([0.0, 1.0, 1.0])
    assert np.allclose(plumb_line_point(point, plane).numpy(), [0.5, 0.5, 0.0])
    assert np.allclose(
        angle_vector_plane(point, plane), np.arctan(np.sqrt(6.0) / np.sqrt(2.0))
    )

    # edge case three points are the same
    plane = tf.constant([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    point = tf.constant([0.0, 1.0, 1.0])
    assert np.allclose(plumb_line_point(point, plane).numpy(), [1.0, 1.0, 0.0])
    assert np.allclose(angle_vector_plane(point, plane), np.pi / 2.0)
