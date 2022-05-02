#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import tensorflow as tf

from unsup_flow.tf import castf, shape


def scatter_pointwise2bev(
    values, pointwise_voxel_coors, pointwise_mask, grid_size, method="avg"
):
    assert method in {"sum", "avg"}

    assert pointwise_voxel_coors.dtype == tf.int32
    assert pointwise_mask.dtype == tf.bool
    assert len(pointwise_voxel_coors.shape) == 3
    assert len(pointwise_mask.shape) == 2
    assert (
        values.shape[:2].as_list()
        == pointwise_voxel_coors.shape[:2].as_list()
        == pointwise_mask.shape[:2].as_list()
    )
    assert pointwise_voxel_coors.shape[2] == 2

    values_shape = shape(values)
    bs, max_num_points = values_shape[:2]
    for s in values_shape[2:]:
        assert isinstance(s, int)

    tf.Assert(
        tf.reduce_min(pointwise_voxel_coors) >= 0,
        data=["negative voxel coors dont make sense"],
    )
    tf.Assert(
        tf.reduce_all(tf.reduce_max(pointwise_voxel_coors, axis=[0, 1]) < grid_size),
        data=["too large voxel coors for grid size %s" % str(grid_size)],
    )
    tf.Assert(
        tf.reduce_all(
            tf.equal(tf.boolean_mask(pointwise_voxel_coors, ~pointwise_mask), 0)
        ),
        data=[
            "not all masked voxel coors were set too 0",
            tf.shape(tf.boolean_mask(pointwise_voxel_coors, ~pointwise_mask)),
            tf.boolean_mask(pointwise_voxel_coors, ~pointwise_mask),
        ],
    )
    tf.Assert(
        tf.reduce_all(tf.math.is_nan(tf.boolean_mask(values, ~pointwise_mask))),
        data=["not all values outside masked were set to NaN"],
    )

    scatter_nd_inds = tf.concat(
        [
            tf.broadcast_to(
                tf.range(bs)[:, None],
                [bs, max_num_points],
            )[..., None],
            pointwise_voxel_coors,
        ],
        axis=-1,
    )
    target_shape = [bs, *grid_size] + values_shape[2:]

    summed_result = tf.scatter_nd(
        indices=scatter_nd_inds,
        updates=tf.where(pointwise_mask, values, tf.zeros_like(values)),
        shape=target_shape,
    )
    if method == "sum":
        bev_valid_mask = tf.scatter_nd(
            indices=scatter_nd_inds,
            updates=pointwise_mask,
            shape=[bs, *grid_size],
        )
        assert bev_valid_mask.dtype == tf.bool
        return summed_result, bev_valid_mask
    else:
        assert method == "avg"

        assert values.dtype == tf.float32
        counts = tf.scatter_nd(
            indices=scatter_nd_inds,
            updates=tf.cast(pointwise_mask, tf.int32),
            shape=[bs, *grid_size],
        )
        bev_valid_mask = counts > 0
        assert bev_valid_mask.dtype == tf.bool
        avg_nan_padded = tf.where(
            bev_valid_mask, summed_result, tf.fill(target_shape, float("nan"))
        ) / castf(counts)
        return avg_nan_padded, bev_valid_mask
