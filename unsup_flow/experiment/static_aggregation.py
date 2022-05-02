#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import numpy as np
import tensorflow as tf

from unsup_flow.losses.static_points_loss import padded2ragged
from unsup_flow.losses.weighted_pc_alignment import weighted_pc_alignment
from unsup_flow.tf import cast32, cast64, shape


def compute_static_aggregated_flow(
    static_flow,
    staticness,
    pc,
    pointwise_voxel_coordinates_fs,
    pointwise_valid_mask,
    voxel_center_metric_coordinates,
    use_eps_for_weighted_pc_alignment=False,
):
    with tf.name_scope("static_aggregated_flow"):
        assert len(static_flow.shape) == 4
        assert static_flow.shape[-1] == 2
        static_3d_flow_grid = tf.concat(
            [static_flow, tf.zeros_like(static_flow[..., :1])],
            axis=-1,
        )

        tf.Assert(
            tf.reduce_all(pointwise_voxel_coordinates_fs >= 0),
            data=["negative pixel coordinates found"],
        )
        tf.Assert(
            tf.reduce_all(
                pointwise_voxel_coordinates_fs < tf.shape(static_3d_flow_grid)[1:3]
            ),
            data=["too large pixel coordinates found"],
        )
        pointwise_flow = grid_flow_to_pointwise_flow(
            static_3d_flow_grid,
            pointwise_voxel_coordinates_fs,
            pointwise_valid_mask,
        )

        voxel_center_metric_coordinates_f32 = voxel_center_metric_coordinates.astype(
            np.float32
        )
        pc0_grid = tf.constant(
            np.concatenate(
                [
                    voxel_center_metric_coordinates_f32,
                    np.zeros_like(voxel_center_metric_coordinates_f32[..., :1]),
                ],
                axis=-1,
            )
        )
        assert pc0_grid.shape == static_3d_flow_grid.shape[1:]
        grid_shape = shape(static_3d_flow_grid)
        batched_pc0_grid = tf.broadcast_to(pc0_grid, grid_shape)

        pointwise_staticness = tf.gather_nd(
            staticness, pointwise_voxel_coordinates_fs, batch_dims=1
        )
        pointwise_staticness = tf.where(
            pointwise_valid_mask,
            pointwise_staticness,
            np.nan * tf.ones_like(pointwise_staticness),
        )

        pc_ragged = padded2ragged(pc)
        pc_flow_ragged = padded2ragged(pc + pointwise_flow)
        weights_ragged = padded2ragged(pointwise_staticness)

        trafo, not_enough_points = weighted_pc_alignment(
            pc_ragged,
            pc_flow_ragged,
            weights_ragged,
            use_eps_for_weighted_pc_alignment,
        )

        static_aggr_flow = cast32(
            tf.einsum(
                "bij,bhwj->bhwi",
                trafo - tf.eye(4, batch_shape=[grid_shape[0]], dtype=tf.float64),
                cast64(
                    tf.concat(
                        [
                            batched_pc0_grid,
                            tf.ones_like(batched_pc0_grid[..., 0][..., None]),
                        ],
                        axis=-1,
                    )
                ),
            )
        )[..., 0:2]

        return static_aggr_flow, trafo, not_enough_points


def grid_flow_to_pointwise_flow(
    grid_flow_3d,
    pointwise_voxel_coordinates_fs,
    pointwise_valid_mask,
):
    pointwise_flow = tf.gather_nd(
        grid_flow_3d, pointwise_voxel_coordinates_fs, batch_dims=1
    )
    pointwise_flow = tf.where(
        tf.tile(
            pointwise_valid_mask[..., None],
            [1, 1, pointwise_flow.shape[-1]],
        ),
        pointwise_flow,
        np.nan * tf.ones_like(pointwise_flow),
    )
    return pointwise_flow
