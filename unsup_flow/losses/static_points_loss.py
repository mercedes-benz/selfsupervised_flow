#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import numpy as np
import tensorflow as tf

from unsup_flow.tf import cast32, cast64, func2namescope


def ragsub(x, y):
    return tf.ragged.map_flat_values(tf.math.subtract, x, y)


@func2namescope
def get_defined_shape(tensor):
    dyn = tf.shape(tensor)
    stat = tensor.shape.as_list()
    return [s if s is not None else dyn[i] for i, s in enumerate(stat)]


@func2namescope
def padded2ragged(tensor, ragged_dim=1, pad_val=np.nan):
    # NOTE: from_uniform_row_length only exists for tf>=2.1.0
    # so for tf<=2.0.0 we assert that ragged_dim is 1
    assert ragged_dim == 1
    if np.isnan(pad_val):  # bc NaN != NaN
        pad_mask = tf.math.is_nan(tensor)
    else:
        pad_mask = tf.equal(tensor, pad_val)
    s = get_defined_shape(tensor)
    if ragged_dim < 0:
        ragged_dim = len(s) + ragged_dim
    flat_shape = s
    for _ in range(ragged_dim - 1):
        flat_shape = [flat_shape[0] * flat_shape[1], *flat_shape[2:]]
    reshaped_pad_mask = tf.reshape(pad_mask, flat_shape)
    flat_shape = [flat_shape[0] * flat_shape[1], *flat_shape[2:]]
    result_tensor = tf.reshape(tensor, flat_shape)
    row_mask = tf.reduce_all(
        reshaped_pad_mask, axis=list(range(2, 1 + len(flat_shape)))
    )

    tf.debugging.assert_equal(
        tf.reduce_any(reshaped_pad_mask, axis=list(range(2, 1 + len(flat_shape)))),
        row_mask,
        message="padding was inconsistent with entries having some but not all pad values",
    )

    row_mask = tf.logical_not(row_mask)
    result_tensor = tf.boolean_mask(result_tensor, tf.reshape(row_mask, [-1]))
    row_lengths = tf.reduce_sum(tf.cast(row_mask, tf.int64), axis=-1)
    result_tensor = tf.RaggedTensor.from_row_lengths(result_tensor, row_lengths)

    for outer_dim in s[1:ragged_dim][::-1]:
        result_tensor = tf.RaggedTensor.from_uniform_row_length(
            result_tensor, tf.cast(outer_dim, tf.int64)
        )
    return result_tensor


class SymmetricStaticPointsLoss(tf.keras.layers.Layer):
    def call(
        self,
        pc0,
        static_flow_fw,
        static_aggr_trafo_fw,
        staticness_fw,
        pc1,
        static_flow_bw,
        static_aggr_trafo_bw,
        staticness_bw,
        summaries,
    ):
        pc0_ragged = padded2ragged(pc0)
        flow0_ragged = padded2ragged(static_flow_fw)
        w0_ragged = padded2ragged(staticness_fw)
        with tf.control_dependencies(
            [
                tf.assert_equal(
                    tf.shape(pc0_ragged.values)[0], tf.shape(flow0_ragged.values)[0]
                ),
                tf.assert_equal(
                    tf.shape(pc0_ragged.values)[0], tf.shape(w0_ragged.values)[0]
                ),
            ]
        ):
            pc1_ragged = padded2ragged(pc1)
            flow1_ragged = padded2ragged(static_flow_bw)
            w1_ragged = padded2ragged(staticness_bw)
        with tf.control_dependencies(
            [
                tf.assert_equal(
                    tf.shape(pc1_ragged.values)[0], tf.shape(flow1_ragged.values)[0]
                ),
                tf.assert_equal(
                    tf.shape(pc1_ragged.values)[0], tf.shape(w1_ragged.values)[0]
                ),
            ]
        ):
            loss0, self.trafo_fw = static_points_loss(
                pc0_ragged, flow0_ragged, w0_ragged, static_aggr_trafo_fw
            )
            loss1, self.trafo_bw = static_points_loss(
                pc1_ragged, flow1_ragged, w1_ragged, static_aggr_trafo_bw
            )
        for_back_trafo = tf.einsum("boc,bcx->box", self.trafo_bw, self.trafo_fw)
        # for_back_trafo_loss = tf.keras.losses.MeanSquaredError()(
        #     y_true=tf.eye(3, num_columns=4, batch_shape=tf.shape(for_back_trafo)[:1]),
        #     y_pred=for_back_trafo[..., :3, :],
        # )
        for_back_trafo_loss = tf.reduce_mean(
            trafo_distance(
                for_back_trafo
                - tf.eye(4, batch_shape=tf.shape(for_back_trafo)[:1], dtype=tf.float64),
                tf.concat([pc0, pc1], axis=1),
            )
        )
        static_flow_loss = 0.5 * (loss0 + loss1)
        if summaries["metrics_eval"]:
            tf.summary.text("forward_static_trafo", tf.as_string(self.trafo_fw[0]))
            tf.summary.text("backward_static_trafo", tf.as_string(self.trafo_bw[0]))
            tf.summary.text("fw_and_bw_static_trafo", tf.as_string(for_back_trafo[0]))
            tf.summary.scalar("static_flow_loss", static_flow_loss)
            tf.summary.scalar("for_back_trafo_loss", for_back_trafo_loss)
        # if trafo_loss is not None:
        #     trafo_constraint_loss0 = trafo_loss(self.trafo_fw)
        #     trafo_constraint_loss1 = trafo_loss(self.trafo_bw)
        #     return (
        #         static_flow_loss,
        #         for_back_trafo_loss,
        #         0.5 * (trafo_constraint_loss0 + trafo_constraint_loss1),
        #     )
        # else:
        return static_flow_loss, for_back_trafo_loss


@func2namescope
def trafo_distance(delta_trafos: tf.Tensor, points: tf.Tensor):
    dim = delta_trafos.shape[-1] - 1
    assert delta_trafos.shape[-2] == dim + 1
    assert delta_trafos.dtype == tf.float64
    assert points.shape[-1] == dim
    assert len(points.shape) == 3
    assert points.dtype == tf.float32
    points = tf.stop_gradient(points)
    p_mask = tf.logical_not(tf.reduce_all(tf.math.is_nan(points), axis=-1))
    count = tf.reduce_sum(tf.cast(p_mask, tf.int32), axis=-1)
    points_h = tf.concat([points, tf.ones_like(points[..., :1])], axis=-1)
    points_h = tf.where(
        tf.tile(p_mask[..., None], [1, 1, 4]), points_h, tf.zeros_like(points_h)
    )
    delta_points = cast32(
        tf.einsum("b...ij,bkj->b...ki", delta_trafos[..., :3, :], cast64(points_h))
    )
    delta_lengths_sqr = tf.reduce_sum(delta_points ** 2, axis=-1)
    avg_dist_sqr = tf.reduce_sum(delta_lengths_sqr, axis=-1) / tf.cast(
        count, delta_lengths_sqr.dtype
    )
    return avg_dist_sqr


@func2namescope
def static_points_loss(
    pc: tf.RaggedTensor,
    flow: tf.RaggedTensor,
    weights: tf.RaggedTensor,
    trafo: tf.Tensor,
):
    assert pc.shape.as_list()[2:] == [3], pc.shape.as_list()
    assert flow.shape.as_list()[2:] == [3]
    assert len(weights.shape) == 2
    assert trafo.shape.as_list()[1:] == [
        4,
        4,
    ], "The returned trafo from weighted_pc_alignment has only shape %s!" % str(
        trafo.shape.as_list()[1:]
    )
    pc_hom = pc.with_flat_values(
        tf.concat([pc.flat_values, tf.ones_like(pc.flat_values[..., :1])], axis=-1)
    )

    assert pc_hom.shape.as_list()[2:] == [4], pc_hom.shape.as_list()
    assert trafo.shape.as_list()[1:] == [4, 4], trafo.shape.as_list()

    pc_trafo_hom = tf.reduce_sum(
        cast64(pc_hom[:, :, None, :]) * tf.stop_gradient(trafo)[:, None, :, :], axis=-1
    )
    pc_trafo_hom.values.set_shape(pc_trafo_hom.values.shape.as_list()[:-1] + [4])
    assert pc_trafo_hom.shape.as_list()[2:] == [4], pc_trafo_hom.shape.as_list()

    trafo_flow_est = cast32(ragsub(pc_trafo_hom[..., :3], cast64(pc)))
    assert trafo_flow_est.shape.as_list()[2:] == [3], trafo_flow_est.shape.as_list()
    loss = tf.keras.losses.MeanSquaredError()(
        y_true=trafo_flow_est.values, y_pred=flow.values, sample_weight=weights.values
    )
    return loss, trafo
