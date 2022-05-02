#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import functools as ft
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from cfgattrdict import AttrDict
from unsup_flow.knn.plane_geometry import plumb_line_point
from unsup_flow.losses import huber_delta, squared_sum
from unsup_flow.tf import func2namescope, rank
from unsup_flow.tf_user_ops import k_nearest_neighbor_op


class NearestPointLoss:
    def __init__(
        self,
        *args,
        bev_extent: t.Tuple[float, float, float, float],
        L1_delta: float,
        drop_outliers__perc: float,
        fov_mode: str = "ignore_out_fov",
        **kwargs
    ):
        super().__init__()
        assert 0.0 <= drop_outliers__perc < 100.0
        assert fov_mode in {
            "ignore_out_fov",
            "use_nearest",
            "mask_close_fov",
        }

        self.bev_extent = bev_extent
        self.drop_outliers__perc = drop_outliers__perc
        self.huber_loss = ft.partial(huber_delta, delta=L1_delta, mode="large_grad_1")
        self.fov_mode = fov_mode

    def __call__(
        self, *, cloud_a__b, nearest_cloud_a__b, nearest_dist_sqr_a__b, weights__b
    ):
        fov_dist_minx_cloud_a__b = cloud_a__b[..., 0] - self.bev_extent[0]
        fov_dist_miny_cloud_a__b = cloud_a__b[..., 1] - self.bev_extent[1]
        fov_dist_maxx_cloud_a__b = self.bev_extent[2] - cloud_a__b[..., 0]
        fov_dist_maxy_cloud_a__b = self.bev_extent[3] - cloud_a__b[..., 1]
        min_fov_dist_cloud_a__b = tf.reduce_min(
            tf.stack(
                [
                    fov_dist_minx_cloud_a__b,
                    fov_dist_miny_cloud_a__b,
                    fov_dist_maxx_cloud_a__b,
                    fov_dist_maxy_cloud_a__b,
                ],
                axis=-1,
            ),
            axis=-1,
        )

        weights__b = weights__b * tf.cast(
            min_fov_dist_cloud_a__b > 0.0, dtype=weights__b.dtype
        )

        if self.fov_mode == "ignore_out_fov":
            pass
        elif self.fov_mode == "use_nearest":
            nearest_dist_sqr_a__b = tf.minimum(
                nearest_dist_sqr_a__b, tf.square(min_fov_dist_cloud_a__b)
            )
        elif self.fov_mode == "mask_close_fov":
            weights__b = weights__b * tf.cast(
                nearest_dist_sqr_a__b < tf.square(min_fov_dist_cloud_a__b),
                dtype=weights__b.dtype,
            )
        else:
            raise ValueError("Unknown fov_mode: %s" % self.fov_mode)

        loss = self.huber_loss(err_sqr=nearest_dist_sqr_a__b) * weights__b
        if self.drop_outliers__perc > 0.0:
            loss_threshold = tf.map_fn(
                lambda x: tfp.stats.percentile(
                    tf.boolean_mask(x, ~tf.math.is_nan(x)),
                    100.0 - self.drop_outliers__perc,
                    interpolation="higher",
                    preserve_gradients=False,
                ),
                loss,
                dtype=loss.dtype,
                swap_memory=False,
                parallel_iterations=8,
            )
            loss = tf.where(
                loss
                <= loss_threshold[(slice(None),) + tuple([None] * (rank(loss) - 1))],
                loss,
                tf.zeros_like(loss),
            )

        return loss


def normalized_opposite_flow_loss_function(
    *, flow_b_to_a, nearest_flow_a_to_b__b, weights__b
):

    sum_flow = flow_b_to_a + nearest_flow_a_to_b__b
    length_b_to_a = tf.linalg.norm(flow_b_to_a, axis=-1)
    length_b_a_to_b = tf.linalg.norm(nearest_flow_a_to_b__b, axis=-1)
    sum_length = length_b_to_a + length_b_a_to_b
    guarded_sum_length = tf.stop_gradient(
        tf.where(
            tf.equal(sum_length, tf.zeros_like(sum_length)),
            tf.ones_like(sum_length),
            sum_length,
        )
    )
    if tf.__version__[0] == "2":
        loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )(
            y_true=tf.zeros_like(sum_flow),
            y_pred=sum_flow / guarded_sum_length[..., None],
            sample_weight=weights__b,
        )
    else:
        loss = tf.losses.mean_squared_error(
            labels=tf.zeros_like(sum_flow),
            predictions=sum_flow / guarded_sum_length[..., None],
            weights=weights__b,
            reduction=tf.keras.losses.Reduction.NONE,
        )
    return loss


def get_idx_dists_for_knn(ref_pts, query_pts, num_neighbors=1):
    assert rank(ref_pts) == 2
    assert rank(query_pts) == 2
    clean_ref_pts = clean_data_from_nans(ref_pts)
    clean_query_pts = clean_data_from_nans(query_pts)
    num_pad_query = tf.shape(query_pts)[0] - tf.shape(clean_query_pts)[0]

    indices, _ = k_nearest_neighbor_op(clean_ref_pts, clean_query_pts, num_neighbors)
    indices = tf.concat(
        [
            indices,
            tf.ones((num_pad_query, num_neighbors), tf.int32)
            * tf.shape(clean_ref_pts)[0],
        ],
        axis=0,
    )

    return indices


def clean_data_from_nans(data):
    non_nan_mask = tf.logical_not(tf.reduce_all(tf.math.is_nan(data), axis=-1))
    nbr_non_nan_entries = tf.reduce_sum(tf.cast(non_nan_mask, tf.int32), axis=-1)
    non_nan_sequence_mask = tf.sequence_mask(
        nbr_non_nan_entries, maxlen=tf.shape(non_nan_mask)[-1]
    )
    with tf.control_dependencies(
        [tf.debugging.assert_equal(non_nan_mask, non_nan_sequence_mask)]
    ):
        result = tf.boolean_mask(data, non_nan_mask)
    return result


# @tf.function(
#     input_signature=[
#         tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
#         tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
#         tf.TensorSpec(shape=[None, None], dtype=tf.float32),
#         tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
#         tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
#         tf.TensorSpec(shape=[None, None], dtype=tf.float32),
#     ]
# )
@func2namescope
def get_flow_matches_loss(
    cloud_0: tf.Tensor,
    flow_0_to_1: tf.Tensor,
    point_disappears_0_1: tf.Tensor,
    cloud_1: tf.Tensor,
    flow_1_to_0: tf.Tensor,
    point_disappears_1_0: tf.Tensor,
    loss_function,
    opposite_flow_loss_function=normalized_opposite_flow_loss_function,
    summaries=None,
    nearest_dist_mode: str = "point",
):
    with tf.name_scope("fw"):
        (
            forward_loss,
            forward_opposite_flow_loss,
            forward_knn_results,
        ) = compute_flow_loss(
            cloud_1,
            cloud_0,
            flow_1_to_0,
            flow_0_to_1,
            point_disappears_1_0,
            point_disappears_0_1,
            loss_function=loss_function,
            opposite_flow_loss_function=opposite_flow_loss_function,
            summaries=summaries,
            nearest_dist_mode=nearest_dist_mode,
        )

    with tf.name_scope("bw"):
        (
            backward_loss,
            backward_opposite_flow_loss,
            backward_knn_results,
        ) = compute_flow_loss(
            cloud_0,
            cloud_1,
            flow_0_to_1,
            flow_1_to_0,
            point_disappears_0_1,
            point_disappears_1_0,
            loss_function=loss_function,
            opposite_flow_loss_function=opposite_flow_loss_function,
            summaries=summaries,
            nearest_dist_mode=nearest_dist_mode,
        )

    return (
        forward_loss,
        forward_opposite_flow_loss,
        forward_knn_results,
        backward_loss,
        backward_opposite_flow_loss,
        backward_knn_results,
    )


def smoothness_penalty(cloud_a, flow_a_b, num_neighbors_for_smoothness):
    mask = tf.logical_not(tf.reduce_all(tf.math.is_nan(cloud_a), axis=-1))
    # cloud_a, flow_a_b = (
    #     tf.pad(data, [[0, 0], [0, 1], [0, 0]], constant_values=np.nan)
    #     for data in [cloud_a, flow_a_b]
    # )
    nn_indices_a = tf.map_fn(
        lambda x: get_idx_dists_for_knn(
            x[0], x[1], num_neighbors=num_neighbors_for_smoothness + 1
        ),
        [cloud_a, cloud_a],
        dtype=tf.int32,
        swap_memory=True,
        parallel_iterations=8,
    )
    nn_indices_without_query_pt_a = nn_indices_a[..., 1:]

    flow_vecs_of_neighbors_a = tf.gather_nd(
        flow_a_b, nn_indices_without_query_pt_a[..., None], batch_dims=1
    )

    flow_a_b_expanded = tf.expand_dims(flow_a_b, axis=-2)
    mean_squared_l2_norm = tf.reduce_mean(
        squared_sum(flow_a_b_expanded - flow_vecs_of_neighbors_a, axis=-1),
        axis=-1,
    )
    mean_squared_l2_norm = tf.reduce_mean(tf.boolean_mask(mean_squared_l2_norm, mask))
    return mean_squared_l2_norm


def temporal_cls_consistency(
    *, cloud_a, flow_a_b, cloud_b, class_logits_a, class_probs_b, mask_a
):
    nn_indices_at_b__a = tf.map_fn(
        lambda x: get_idx_dists_for_knn(x[0], x[1], num_neighbors=1),
        [cloud_b, cloud_a + flow_a_b],
        dtype=tf.int32,
        swap_memory=True,
        parallel_iterations=8,
    )
    assert nn_indices_at_b__a.shape[-1] == 1
    class_probs_at_b__a = tf.gather_nd(class_probs_b, nn_indices_at_b__a, batch_dims=1)
    masked_class_logits_a = tf.boolean_mask(class_logits_a, mask_a)
    masked_class_probs_at_b__a = tf.boolean_mask(class_probs_at_b__a, mask_a)
    xentropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=masked_class_probs_at_b__a, logits=masked_class_logits_a
    )
    return tf.reduce_mean(xentropy)


def compute_flow_loss(
    cloud_a,
    cloud_b,
    flow_a_to_b,
    flow_b_to_a,
    point_disappears_a_to_b,
    point_disappears_b_to_a,
    loss_function,
    opposite_flow_loss_function=normalized_opposite_flow_loss_function,
    summaries=None,
    nearest_dist_mode: str = "point",
):
    # #region check shapes and dtypes
    assert nearest_dist_mode in {"point", "plane"}
    assert tf.__version__[0] == "2"
    assert rank(cloud_a) == 3
    assert rank(cloud_b) == 3
    assert rank(flow_a_to_b) == 3
    assert rank(flow_b_to_a) == 3
    assert rank(point_disappears_a_to_b) == 2
    assert rank(point_disappears_b_to_a) == 2
    assert cloud_a.shape[-1] == 3
    assert cloud_b.shape[-1] == 3
    assert flow_a_to_b.shape[-1] == 3
    assert flow_b_to_a.shape[-1] == 3
    # #endregion check shapes and dtypes

    # #region pad with nan to guarantee for the gather operation that there is at least one padded value
    cloud_a, cloud_b, flow_a_to_b, flow_b_to_a = (
        tf.pad(data, [[0, 0], [0, 1], [0, 0]], constant_values=np.nan)
        for data in [cloud_a, cloud_b, flow_a_to_b, flow_b_to_a]
    )
    point_disappears_a_to_b, point_disappears_b_to_a = (
        tf.pad(data, [[0, 0], [0, 1]], constant_values=np.nan)
        for data in [point_disappears_a_to_b, point_disappears_b_to_a]
    )
    # #endregion pad with nan to guarantee for the gather operation that there is at least one padded value

    # notation: X_a__b, where X is name/meaning, a denotes the time idx for the values, but b denotes the set of support points
    # especially: X_a__b has as many entries as cloud_b, not cloud_a

    cloud_a__b = cloud_b + flow_b_to_a
    plane_indices_into_a__b = tf.map_fn(
        lambda x: get_idx_dists_for_knn(x[0], x[1], num_neighbors=3),
        [cloud_a, cloud_a__b],
        dtype=tf.int32,
        swap_memory=True,
        parallel_iterations=8,  # NOTE: use a fitting value if batch size changes dramatically
    )
    plane_nearest_cloud_a__b = tf.gather(cloud_a, plane_indices_into_a__b, batch_dims=1)
    indices_into_a__b = plane_indices_into_a__b[..., 0]
    nearest_cloud_a__b = plane_nearest_cloud_a__b[..., 0, :]
    plumb_line_point_cloud_a__b = plumb_line_point(cloud_a__b, plane_nearest_cloud_a__b)

    nearest_dist_sqr_a__b = squared_sum(nearest_cloud_a__b - cloud_a__b, axis=-1)
    plumb_line_point_dist_sqr_a__b = squared_sum(
        plumb_line_point_cloud_a__b - cloud_a__b, axis=-1
    )

    rel_err = 1e-5
    abs_err = 1e-5
    tf.Assert(
        tf.reduce_all(
            (
                nearest_dist_sqr_a__b * (1.0 + rel_err) + abs_err
                >= plumb_line_point_dist_sqr_a__b
            )
            | tf.math.is_nan(nearest_dist_sqr_a__b)
        ),
        data=[
            tf.math.count_nonzero(
                ~(
                    (nearest_dist_sqr_a__b >= plumb_line_point_dist_sqr_a__b)
                    | tf.math.is_nan(nearest_dist_sqr_a__b)
                )
            ),
            tf.boolean_mask(
                nearest_dist_sqr_a__b,
                ~(
                    (nearest_dist_sqr_a__b >= plumb_line_point_dist_sqr_a__b)
                    | tf.math.is_nan(nearest_dist_sqr_a__b)
                ),
            ),
            tf.boolean_mask(
                plumb_line_point_dist_sqr_a__b,
                ~(
                    (nearest_dist_sqr_a__b >= plumb_line_point_dist_sqr_a__b)
                    | tf.math.is_nan(nearest_dist_sqr_a__b)
                ),
            ),
        ],
    )

    if nearest_dist_mode == "point":
        selected_nearest_cloud_a__b = {
            "points": nearest_cloud_a__b,
            "dists_sqr": nearest_dist_sqr_a__b,
        }
    else:
        assert nearest_dist_mode == "plane"
        selected_nearest_cloud_a__b = {
            "points": plumb_line_point_cloud_a__b,
            "dists_sqr": plumb_line_point_dist_sqr_a__b,
        }

    nearest_point_disappears_a_to_b__b = tf.gather(
        point_disappears_a_to_b, indices_into_a__b, batch_dims=1
    )

    nearest_flow_a_to_b__b = tf.gather(flow_a_to_b, indices_into_a__b, batch_dims=1)

    weights__b = (1.0 - nearest_point_disappears_a_to_b__b) * (
        1.0 - point_disappears_b_to_a
    )
    if summaries is not None and summaries["imgs_eval"]:
        draw_correlation_hist(
            tf.sqrt(selected_nearest_cloud_a__b["dists_sqr"]),
            point_disappears_b_to_a,
            max_outputs=1,
            summaries=summaries,
        )

    loss = loss_function(
        cloud_a__b=cloud_a__b,
        nearest_cloud_a__b=selected_nearest_cloud_a__b["points"],
        nearest_dist_sqr_a__b=selected_nearest_cloud_a__b["dists_sqr"],
        weights__b=weights__b,
    )
    opposite_flow_loss = opposite_flow_loss_function(
        flow_b_to_a=flow_b_to_a,
        nearest_flow_a_to_b__b=nearest_flow_a_to_b__b,
        weights__b=weights__b,
    )
    # finally remove padding from above
    return (
        loss[:, :-1],
        opposite_flow_loss[:, :-1],
        AttrDict(
            nn_plane_indices=plane_indices_into_a__b[:, :-1],
            nearest_dist_sqr=nearest_dist_sqr_a__b[:, :-1],
            nearest_dist=tf.sqrt(nearest_dist_sqr_a__b[:, :-1]),
            plumb_line_dist_sqr=plumb_line_point_dist_sqr_a__b[:, :-1],
            plumb_line_dist=tf.sqrt(plumb_line_point_dist_sqr_a__b[:, :-1]),
        ),
    )


def draw_with_mpl(res_flow_b, point_disappears_b_to_a, fig_size, fig_dpi):
    rect_hist = [0.08, 0.08, 0.72, 0.72]
    rect_histx = [0.08, 0.82, 0.72, 0.17]

    fig = Figure(
        figsize=(fig_size, fig_size), dpi=fig_dpi, facecolor="w", edgecolor="k"
    )

    axHist2D = fig.add_axes(rect_hist)
    axHistx = fig.add_axes(rect_histx)

    axHistx.set_xticks([], [])

    # the hist plot:
    nbins = 10
    bx = np.linspace(res_flow_b.min(), res_flow_b.max(), num=nbins + 1)
    by = np.linspace(0.0, 1.0, num=nbins + 1)

    h, bx, by = np.histogram2d(res_flow_b, point_disappears_b_to_a, bins=[bx, by])
    h /= np.maximum(1.0, h.sum(axis=-1, keepdims=True))
    axHist2D.imshow(
        h.T, aspect="auto", origin="lower", extent=[bx[0], bx[-1], by[0], by[-1]]
    )

    # now determine nice limits by hand:
    axHist2D.set_xlim((bx[0], bx[-1]))
    axHist2D.set_ylim((by[0], by[-1]))
    axHist2D.set_xlabel("nearest neighbor distance", fontsize=fig_size)
    axHist2D.set_ylabel("disappearing probability", fontsize=fig_size)

    axHistx.hist(res_flow_b, bins=bx)
    axHistx.set_yscale("log")
    axHistx.set_xlim(axHist2D.get_xlim())
    axHistx.set_ylabel("counts")

    canvas = FigureCanvasAgg(fig)
    canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape((fig_dpi * fig_size, fig_dpi * fig_size, 3))[
        :, : int(fig_dpi * fig_size / 1.2)
    ]

    fig.clear()
    plt.close(fig)

    return image


def wrap_draw_with_mpl(res_flow_b, point_disappears_b_to_a, fig_size, fig_dpi):
    res_flow_b, point_disappears_b_to_a = (
        x.numpy() if isinstance(x, tf.Tensor) else x
        for x in [res_flow_b, point_disappears_b_to_a]
    )
    if len(res_flow_b.shape) > 1:
        result = []
        for k in range(res_flow_b.shape[0]):
            result.append(
                wrap_draw_with_mpl(
                    res_flow_b[k],
                    point_disappears_b_to_a[k],
                    fig_size=fig_size,
                    fig_dpi=fig_dpi,
                )
            )
        return np.stack(result, axis=0)
    else:
        mask = np.logical_not(np.isnan(res_flow_b))
        res_flow_b = res_flow_b[mask]
        point_disappears_b_to_a = point_disappears_b_to_a[mask]
        return draw_with_mpl(
            res_flow_b, point_disappears_b_to_a, fig_size=fig_size, fig_dpi=fig_dpi
        )


def draw_correlation_hist(
    res_flow_b,
    point_disappears_b_to_a,
    fig_size=8,
    fig_dpi=120,
    max_outputs=3,
    summaries=None,
):
    if summaries is not None:
        with tf.name_scope("__dummy_scope_name__") as scope:
            scope = "/".join(scope.split("/")[:-2]) + "/"
            if summaries["imgs_eval"]:
                with tf.name_scope(""):
                    with tf.name_scope(scope):
                        hist_images = tf.py_function(
                            ft.partial(
                                wrap_draw_with_mpl, fig_size=fig_size, fig_dpi=fig_dpi
                            ),
                            [res_flow_b, point_disappears_b_to_a],
                            Tout=tf.uint8,
                        )
                        with tf.device("/cpu"):
                            tf.summary.image(
                                "disappearing_distance_corr",
                                hist_images,
                                max_outputs=max_outputs,
                            )
