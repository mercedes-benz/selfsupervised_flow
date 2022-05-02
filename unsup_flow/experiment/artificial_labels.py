#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT

from typing import Dict

import numpy as np
import tensorflow as tf

from cfgattrdict import ConfigAttrDict
from unsup_flow.tf import castf, shape
from unsup_flow.tf.numerical_stability import numerically_stable_quotient_lin_comb_exps


def gauss_static_prob(knn_static, knn_dynamic, static_w, dynamic_w):
    """
    is_static_odds = prob_static / prob_dynamic
    unscaled_prob_ = exp(-0.5 (knn_/sigma_)**2)
    =>
    is_static_odds = exp(0.5 * ( (knn_dynamic/sigma_dynamic)**2 - (knn_static/sigma_static)**2 ))
    w.r.t norm_prob_static+norm_prob_dynamic=1 =>
    1/is_static_odds + 1 = 1 / norm_prob_static
    norm_prob_static = is_static_odds / (is_static_odds + 1)
    """
    is_static_odds_exp = 0.5 * (
        (knn_dynamic / dynamic_w) ** 2 - (knn_static / static_w) ** 2
    )
    return numerically_stable_quotient_lin_comb_exps(
        num_exps=[is_static_odds_exp],
        num_weights=[1.0],
        denom_exps=[
            is_static_odds_exp,
            tf.zeros_like(is_static_odds_exp),
        ],
        denom_weights=[1.0, 1.0],
    )


def discrepancy_labels(
    *,
    knn_results,
    static_flow_key: str,
    knn_dist_key: str,
    discrepancy_threshold: float,
):
    is_static_artificial_label_fw = tf.where(
        tf.abs(
            knn_results[static_flow_key]["fw_knn"][knn_dist_key]
            - knn_results["dynamic"]["fw_knn"][knn_dist_key]
        )
        > discrepancy_threshold,
        castf(
            tf.less_equal(
                knn_results[static_flow_key]["fw_knn"][knn_dist_key],
                knn_results["dynamic"]["fw_knn"][knn_dist_key],
            )
        ),
        tf.fill(
            shape(knn_results[static_flow_key]["fw_knn"][knn_dist_key]),
            np.float32(0.5),
        ),
    )
    is_static_artificial_label_bw = tf.where(
        tf.abs(
            knn_results[static_flow_key]["bw_knn"][knn_dist_key]
            - knn_results["dynamic"]["bw_knn"][knn_dist_key]
        )
        > discrepancy_threshold,
        castf(
            tf.less_equal(
                knn_results[static_flow_key]["bw_knn"][knn_dist_key],
                knn_results["dynamic"]["bw_knn"][knn_dist_key],
            )
        ),
        tf.fill(
            shape(knn_results[static_flow_key]["bw_knn"][knn_dist_key]),
            np.float32(0.5),
        ),
    )
    # #region soft labels to weights
    # instead of smooth labels, make labels hard and transfer smoothness
    # to weights
    artificial_label_weights = {
        "forward": tf.abs(2.0 * is_static_artificial_label_fw - 1.0),
        "backward": tf.abs(2.0 * is_static_artificial_label_bw - 1.0),
    }
    is_static_artificial_label_fw = castf(is_static_artificial_label_fw >= 0.5)
    is_static_artificial_label_bw = castf(is_static_artificial_label_bw >= 0.5)
    # #endregion soft labels to weights

    return (
        is_static_artificial_label_fw,
        is_static_artificial_label_bw,
        artificial_label_weights,
    )


def constant_labels(
    *,
    knn_results,
    static_flow_key: str,
    knn_dist_sqr_key: str,
):

    is_static_artificial_label_fw = castf(
        tf.less_equal(
            knn_results[static_flow_key]["fw_knn"][knn_dist_sqr_key],
            knn_results["dynamic"]["fw_knn"][knn_dist_sqr_key],
        )
    )
    is_static_artificial_label_bw = castf(
        tf.less_equal(
            knn_results[static_flow_key]["bw_knn"][knn_dist_sqr_key],
            knn_results["dynamic"]["bw_knn"][knn_dist_sqr_key],
        )
    )
    artificial_label_weights = {
        "forward": tf.ones_like(is_static_artificial_label_fw),
        "backward": tf.ones_like(is_static_artificial_label_bw),
    }

    return (
        is_static_artificial_label_fw,
        is_static_artificial_label_bw,
        artificial_label_weights,
    )


def big_mixture_labels(
    *,
    prediction_fw,
    prediction_bw,
    knn_results,
    static_flow_key: str,
    knn_dist_key: str,
    mixture_dist: float,
):
    assert static_flow_key in {"static", "static_aggr"}, static_flow_key
    dyn_stat_flow_dist_fw = tf.linalg.norm(
        prediction_fw["dynamic_flow"] - prediction_fw["%s_flow" % static_flow_key],
        axis=-1,
    )
    dyn_stat_flow_dist_bw = tf.linalg.norm(
        prediction_bw["dynamic_flow"] - prediction_bw["%s_flow" % static_flow_key],
        axis=-1,
    )

    knn_sign_delta_dists_fw = (
        knn_results[static_flow_key]["fw_knn"][knn_dist_key]
        - knn_results["dynamic"]["fw_knn"][knn_dist_key]
    )
    knn_sign_delta_dists_bw = (
        knn_results[static_flow_key]["bw_knn"][knn_dist_key]
        - knn_results["dynamic"]["bw_knn"][knn_dist_key]
    )

    knn_delta_dists_fw = tf.abs(knn_sign_delta_dists_fw)
    knn_delta_dists_bw = tf.abs(knn_sign_delta_dists_bw)

    prob_flow_aggreement_fw = tf.exp(-dyn_stat_flow_dist_fw / mixture_dist)
    prob_flow_aggreement_bw = tf.exp(-dyn_stat_flow_dist_bw / mixture_dist)

    min_weight_fw = prob_flow_aggreement_fw
    min_weight_bw = prob_flow_aggreement_bw

    weight_fw = 1.0 - tf.exp(-knn_delta_dists_fw / mixture_dist)
    weight_bw = 1.0 - tf.exp(-knn_delta_dists_bw / mixture_dist)

    weight_fw = min_weight_fw + weight_fw * (1.0 - min_weight_fw)
    weight_bw = min_weight_bw + weight_bw * (1.0 - min_weight_bw)

    prob_static_fw = tf.sigmoid(knn_sign_delta_dists_fw / mixture_dist)
    prob_static_bw = tf.sigmoid(knn_sign_delta_dists_bw / mixture_dist)

    is_static_artificial_label_fw = castf(
        (0.9 * prob_static_fw + 0.1 * prob_flow_aggreement_fw) >= 0.5
    )
    is_static_artificial_label_bw = castf(
        (0.9 * prob_static_bw + 0.1 * prob_flow_aggreement_bw) >= 0.5
    )

    return (
        is_static_artificial_label_fw,
        is_static_artificial_label_bw,
        {"forward": weight_fw, "backward": weight_bw},
    )


def compute_artificial_label_loss(
    *,
    el,
    prediction_fw: Dict[str, tf.Tensor],
    mask_fw: tf.Tensor,
    prediction_bw: Dict[str, tf.Tensor],
    mask_bw: tf.Tensor,
    knn_results: Dict[str, tf.Tensor],
    final_scale: int,
    loss_cfg: ConfigAttrDict,
    summaries: Dict,
):
    static_flow_key = (
        "static_aggr" if loss_cfg.artificial_labels.use_static_aggr_flow else "static"
    )
    if loss_cfg.artificial_labels.knn_mode == "point":
        knn_dist_sqr_key = "nearest_dist_sqr"
        knn_dist_key = "nearest_dist"
    else:
        assert loss_cfg.artificial_labels.knn_mode == "plane"
        knn_dist_sqr_key = "plumb_line_dist_sqr"
        knn_dist_key = "plumb_line_dist"

    assert loss_cfg.artificial_labels.weight_mode in {
        "constant",
        "gaussian",
        "discrepancy",
        "big_mixture",
    }
    assert (loss_cfg.artificial_labels.weight_mode == "gaussian") == (
        loss_cfg.artificial_labels.gauss_widths is not None
    )
    if loss_cfg.artificial_labels.weight_mode == "gaussian":
        assert loss_cfg.artificial_labels.gauss_widths is not None
        static_w = loss_cfg.artificial_labels.gauss_widths.static
        dynamic_w = loss_cfg.artificial_labels.gauss_widths.dynamic
        assert static_w > 0.0
        assert dynamic_w > 0.0

        is_static_artificial_label_fw = gauss_static_prob(
            knn_results[static_flow_key]["fw_knn"][knn_dist_key],
            knn_results["dynamic"]["fw_knn"][knn_dist_key],
            static_w,
            dynamic_w,
        )
        is_static_artificial_label_bw = gauss_static_prob(
            knn_results[static_flow_key]["bw_knn"][knn_dist_key],
            knn_results["dynamic"]["bw_knn"][knn_dist_key],
            static_w,
            dynamic_w,
        )
        # #region soft labels to weights
        # instead of smooth labels, make labels hard and transfer smoothness
        # to weights
        artificial_label_weights = {
            "forward": tf.abs(2.0 * is_static_artificial_label_fw - 1.0),
            "backward": tf.abs(2.0 * is_static_artificial_label_bw - 1.0),
        }
        is_static_artificial_label_fw = castf(is_static_artificial_label_fw >= 0.5)
        is_static_artificial_label_bw = castf(is_static_artificial_label_bw >= 0.5)
        # #endregion soft labels to weights

    elif loss_cfg.artificial_labels.weight_mode == "constant":
        (
            is_static_artificial_label_fw,
            is_static_artificial_label_bw,
            artificial_label_weights,
        ) = constant_labels(
            knn_results=knn_results,
            static_flow_key=static_flow_key,
            knn_dist_sqr_key=knn_dist_sqr_key,
        )

    elif loss_cfg.artificial_labels.weight_mode == "big_mixture":
        (
            is_static_artificial_label_fw,
            is_static_artificial_label_bw,
            artificial_label_weights,
        ) = big_mixture_labels(
            prediction_fw=prediction_fw,
            prediction_bw=prediction_bw,
            knn_results=knn_results,
            static_flow_key=static_flow_key,
            knn_dist_key=knn_dist_key,
            mixture_dist=loss_cfg.artificial_labels.mixture_distance,
        )

    else:
        assert loss_cfg.artificial_labels.weight_mode == "discrepancy"
        (
            is_static_artificial_label_fw,
            is_static_artificial_label_bw,
            artificial_label_weights,
        ) = discrepancy_labels(
            knn_results=knn_results,
            static_flow_key=static_flow_key,
            knn_dist_key=knn_dist_key,
            discrepancy_threshold=loss_cfg.artificial_labels.discrepancy_threshold,
        )

    ce_loss_fw = tf.reduce_mean(
        tf.keras.backend.binary_crossentropy(
            tf.boolean_mask(is_static_artificial_label_fw, mask_fw),
            tf.boolean_mask(prediction_fw["staticness"], mask_fw),
            from_logits=False,
        )
        * tf.boolean_mask(
            tf.stop_gradient(artificial_label_weights["forward"]), mask_fw
        )
    )
    ce_loss_bw = tf.reduce_mean(
        tf.keras.backend.binary_crossentropy(
            tf.boolean_mask(is_static_artificial_label_bw, mask_bw),
            tf.boolean_mask(prediction_bw["staticness"], mask_bw),
            from_logits=False,
        )
        * tf.boolean_mask(
            tf.stop_gradient(artificial_label_weights["backward"]), mask_bw
        )
    )
    return ce_loss_fw, ce_loss_bw
