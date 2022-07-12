#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from cfgattrdict import AttrDict
from unsup_flow.experiment.artificial_labels import compute_artificial_label_loss
from unsup_flow.knn.knn_wrapper import (
    NearestPointLoss,
    get_flow_matches_loss,
    smoothness_penalty,
    temporal_cls_consistency,
)
from unsup_flow.losses import huber_delta, squared_sum
from unsup_flow.losses.static_points_loss import SymmetricStaticPointsLoss
from unsup_flow.tf import cast32, cast64, castf, make_str


def homogenize_coors(coors):
    assert coors.shape[-1] == 3
    return tf.concat(
        [
            coors,
            tf.ones(tf.concat([tf.shape(coors)[:-1], [1]], axis=0), dtype=coors.dtype),
        ],
        axis=-1,
    )


with tf.name_scope("") as root_scope:
    pass


class SupervisedLoss(tf.keras.layers.Layer):
    def __init__(
        self, cfg, *args, initial_grid_size: Tuple[int, int], final_scale: int, **kwargs
    ):
        super().__init__(*args, autocast=False, **kwargs)
        self.cfg = cfg
        self.initial_grid_size = initial_grid_size
        self.final_scale = final_scale
        # self.symmetric_static_points_loss = SymmetricStaticPointsLoss()

    def call(
        self,
        el,
        prediction_fw,
        prediction_bw,
        moving_dynamicness_threshold,
        summaries,
        training,
    ):
        loss_cfg = self.cfg

        mode = loss_cfg.mode
        assert mode in {
            "total_flow",
            "norig_and_ego",
            "dyn_and_stat_with_cls",
            "norig_and_ego_with_cls",
        }

        ego_flow_t0_t1_transform = tf.linalg.inv(el["odom_t0_t1"]) - np.eye(4)
        gt_ego_flow_t0_t1 = cast32(
            tf.einsum(
                "bij,bnj->bni",
                ego_flow_t0_t1_transform,
                cast64(homogenize_coors(el["pcl_t0"]["pc"])),
            )[..., :3]
        )
        ego_flow_t1_t0_transform = el["odom_t0_t1"] - np.eye(4)
        gt_ego_flow_t1_t0 = cast32(
            tf.einsum(
                "bij,bnj->bni",
                ego_flow_t1_t0_transform,
                cast64(homogenize_coors(el["pcl_t1"]["pc"])),
            )[..., :3]
        )

        mask_t0 = ~tf.reduce_any(tf.math.is_nan(el["pcl_t0"]["pc"]), axis=-1)
        mask_t1 = ~tf.reduce_any(tf.math.is_nan(el["pcl_t1"]["pc"]), axis=-1)

        with tf.control_dependencies(
            [
                tf.debugging.assert_near(
                    tf.boolean_mask(prediction_fw["disappearing"], mask_t0), 0.0
                ),
                tf.debugging.assert_near(
                    tf.boolean_mask(prediction_fw["groundness"], mask_t0), 0.0
                ),
                tf.debugging.assert_near(
                    tf.boolean_mask(prediction_bw["disappearing"], mask_t1), 0.0
                ),
                tf.debugging.assert_near(
                    tf.boolean_mask(prediction_bw["groundness"], mask_t1), 0.0
                ),
            ]
        ):
            static_flow_key = (
                "static_aggr_flow"
                if loss_cfg.use_aggregated_static_flow_instead_of_static_flow
                else "static_flow"
            )
            with tf.name_scope("forward"):
                loss_fw = getattr(self, mode)(
                    gt_ego_flow=gt_ego_flow_t0_t1,
                    gt_agg_flow=el["flow_t0_t1"],
                    gt_flow=el["flow_gt_t0_t1"],
                    pred_stat_flow=prediction_fw[static_flow_key],
                    pred_dyn_flow=prediction_fw["dynamic_flow"],
                    pred_class_logits=prediction_fw["class_logits"],
                    pred_class_probs=prediction_fw["class_probs"],
                    mask=mask_t0,
                    el_pcl=el["pcl_t0"],
                    initial_grid_size=self.initial_grid_size,
                    moving_dynamicness_threshold=moving_dynamicness_threshold,
                    summaries=summaries,
                    training=training,
                )
            with tf.name_scope("backward"):
                loss_bw = getattr(self, mode)(
                    gt_ego_flow=gt_ego_flow_t1_t0,
                    gt_agg_flow=el["flow_t1_t0"],
                    gt_flow=el["flow_gt_t1_t0"],
                    pred_stat_flow=prediction_bw[static_flow_key],
                    pred_dyn_flow=prediction_bw["dynamic_flow"],
                    pred_class_logits=prediction_bw["class_logits"],
                    pred_class_probs=prediction_bw["class_probs"],
                    mask=mask_t1,
                    el_pcl=el["pcl_t1"],
                    initial_grid_size=self.initial_grid_size,
                    moving_dynamicness_threshold=moving_dynamicness_threshold,
                    summaries=summaries,
                    training=training,
                )
            return loss_fw + loss_bw

    def total_flow(
        self,
        *,
        gt_ego_flow,
        gt_agg_flow,
        gt_flow,
        pred_stat_flow,
        pred_dyn_flow,
        pred_class_logits,
        pred_class_probs,
        mask,
        el_pcl,
        initial_grid_size,
        moving_dynamicness_threshold,
        summaries,
        training,
    ):
        stat_prob = pred_class_probs[..., 0]
        dyn_prob = pred_class_probs[..., 1]

        with tf.control_dependencies(
            [
                tf.debugging.assert_near(
                    tf.boolean_mask(dyn_prob, mask),
                    0.0,
                    message="not all dynamic class probs where set to 0",
                ),
                tf.debugging.assert_near(
                    tf.boolean_mask(stat_prob, mask),
                    1.0,
                    message="not all static class probs where set to 1.0",
                ),
            ]
        ):
            flow_diff = gt_agg_flow - pred_stat_flow
            flow_diff = tf.boolean_mask(flow_diff, mask)

            if self.cfg.L1_delta == -1.0:
                loss = tf.reduce_mean(squared_sum(flow_diff, axis=-1))
            else:
                loss = tf.reduce_mean(
                    huber_delta(
                        err_sqr=squared_sum(flow_diff, axis=-1),
                        delta=self.cfg.L1_delta,
                    )
                )

            return loss

    def norig_and_ego(
        self,
        *,
        gt_ego_flow,
        gt_agg_flow,
        pred_stat_flow,
        pred_dyn_flow,
        pred_class_logits,
        pred_class_probs,
        mask,
        el_pcl,
        initial_grid_size,
        summaries,
    ):
        # stat_prob = pred_class_probs[..., 0]
        dyn_prob = pred_class_probs[..., 1]

        with tf.control_dependencies(
            [tf.debugging.assert_near(tf.boolean_mask(dyn_prob, mask), 1.0)]
        ):
            gt_nonrig_flow = gt_agg_flow - gt_ego_flow

            ego_flow_diff = gt_ego_flow - pred_stat_flow
            ego_flow_diff = tf.boolean_mask(ego_flow_diff, mask)
            norig_flow_diff = gt_nonrig_flow - pred_dyn_flow
            norig_flow_diff = tf.boolean_mask(norig_flow_diff, mask)

            if self.cfg.ego_L1_delta == -1.0:
                ego_loss = tf.reduce_mean(squared_sum(ego_flow_diff, axis=-1))
            else:
                ego_loss = tf.reduce_mean(
                    huber_delta(
                        err_sqr=squared_sum(ego_flow_diff, axis=-1),
                        delta=self.cfg.ego_L1_delta,
                    )
                )

            if self.cfg.norig_L1_delta == -1.0:
                norig_loss = tf.reduce_mean(squared_sum(norig_flow_diff, axis=-1))
            else:
                norig_loss = tf.reduce_mean(
                    huber_delta(
                        err_sqr=squared_sum(norig_flow_diff, axis=-1),
                        delta=self.cfg.norig_L1_delta,
                    )
                )
            if summaries["metrics_eval"]:
                tf.summary.scalar("loss/ego", ego_loss)
                tf.summary.scalar("loss/norig", norig_loss)
            return self.cfg.weights.ego * ego_loss + self.cfg.weights.norig * norig_loss

    def dyn_and_stat_with_cls(
        self,
        *,
        gt_ego_flow,
        gt_agg_flow,
        gt_flow,
        pred_stat_flow,
        pred_dyn_flow,
        pred_class_logits,
        pred_class_probs,
        mask,
        el_pcl,
        initial_grid_size,
        moving_dynamicness_threshold,
        summaries,
        training,
    ):
        dyn_prob = pred_class_probs[..., 1]
        stat_logit = pred_class_logits[..., 0]
        dyn_logit = pred_class_logits[..., 1]

        gt_nonrig_flow = gt_agg_flow - gt_ego_flow
        gt_is_dyn = tf.linalg.norm(gt_nonrig_flow, axis=-1) >= self.cfg.dyn_threshold
        masked_gt_is_dyn = tf.boolean_mask(gt_is_dyn, mask)

        stat_flow_diff = gt_ego_flow - pred_stat_flow
        masked_stat_flow_diff = tf.boolean_mask(stat_flow_diff, mask)
        dyn_flow_diff = gt_agg_flow - pred_dyn_flow
        masked_dyn_flow_diff = tf.boolean_mask(dyn_flow_diff, mask)

        flow_diff_with_stat_aggr = gt_agg_flow - pred_stat_flow
        if self.cfg.cls_by_current_error:
            cls_labels_is_dynamic = squared_sum(dyn_flow_diff) < squared_sum(
                flow_diff_with_stat_aggr
            )
            masked_cls_labels_is_dynamic = tf.boolean_mask(cls_labels_is_dynamic, mask)
        else:
            masked_cls_labels_is_dynamic = masked_gt_is_dyn
            cls_labels_is_dynamic = gt_is_dyn

        cls_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
            masked_cls_labels_is_dynamic,
            tf.boolean_mask(dyn_logit - stat_logit, mask),
        )

        if self.cfg.static_L1_delta == -1.0:
            stat_loss = tf.reduce_mean(squared_sum(masked_stat_flow_diff, axis=-1))
        else:
            stat_loss = tf.reduce_mean(
                tf.boolean_mask(
                    huber_delta(
                        err_sqr=squared_sum(masked_stat_flow_diff, axis=-1),
                        delta=self.cfg.static_L1_delta,
                    ),
                    ~masked_gt_is_dyn,
                )
            )

        if self.cfg.dynamic_L1_delta == -1.0:
            dyn_loss = tf.reduce_mean(squared_sum(masked_dyn_flow_diff, axis=-1))
        else:
            dyn_loss = tf.reduce_mean(
                tf.boolean_mask(
                    huber_delta(
                        err_sqr=squared_sum(masked_dyn_flow_diff, axis=-1),
                        delta=self.cfg.dynamic_L1_delta,
                    ),
                    masked_gt_is_dyn,
                )
            )

        moving_dynamicness_threshold.update(
            epes_stat_flow=tf.linalg.norm(
                tf.boolean_mask(flow_diff_with_stat_aggr, mask), axis=-1
            ),
            epes_dyn_flow=tf.linalg.norm(tf.boolean_mask(dyn_flow_diff, mask), axis=-1),
            moving_mask=None
            if moving_dynamicness_threshold.num_still is None
            else tf.boolean_mask(gt_flow["moving_mask"], mask),
            dynamicness_scores=tf.boolean_mask(dyn_prob, mask),
            summaries=summaries,
            training=training,
        )

        if summaries["metrics_eval"]:
            tf.summary.scalar("loss/cls", cls_loss)
            tf.summary.scalar("loss/static", stat_loss)
            tf.summary.scalar("loss/dynamic", dyn_loss)

        return (
            self.cfg.weights.cls * cls_loss
            + self.cfg.weights.static * stat_loss
            + self.cfg.weights.dynamic * dyn_loss
        )

    def norig_and_ego_with_cls(
        self,
        *,
        gt_ego_flow,
        gt_agg_flow,
        pred_stat_flow,
        pred_dyn_flow,
        pred_class_logits,
        pred_class_probs,
        mask,
        el_pcl,
        initial_grid_size,
        summaries,
    ):
        ego_logit = pred_class_logits[..., 0]
        norig_logit = pred_class_logits[..., 1]

        gt_norig_flow = gt_agg_flow - gt_ego_flow
        gt_is_norig = tf.linalg.norm(gt_norig_flow, axis=-1) >= self.cfg.dyn_threshold
        gt_is_norig = tf.boolean_mask(gt_is_norig, mask)

        cls_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
            gt_is_norig,
            tf.boolean_mask(norig_logit - ego_logit, mask),
        )

        ego_flow_diff = gt_ego_flow - pred_stat_flow
        ego_flow_diff = tf.boolean_mask(ego_flow_diff, mask)
        norig_flow_diff = gt_norig_flow - pred_dyn_flow
        norig_flow_diff = tf.boolean_mask(norig_flow_diff, mask)

        if self.cfg.ego_L1_delta == -1.0:
            ego_loss = tf.reduce_mean(squared_sum(ego_flow_diff, axis=-1))
        else:
            ego_loss = tf.reduce_mean(
                huber_delta(
                    err_sqr=squared_sum(ego_flow_diff, axis=-1),
                    delta=self.cfg.ego_L1_delta,
                )
            )

        if self.cfg.norig_L1_delta == -1.0:
            norig_loss = tf.reduce_mean(squared_sum(norig_flow_diff, axis=-1))
        else:
            norig_loss = tf.reduce_mean(
                tf.boolean_mask(
                    huber_delta(
                        err_sqr=squared_sum(norig_flow_diff, axis=-1),
                        delta=self.cfg.norig_L1_delta,
                    ),
                    gt_is_norig,
                )
            )
        if summaries["metrics_eval"]:
            tf.summary.scalar("loss/cls", cls_loss)
            tf.summary.scalar("loss/ego", ego_loss)
            tf.summary.scalar("loss/norig", norig_loss)

        return (
            self.cfg.weights.cls * cls_loss
            + self.cfg.weights.ego * ego_loss
            + self.cfg.weights.norig * norig_loss
        )


class UnsupervisedLoss(tf.keras.layers.Layer):
    def __init__(
        self,
        cfg,
        *args,
        model_cfg,
        bev_extent: Tuple[float, float, float, float],
        final_scale: int,
        **kwargs,
    ):
        super().__init__(*args, autocast=False, **kwargs)
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.final_scale = final_scale
        self.symmetric_static_points_loss = SymmetricStaticPointsLoss()
        self.knn_loss_function = NearestPointLoss(
            bev_extent=bev_extent, **self.cfg.knn_loss
        )

    def call(
        self,
        el,
        prediction_fw,
        prediction_bw,
        moving_dynamicness_threshold,
        summaries,
        training,
    ):
        total_loss = 0.0
        loss_cfg = self.cfg

        predictions = {"fw": prediction_fw, "bw": prediction_bw}
        is_static_flow_penalty_active = (
            loss_cfg.set_default("static_flow_penalty_factor", 1.0) != 0.0
        )
        is_fw_bw_static_trafo_penalty_factor_active = (
            loss_cfg.set_default("fw_bw_static_trafo_penalty_factor", 1.0) != 0.0
        )
        if is_static_flow_penalty_active or is_fw_bw_static_trafo_penalty_factor_active:
            with tf.name_scope("static_loss_components"):
                if loss_cfg.artificial_labels.cross_entropy_penalty > 0.0:
                    staticness_fw = tf.stop_gradient(prediction_fw["staticness"])
                    staticness_bw = tf.stop_gradient(prediction_bw["staticness"])
                else:
                    staticness_fw = prediction_fw["staticness"]
                    staticness_bw = prediction_bw["staticness"]
                (
                    static_flow_loss,
                    for_back_static_trafo_loss,
                ) = self.symmetric_static_points_loss(
                    pc0=el["pcl_t0"]["pc"],
                    static_flow_fw=prediction_fw["static_flow"],
                    static_aggr_trafo_fw=prediction_fw["static_aggr_trafo"],
                    staticness_fw=staticness_fw,
                    pc1=el["pcl_t1"]["pc"],
                    static_flow_bw=prediction_bw["static_flow"],
                    static_aggr_trafo_bw=prediction_bw["static_aggr_trafo"],
                    staticness_bw=staticness_bw,
                    summaries=summaries,
                )

                if is_static_flow_penalty_active:
                    assert loss_cfg.static_flow_penalty_factor > 0.0
                    total_loss = (
                        total_loss
                        + static_flow_loss * loss_cfg.static_flow_penalty_factor
                    )
                if is_fw_bw_static_trafo_penalty_factor_active:
                    assert loss_cfg.fw_bw_static_trafo_penalty_factor > 0.0
                    total_loss = (
                        total_loss
                        + for_back_static_trafo_loss
                        * loss_cfg.fw_bw_static_trafo_penalty_factor
                    )

        with tf.name_scope("occlusion_loss_components"):
            disappearing_masked_fw = tf.boolean_mask(
                prediction_fw["disappearing"],
                tf.logical_not(tf.math.is_nan(prediction_fw["disappearing"])),
            )
            disappearing_masked_bw = tf.boolean_mask(
                prediction_bw["disappearing"],
                tf.logical_not(tf.math.is_nan(prediction_bw["disappearing"])),
            )
            occlusion_loss_fw = tf.reduce_mean(disappearing_masked_fw)
            occlusion_loss_bw = tf.reduce_mean(disappearing_masked_bw)
            occlusion_loss = tf.reduce_mean(
                tf.concat([disappearing_masked_fw, disappearing_masked_bw], axis=0)
            )
            if summaries["metrics_eval"]:
                tf.summary.scalar("occlusion_loss", occlusion_loss)
                tf.summary.scalar("occlusion_loss_fw", occlusion_loss_fw)
                tf.summary.scalar("occlusion_loss_bw", occlusion_loss_bw)

            if loss_cfg.set_default("occlusion_penalty_factor", 0.0) != 0.0:
                assert loss_cfg.occlusion_penalty_factor > 0.0
                total_loss = (
                    total_loss + occlusion_loss * loss_cfg.occlusion_penalty_factor
                )

        with tf.name_scope("knn_loss_components"):
            mask_fw = tf.logical_not(
                tf.reduce_all(tf.math.is_nan(el["pcl_t0"]["pc"]), axis=-1)
            )
            mask_bw = tf.logical_not(
                tf.reduce_all(tf.math.is_nan(el["pcl_t1"]["pc"]), axis=-1)
            )
            masks = {"fw": mask_fw, "bw": mask_bw}

            eval_flow_types = {"aggregated"}
            if loss_cfg.artificial_labels.cross_entropy_penalty > 0.0:
                eval_flow_types.add("dynamic")
                if loss_cfg.artificial_labels.use_static_aggr_flow:
                    eval_flow_types.add("static_aggr")
                else:
                    eval_flow_types.add("static")
            if loss_cfg.knn_on_dynamic_penalty != 0.0:
                assert loss_cfg.knn_on_dynamic_penalty > 0.0
                eval_flow_types.add("dynamic")
            if loss_cfg.knn_on_static_penalty != 0.0:
                assert loss_cfg.knn_on_static_penalty > 0.0
                if self.model_cfg.use_static_aggr_flow_for_aggr_flow:
                    eval_flow_types.add("static_aggr")
                else:
                    eval_flow_types.add("static")

            eval_flow_types = list(eval_flow_types)
            pcl_t0s = [el["pcl_t0"]["pc"]] * len(eval_flow_types)
            pcl_t1s = [el["pcl_t1"]["pc"]] * len(eval_flow_types)
            fw_disapp = [prediction_fw["disappearing"]] * len(eval_flow_types)
            bw_disapp = [prediction_bw["disappearing"]] * len(eval_flow_types)
            fw_flows = []
            bw_flows = []
            for flow_type in eval_flow_types:
                fw_flows.append(prediction_fw["%s_flow" % flow_type])
                bw_flows.append(prediction_bw["%s_flow" % flow_type])

            (fw_flows, bw_flows, pcl_t0s, pcl_t1s, fw_disapp, bw_disapp,) = [
                tf.concat(el, axis=0)
                for el in [
                    fw_flows,
                    bw_flows,
                    pcl_t0s,
                    pcl_t1s,
                    fw_disapp,
                    bw_disapp,
                ]
            ]

            (
                forward_flow_loss_per_type,
                forward_opposite_flow_loss_per_type,
                forward_knn_results_per_type,
                backward_flow_loss_per_type,
                backward_opposite_flow_loss_per_type,
                backward_knn_results_per_type,
            ) = get_flow_matches_loss(
                pcl_t0s,
                fw_flows,
                fw_disapp,
                pcl_t1s,
                bw_flows,
                bw_disapp,
                loss_function=self.knn_loss_function,
                summaries=summaries,
                nearest_dist_mode=loss_cfg.knn_dist_measure,
            )

            knn_results = {}
            for i, flow_type in enumerate(eval_flow_types):
                knn_results[flow_type] = {
                    "fw": forward_flow_loss_per_type[i, ...][None, ...],
                    "fw_opp": forward_opposite_flow_loss_per_type[i, ...][None, ...],
                    "fw_knn": AttrDict(
                        **{
                            k: v[i, ...][None, ...]
                            for k, v in forward_knn_results_per_type.items()
                        }
                    ),
                    "bw": backward_flow_loss_per_type[i, ...][None, ...],
                    "bw_opp": backward_opposite_flow_loss_per_type[i, ...][None, ...],
                    "bw_knn": AttrDict(
                        **{
                            k: v[i, ...][None, ...]
                            for k, v in backward_knn_results_per_type.items()
                        }
                    ),
                }
            # assert set(knn_results_new.keys()) == set(knn_results.keys())

            # def compare_Tensor_dict(some_dict, some_other_dict):
            #     for key, val in some_dict.items():
            #         if isinstance(val, dict) or isinstance(val, AttrDict):
            #             compare_Tensor_dict(val, some_other_dict[key])
            #         else:
            #             assert np.array_equal(val.numpy(), some_other_dict[key].numpy())
            #             print("{0} has passed!".format(key))

            # compare_Tensor_dict(knn_results, knn_results_new)

            if loss_cfg.artificial_labels.cross_entropy_penalty > 0.0:
                ce_loss_fw, ce_loss_bw = compute_artificial_label_loss(
                    el=el,
                    prediction_fw=prediction_fw,
                    mask_fw=mask_fw,
                    prediction_bw=prediction_bw,
                    mask_bw=mask_bw,
                    knn_results=knn_results,
                    final_scale=self.final_scale,
                    loss_cfg=loss_cfg,
                    summaries=summaries,
                )
            else:
                assert loss_cfg.artificial_labels.cross_entropy_penalty == 0.0
                assert loss_cfg.artificial_labels.gauss_widths is None

            forward_flow_loss = tf.reduce_mean(
                tf.boolean_mask(knn_results["aggregated"]["fw"], mask_fw)
            )
            forward_opposite_flow_loss = tf.reduce_mean(
                tf.boolean_mask(knn_results["aggregated"]["fw_opp"], mask_fw)
            )
            backward_flow_loss = tf.reduce_mean(
                tf.boolean_mask(knn_results["aggregated"]["bw"], mask_bw)
            )
            backward_opposite_flow_loss = tf.reduce_mean(
                tf.boolean_mask(knn_results["aggregated"]["bw_opp"], mask_bw)
            )

            flow_loss = 0.5 * (backward_flow_loss + forward_flow_loss)
            opposite_flow_loss = 0.5 * (
                backward_opposite_flow_loss + forward_opposite_flow_loss
            )
            if summaries["metrics_eval"]:
                tf.summary.scalar("knn_forward_flow_loss", forward_flow_loss)
                tf.summary.scalar("knn_backward_flow_loss", backward_flow_loss)
                tf.summary.scalar(
                    "knn_forward_opposite_flow_loss",
                    forward_opposite_flow_loss,
                )
                tf.summary.scalar(
                    "knn_backward_opposite_flow_loss",
                    backward_opposite_flow_loss,
                )

            if loss_cfg.set_default("knn_loss_penalty_factor", 1.0) != 0.0:
                assert loss_cfg.knn_loss_penalty_factor > 0.0
                total_loss = total_loss + flow_loss * loss_cfg.knn_loss_penalty_factor

            if loss_cfg.knn_on_dynamic_penalty != 0.0:
                assert loss_cfg.knn_on_dynamic_penalty > 0.0
                fw_dyn_loss = tf.reduce_mean(
                    tf.boolean_mask(knn_results["dynamic"]["fw"], mask_fw)
                )
                bw_dyn_loss = tf.reduce_mean(
                    tf.boolean_mask(knn_results["dynamic"]["bw"], mask_bw)
                )
                dynamic_flow_loss = 0.5 * (bw_dyn_loss + fw_dyn_loss)
                total_loss = (
                    total_loss + dynamic_flow_loss * loss_cfg.knn_on_dynamic_penalty
                )
                if summaries["metrics_eval"]:
                    tf.summary.scalar("knn_dynamic_flow_loss", dynamic_flow_loss)

            if loss_cfg.knn_on_static_penalty != 0.0:
                assert loss_cfg.knn_on_static_penalty > 0.0
                if self.model_cfg.use_static_aggr_flow_for_aggr_flow:
                    static_flow_key = "static_aggr"
                else:
                    static_flow_key = "static"

                fw_stat_loss = tf.reduce_mean(
                    tf.boolean_mask(knn_results[static_flow_key]["fw"], mask_fw)
                )
                bw_stat_loss = tf.reduce_mean(
                    tf.boolean_mask(knn_results[static_flow_key]["bw"], mask_bw)
                )
                static_flow_loss = 0.5 * (bw_stat_loss + fw_stat_loss)
                total_loss = (
                    total_loss + static_flow_loss * loss_cfg.knn_on_static_penalty
                )
                if summaries["metrics_eval"]:
                    tf.summary.scalar(
                        "knn_%s_flow_loss" % static_flow_key, static_flow_loss
                    )

            if loss_cfg.set_default("opposite_flow_penalty_factor", 1.0) != 0.0:
                assert loss_cfg.opposite_flow_penalty_factor > 0.0
                total_loss = (
                    total_loss
                    + opposite_flow_loss * loss_cfg.opposite_flow_penalty_factor
                )

        with tf.name_scope("update_dynamicness_threshold"):
            # mask_fw ,mask_bw

            if self.model_cfg.output_modification.static_logit == "net":
                assert self.model_cfg.output_modification.dynamic_logit == "net"

                assert "dynamic" in eval_flow_types
                static_key = "static"
                if self.model_cfg.use_static_aggr_flow_for_aggr_flow:
                    static_key = "static_aggr"
                assert static_key in eval_flow_types

                epes_stat_flow = []
                epes_dyn_flow = []
                dynamicness_scores = []
                for flowdir in ["fw", "bw"]:
                    epes_stat_flow.append(
                        tf.boolean_mask(
                            knn_results[static_key]["%s_knn" % flowdir]["nearest_dist"],
                            masks[flowdir],
                        )
                    )
                    epes_dyn_flow.append(
                        tf.boolean_mask(
                            knn_results["dynamic"]["%s_knn" % flowdir]["nearest_dist"],
                            masks[flowdir],
                        )
                    )
                    dynamicness_scores.append(
                        tf.boolean_mask(
                            predictions[flowdir]["dynamicness"], masks[flowdir]
                        )
                    )

                moving_dynamicness_threshold.update(
                    epes_stat_flow=tf.concat(epes_stat_flow, axis=0),
                    epes_dyn_flow=tf.concat(epes_dyn_flow, axis=0),
                    moving_mask=None,
                    dynamicness_scores=tf.concat(dynamicness_scores, axis=0),
                    summaries=summaries,
                    training=training,
                )

        with tf.name_scope("punish_dynamicness"):

            def get_dynamicness_losses(
                dynamicness, mask, static_with_high_dyn_penalty__perc: float = 80.0
            ):
                assert dynamicness.dtype == tf.float32
                assert len(dynamicness.shape) == 1
                assert mask.dtype == tf.bool
                assert len(mask.shape) == 1
                dynamicness = tf.boolean_mask(dynamicness, mask)
                tf.Assert(
                    tf.reduce_all(tf.logical_not(tf.math.is_nan(dynamicness))),
                    data=["found NANs in masked dynamicness tensor"],
                )
                dynamicness_threshold = tfp.stats.percentile(
                    tf.boolean_mask(dynamicness, ~tf.math.is_nan(dynamicness)),
                    static_with_high_dyn_penalty__perc,
                    interpolation="lower",
                    preserve_gradients=False,
                )
                dynamicness_mean_overall = tf.reduce_mean(dynamicness)
                low_dynamicness_mask = dynamicness < dynamicness_threshold
                dynamicness_mean_over_lower_percentile = tf.reduce_mean(
                    tf.boolean_mask(dynamicness, low_dynamicness_mask)
                )

                dynamicness_over_lower_percentile = tf.boolean_mask(
                    dynamicness, low_dynamicness_mask
                )
                dynamicness_mean_over_lower_percentile = tf.reduce_sum(
                    dynamicness_over_lower_percentile
                ) / tf.cast(
                    tf.maximum(1, tf.size(dynamicness_over_lower_percentile)),
                    tf.float32,
                )
                return dynamicness_mean_overall, dynamicness_mean_over_lower_percentile

            (
                dynamicness_mean_overall_fw,
                dynamicness_mean_over_lower_percentile_fw,
            ) = tf.map_fn(
                lambda x_and_mask: get_dynamicness_losses(
                    x_and_mask[0],
                    x_and_mask[1],
                    static_with_high_dyn_penalty__perc=loss_cfg.dynamicness.static_with_high_dyn_penalty__perc,
                ),
                [prediction_fw["dynamicness"], mask_fw],
                dtype=(
                    prediction_fw["dynamicness"].dtype,
                    prediction_fw["dynamicness"].dtype,
                ),
                swap_memory=False,
                parallel_iterations=8,
            )
            (
                dynamicness_mean_overall_bw,
                dynamicness_mean_over_lower_percentile_bw,
            ) = tf.map_fn(
                lambda x_and_mask: get_dynamicness_losses(
                    x_and_mask[0],
                    x_and_mask[1],
                    static_with_high_dyn_penalty__perc=loss_cfg.dynamicness.static_with_high_dyn_penalty__perc,
                ),
                [prediction_bw["dynamicness"], mask_bw],
                dtype=(
                    prediction_bw["dynamicness"].dtype,
                    prediction_bw["dynamicness"].dtype,
                ),
                swap_memory=False,
                parallel_iterations=8,
            )
            assert (
                loss_cfg.dynamicness.penalty_lower_perc
                >= loss_cfg.dynamicness.penalty_upper_perc
                >= 0.0
            )
            (
                dynamicness_mean_overall_fw,
                dynamicness_mean_over_lower_percentile_fw,
                dynamicness_mean_overall_bw,
                dynamicness_mean_over_lower_percentile_bw,
            ) = map(
                tf.reduce_mean,
                (
                    dynamicness_mean_overall_fw,
                    dynamicness_mean_over_lower_percentile_fw,
                    dynamicness_mean_overall_bw,
                    dynamicness_mean_over_lower_percentile_bw,
                ),
            )
            if summaries["metrics_eval"]:
                tf.summary.scalar("dynamicness/overall_fw", dynamicness_mean_overall_fw)
                tf.summary.scalar("dynamicness/overall_bw", dynamicness_mean_overall_bw)
                tf.summary.scalar(
                    "dynamicness/over_lower_percentile_fw",
                    dynamicness_mean_over_lower_percentile_fw,
                )
                tf.summary.scalar(
                    "dynamicness/over_lower_percentile_bw",
                    dynamicness_mean_over_lower_percentile_bw,
                )
            if loss_cfg.dynamicness.penalty_lower_perc > 0.0:
                total_loss = (
                    total_loss
                    + 0.5
                    * (
                        dynamicness_mean_over_lower_percentile_fw
                        + dynamicness_mean_over_lower_percentile_bw
                    )
                    * loss_cfg.dynamicness.penalty_lower_perc
                )
            if loss_cfg.dynamicness.penalty_upper_perc > 0.0:
                total_loss = (
                    total_loss
                    + 0.5
                    * (dynamicness_mean_overall_fw + dynamicness_mean_overall_bw)
                    * loss_cfg.dynamicness.penalty_upper_perc
                )
        with tf.name_scope("enforce_smoothness"):
            smoothness_penalties = []
            num_neighbors_for_smoothness = loss_cfg.num_neighbors_smoothness_penalty
            for key, factor in loss_cfg.smoothness_penalty_factor.items():
                if factor != 0.0:
                    assert factor > 0.0

                    smoothness_penalty_fw = smoothness_penalty(
                        el["pcl_t0"]["pc"],
                        prediction_fw["%s_flow" % key],
                        num_neighbors_for_smoothness,
                    )
                    smoothness_penalty_bw = smoothness_penalty(
                        el["pcl_t1"]["pc"],
                        prediction_bw["%s_flow" % key],
                        num_neighbors_for_smoothness,
                    )
                    total_smoothness_penalty = 0.5 * (
                        smoothness_penalty_fw + smoothness_penalty_bw
                    )
                    if summaries["metrics_eval"]:
                        tf.summary.scalar(
                            "smoothness_penalty/%s" % key, total_smoothness_penalty
                        )

                    smoothness_penalties.append(factor * total_smoothness_penalty)

            if len(smoothness_penalties) > 0:
                total_loss += tf.add_n(smoothness_penalties)

        with tf.name_scope("temporal_cls_consistency"):
            if loss_cfg.temporal_cls_consistency_penalty_factor != 0.0:
                assert loss_cfg.temporal_cls_consistency_penalty_factor > 0.0
                tcc_fw = temporal_cls_consistency(
                    cloud_a=el["pcl_t0"]["pc"],
                    flow_a_b=prediction_fw["aggregated_flow"],
                    cloud_b=el["pcl_t1"]["pc"],
                    class_logits_a=prediction_fw["class_logits"],
                    class_probs_b=prediction_bw["class_probs"],
                    mask_a=mask_fw,
                )
                tcc_bw = temporal_cls_consistency(
                    cloud_a=el["pcl_t1"]["pc"],
                    flow_a_b=prediction_bw["aggregated_flow"],
                    cloud_b=el["pcl_t0"]["pc"],
                    class_logits_a=prediction_bw["class_logits"],
                    class_probs_b=prediction_fw["class_probs"],
                    mask_a=mask_bw,
                )
                total_tcc = 0.5 * (tcc_fw + tcc_bw)
                if summaries["metrics_eval"]:
                    tf.summary.scalar("total", total_tcc)
                total_loss += (
                    loss_cfg.temporal_cls_consistency_penalty_factor * total_tcc
                )

        if loss_cfg.artificial_labels.cross_entropy_penalty > 0.0:
            total_loss = (
                total_loss
                + 0.5
                * (ce_loss_fw + ce_loss_bw)
                * loss_cfg.artificial_labels.cross_entropy_penalty
            )
            if summaries["metrics_eval"]:
                tf.summary.scalar("artificial_static_fw_CE", ce_loss_fw)
                tf.summary.scalar("artificial_static_bw_CE", ce_loss_bw)

        return total_loss


def get_avg_epe_for_class(
    end_point_errors_per_points, semantics, label_idx, valid_labels_mask
):
    static_points_mask_fw = tf.logical_and(
        tf.logical_not(tf.math.is_nan(end_point_errors_per_points)),
        tf.logical_and(
            tf.equal(semantics, label_idx),
            valid_labels_mask,
        ),
    )
    avg_static_epe_fw = tf.reduce_mean(
        tf.boolean_mask(end_point_errors_per_points, static_points_mask_fw)
    )
    return avg_static_epe_fw


def iou_from_confusion_matrix(confusion_matrix, class_names: List[str] = None):
    """
    Thanks to our colleague Larissa for the following method!
    """
    tp = tf.cast(
        tf.linalg.tensor_diag_part(confusion_matrix, name="true_positives"), tf.float64
    )
    sum_over_row = tf.cast(tf.reduce_sum(confusion_matrix, axis=0), tf.float64)
    sum_over_col = tf.cast(tf.reduce_sum(confusion_matrix, axis=1), tf.float64)

    denominator = sum_over_row + sum_over_col - tp

    # Output per-class IoU vector where a NaN entry indicates no instance
    # of this class was present (all observed instances were true negatives)
    class_iou_nan = tf.divide(tp, denominator, name="class_iou_nan")
    # manually compute mean iou from the class iou
    # necessary to compute the miou based on an updated confusion matrix
    tf_miou = tf.reduce_mean(
        tf.boolean_mask(class_iou_nan, tf.logical_not(tf.math.is_nan(class_iou_nan)))
    )

    # If no class iou is available at all, set miou as 0 instead of nan
    tf_miou = tf.where(tf.math.is_nan(tf_miou), tf.cast(0.0, tf.float64), tf_miou)

    # If the denominator is zero, replace it with 1. to avoid division by zero
    denominator = tf.where(
        tf.greater(denominator, 0), denominator, tf.ones_like(denominator)
    )

    # Output per-class IoU vector where NaN values were replaced by zero
    class_iou = tf.divide(tp, denominator, name="class_iou")

    if class_names is not None:
        return {
            "mIoU": tf_miou,
            **{
                "IoU/" + cls_name: class_iou[i]
                for i, cls_name in enumerate(class_names)
            },
        }
    return tf_miou, class_iou


def plot_some_supervised_metrics(el, prediction, label_dict, summaries, direction):
    assert direction in ["fw", "bw"]
    semantics_dict_key = "semantics_t0" if direction == "fw" else "semantics_t1"
    flow_gt_dict_key = "flow_gt_t0_t1" if direction == "fw" else "flow_gt_t1_t0"

    with tf.name_scope("supervised_diagnostics" + "_" + direction + "_"):
        # AEE
        if flow_gt_dict_key in el:
            endpoint_errors = tf.linalg.norm(
                prediction["aggregated_flow"] - el[flow_gt_dict_key]["flow"], axis=-1
            )
            mask = tf.logical_and(
                tf.logical_not(tf.math.is_nan(endpoint_errors)),
                el[flow_gt_dict_key]["exact_gt_mask"],
            )
            avg_endpoint_error = tf.reduce_mean(tf.boolean_mask(endpoint_errors, mask))
            tf.summary.scalar("AEE_overall_from_aggr", avg_endpoint_error)

            if semantics_dict_key in el:
                avg_static_epe = get_avg_epe_for_class(
                    endpoint_errors,
                    el[semantics_dict_key],
                    label_dict["static"],
                    el[flow_gt_dict_key]["exact_gt_mask"],
                )
                tf.summary.scalar("AEE_static_from_aggr", avg_static_epe)

                avg_dynamic_epe = get_avg_epe_for_class(
                    endpoint_errors,
                    el[semantics_dict_key],
                    label_dict["dynamic"],
                    el[flow_gt_dict_key]["exact_gt_mask"],
                )
                tf.summary.scalar("AEE_dynamic_from_aggr", avg_dynamic_epe)

        if semantics_dict_key in el:
            # Classification IOU Stuff
            mask = tf.logical_and(
                ~tf.math.is_nan(prediction["dynamicness"]),
                el[flow_gt_dict_key]["exact_gt_mask"],
            )
            conf_mat = new_compute_custom_conf_mat(
                gt_moving=el[flow_gt_dict_key]["moving_mask"],
                semantics_gt=el[semantics_dict_key],
                label_dict=label_dict,
                mask=mask,
                prediction_is_dynamic=prediction["is_dynamic"],
            )
            ious = compute_custom_class_iou_metrics(conf_mat)

            for k, v in ious.items():
                if len(v.shape) == 0 and v.dtype != tf.string:
                    tf.summary.scalar("classification/%s" % k, v)
                else:
                    tf.summary.text("classification/%s" % k, v)


def new_compute_custom_conf_mat(
    *,
    gt_moving,
    semantics_gt=None,
    label_dict,
    mask,
    prediction_is_dynamic,
):
    assert prediction_is_dynamic.dtype == tf.bool
    masked_gt_moving = tf.boolean_mask(gt_moving, mask)
    masked_prediction_is_dynamic = tf.boolean_mask(prediction_is_dynamic, mask)
    if semantics_gt is None:
        masked_gt_static = ~masked_gt_moving
        masked_gt_ground = tf.zeros_like(masked_gt_moving)
        masked_gt_still_dynamic = tf.zeros_like(masked_gt_moving)
        masked_gt_moving_dynamic = masked_gt_moving
    else:
        masked_semantics = tf.boolean_mask(semantics_gt, mask)
        # tf.debugging.assert_equal(
        #     masked_semantics,
        #     masked_semantics,
        #     message="sanity assert",
        # )
        tf.debugging.assert_equal(
            1,
            tf.math.count_nonzero(
                tf.equal(
                    masked_semantics[..., None],
                    [label_dict[k] for k in sorted(label_dict.keys())],
                ),
                axis=-1,
                dtype=tf.int32,
            ),
            message="unidentified semantics idxs",
        )

        assert "ignore" in label_dict
        ignore_mask = masked_semantics == label_dict["ignore"]
        # tf.debugging.assert_less(
        #     tf.math.count_nonzero(ignore_mask, dtype=tf.int32),
        #     tf.size(ignore_mask) // 10,
        #     message="too many points ignored",
        # )
        masked_semantics = tf.boolean_mask(masked_semantics, ~ignore_mask)
        masked_gt_moving = tf.boolean_mask(masked_gt_moving, ~ignore_mask)
        masked_prediction_is_dynamic = tf.boolean_mask(
            masked_prediction_is_dynamic, ~ignore_mask
        )

        masked_gt_static = masked_semantics == label_dict["static"]
        masked_gt_ground = masked_semantics == label_dict["ground"]
        masked_gt_dynamic = masked_semantics == label_dict["dynamic"]
        masked_gt_still_dynamic = ~masked_gt_moving & masked_gt_dynamic
        masked_gt_moving_dynamic = masked_gt_moving & masked_gt_dynamic
        # tf.debugging.assert_equal(
        #     masked_gt_static,
        #     masked_gt_static & ~masked_gt_moving,
        #     message="there are moving elements semantically being static",
        # )
        # tf.debugging.assert_equal(
        #     masked_gt_moving_dynamic,
        #     masked_gt_moving,
        #     message="there are moving elements not semantically being dynamic",
        # )
    return compute_custom_conf_mat(
        masked_prediction_is_dynamic,
        masked_gt_static,
        masked_gt_ground,
        masked_gt_still_dynamic,
        masked_gt_moving_dynamic,
    )


def compute_custom_conf_mat(
    prediction_is_dynamic,
    is_static_gt,
    is_ground_gt,
    is_still_dynamic_gt,
    is_moving_dynamic_gt,
):
    tf.debugging.assert_equal(
        1,
        tf.math.count_nonzero(
            tf.stack(
                [
                    is_static_gt,
                    is_ground_gt,
                    is_still_dynamic_gt,
                    is_moving_dynamic_gt,
                ],
                axis=-1,
            ),
            axis=-1,
            dtype=tf.int32,
        ),
        message="missing gt cls annotations",
    )

    def casti(x):
        return tf.cast(x, tf.int32)

    # idx = 0*static + 1*ground + 2*still_dynamic + 3*moving_dynamic
    gt_labels_idx = (
        casti(is_ground_gt)
        + 2 * casti(is_still_dynamic_gt)
        + 3 * casti(is_moving_dynamic_gt)
    )
    pred_labels_idx = 3 * casti(prediction_is_dynamic)

    conf_mat = tf.math.confusion_matrix(
        gt_labels_idx, pred_labels_idx, num_classes=4, dtype=tf.int64
    )
    return conf_mat


def conf_mat_to_headered_str_table(conf_mat, class_names):
    conf_mat = make_str(castf(conf_mat))
    header = tf.constant([""] + class_names)
    return tf.concat(
        [header[None, :], tf.concat([header[1:, None], conf_mat], axis=1)], axis=0
    )


def compute_custom_class_iou_metrics(conf_mat):
    clsn = ["static", "ground", "still_dynamic", "moving_dynamic"]
    stat = clsn.index("static")
    grnd = clsn.index("ground")
    stdyn = clsn.index("still_dynamic")
    mvdyn = clsn.index("moving_dynamic")

    moving_acc = conf_mat[mvdyn, mvdyn] / tf.maximum(tf.reduce_sum(conf_mat[mvdyn]), 1)
    static_acc = conf_mat[stat, stat] / tf.maximum(tf.reduce_sum(conf_mat[stat]), 1)
    pred_dyn_of_still_dyn = conf_mat[stdyn, mvdyn] / tf.maximum(
        tf.reduce_sum(conf_mat[stdyn]), 1
    )
    pred_dyn_of_ground = conf_mat[grnd, mvdyn] / tf.maximum(
        tf.reduce_sum(conf_mat[grnd]), 1
    )
    ious_ignore_ground_still_dyn = iou_from_confusion_matrix(
        conf_mat[::3, ::3], ["static", "moving_dynamic"]
    )

    return {
        "conf_mat": tf.gather(
            conf_mat_to_headered_str_table(conf_mat, clsn), [0, 1, 4], axis=-1
        ),
        "moving_acc": moving_acc,
        "static_acc": static_acc,
        "pred_dyn_of_still_dyn": pred_dyn_of_still_dyn,
        "pred_dyn_of_ground": pred_dyn_of_ground,
        **{
            "ignore_ground_still_dyn/" + k: v
            for k, v in ious_ignore_ground_still_dyn.items()
        },
    }
