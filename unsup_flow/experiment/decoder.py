#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


from typing import Dict

import numpy as np
import tensorflow as tf

from cfgattrdict import AttrDict, ConfigAttrDict
from unsup_flow.experiment.static_aggregation import compute_static_aggregated_flow
from unsup_flow.tf import cast32, cast64, castf, max_pool_2d_flow_map
from unsup_flow.tf.numerical_stability import normalized_sigmoid_sum


def scale_gradient(tensor, scaling):
    if scaling == 1.0:
        return tensor
    if scaling == 0.0:
        return tf.stop_gradient(tensor)
    assert scaling > 0.0
    return tensor * scaling - tf.stop_gradient(tensor) * (scaling - 1.0)


def artificial_flow_network_output(
    *,
    network_output_dict: Dict[str, tf.Tensor],
    model_cfg: ConfigAttrDict,
    gt_flow_bev: tf.Tensor,
    gt_static_flow: tf.Tensor,
):
    out_mod_cfg = model_cfg.output_modification
    # #region static_flow
    if out_mod_cfg.static_flow == "net":
        pass
    elif out_mod_cfg.static_flow == "gt":
        network_output_dict["static_flow"] = gt_static_flow
    elif out_mod_cfg.static_flow == "zero":
        network_output_dict["static_flow"] = tf.zeros_like(
            network_output_dict["static_flow"]
        )
    else:
        raise ValueError("unknown output mode: %s" % str(out_mod_cfg.static_flow))
    # #endregion static_flow

    # #region dynamic_flow
    if out_mod_cfg.dynamic_flow == "net":
        pass
    elif out_mod_cfg.dynamic_flow == "gt":
        network_output_dict["dynamic_flow"] = gt_flow_bev
        if model_cfg.dynamic_flow_is_non_rigid_flow:
            network_output_dict["dynamic_flow"] = (
                network_output_dict["dynamic_flow"] - network_output_dict["static_flow"]
            )
    elif out_mod_cfg.dynamic_flow == "zero":
        network_output_dict["dynamic_flow"] = tf.zeros_like(
            network_output_dict["dynamic_flow"]
        )
    else:
        raise ValueError("unknown output mode: %s" % str(out_mod_cfg.dynamic_flow))
    network_output_dict["dynamic_flow"] = scale_gradient(
        network_output_dict["dynamic_flow"],
        out_mod_cfg.dynamic_flow_grad_scale,
    )
    # #endregion dynamic_flow
    return network_output_dict


def artificial_logit_network_output(
    *,
    network_output_dict: Dict[str, tf.Tensor],
    model_cfg: ConfigAttrDict,
    ohe_gt_stat_dyn_ground_label_bev_map: tf.Tensor,
    gt_flow_bev: tf.Tensor,
    gt_static_flow: tf.Tensor,
):
    out_mod_cfg = model_cfg.output_modification
    ones = tf.ones_like(network_output_dict["static_logit"])

    # #region disappearing_logit
    if out_mod_cfg.disappearing_logit == "net":
        pass
    elif out_mod_cfg.disappearing_logit == "gt":
        raise NotImplementedError()
    elif out_mod_cfg.disappearing_logit is True:
        network_output_dict["disappearing_logit"] = 0 * ones
    elif out_mod_cfg.disappearing_logit is False:
        network_output_dict["disappearing_logit"] = -100 * ones
    else:
        raise ValueError(
            "unknown output mode: %s" % str(out_mod_cfg.disappearing_logit)
        )
    # #endregion disappearing_logit

    # #region static_logit
    if out_mod_cfg.static_logit == "net":
        pass
    elif out_mod_cfg.static_logit == "gt_label_based":
        assert out_mod_cfg.dynamic_logit == "gt_label_based"
        if out_mod_cfg.ground_logit is False:
            # add gt labels to static if ground == off
            gt_stat = (
                ohe_gt_stat_dyn_ground_label_bev_map[..., 0:1]
                | ohe_gt_stat_dyn_ground_label_bev_map[..., 2:3]
            )
            gt_stat_flt = castf(gt_stat)
            network_output_dict["static_logit"] = 100.0 * (gt_stat_flt - 1.0)
        elif out_mod_cfg.ground_logit == "gt_label_based":
            network_output_dict["static_logit"] = 100.0 * (
                castf(ohe_gt_stat_dyn_ground_label_bev_map[..., 0:1]) - 1.0
            )
        else:
            raise AssertionError(
                "when using gt_label for cls then ground_logit must be `gt_label_based` or `off`, not %s"
                % out_mod_cfg.ground_logit
            )
    elif out_mod_cfg.static_logit == "gt_flow_based":
        assert out_mod_cfg.dynamic_logit == "gt_flow_based"
        assert out_mod_cfg.ground_logit is False

        norig_flow = gt_flow_bev - gt_static_flow
        bev_is_static_map = tf.linalg.norm(norig_flow, axis=-1, keepdims=True) <= 0.05
        bev_is_static_map = tf.cast(bev_is_static_map, dtype=tf.float32)
        network_output_dict["static_logit"] = 100.0 * (bev_is_static_map - 1.0)
    elif out_mod_cfg.static_logit is True:
        assert out_mod_cfg.dynamic_logit is False
        assert out_mod_cfg.ground_logit is False
        network_output_dict["static_logit"] = tf.stop_gradient(
            tf.reduce_max(
                [
                    network_output_dict["dynamic_logit"],
                    network_output_dict["ground_logit"],
                ]
            )
            + 100.0 * ones
        )
    elif out_mod_cfg.static_logit is False:
        assert (
            out_mod_cfg.dynamic_logit is not False
            or out_mod_cfg.ground_logit is not False
        )
        network_output_dict["static_logit"] = tf.stop_gradient(
            tf.reduce_min(
                [
                    network_output_dict["dynamic_logit"],
                    network_output_dict["ground_logit"],
                ]
            )
            - 100.0 * ones
        )
    else:
        raise ValueError("unknown output mode: %s" % str(out_mod_cfg.static_logit))
    # #endregion static_logit

    # #region dynamic_logit
    if out_mod_cfg.dynamic_logit == "net":
        pass
    elif out_mod_cfg.dynamic_logit == "gt_label_based":
        assert out_mod_cfg.static_logit == "gt_label_based"
        network_output_dict["dynamic_logit"] = 100.0 * (
            castf(ohe_gt_stat_dyn_ground_label_bev_map[..., 1:2]) - 1.0
        )
    elif out_mod_cfg.dynamic_logit == "gt_flow_based":
        network_output_dict["dynamic_logit"] = (
            100.0 - network_output_dict["static_logit"]
        )
    elif out_mod_cfg.dynamic_logit is True:
        # assert out_mod_cfg.static_logit is False
        # assert out_mod_cfg.ground_logit is False
        network_output_dict["dynamic_logit"] = tf.stop_gradient(
            tf.reduce_max(
                [
                    network_output_dict["static_logit"],
                    network_output_dict["ground_logit"],
                ]
            )
            + 100.0 * ones
        )
    elif out_mod_cfg.dynamic_logit is False:
        network_output_dict["dynamic_logit"] = tf.stop_gradient(
            tf.reduce_min(
                [
                    network_output_dict["static_logit"],
                    network_output_dict["ground_logit"],
                ]
            )
            - 100.0 * ones
        )
    else:
        raise ValueError("unknown output mode: %s" % str(out_mod_cfg.dynamic_logit))
    # #endregion dynamic_logit

    # #region ground_logit
    if out_mod_cfg.ground_logit == "net":
        pass
    elif out_mod_cfg.ground_logit == "gt_label_based":
        assert out_mod_cfg.static_logit == "gt_label_based"
        assert out_mod_cfg.dynamic_logit == "gt_label_based"
        network_output_dict["ground_logit"] = 100.0 * (
            castf(ohe_gt_stat_dyn_ground_label_bev_map[..., 2:3]) - 1.0
        )
    elif out_mod_cfg.ground_logit is True:
        assert out_mod_cfg.static_logit is False
        assert out_mod_cfg.dynamic_logit is False
        network_output_dict["ground_logit"] = tf.stop_gradient(
            tf.reduce_max(
                [
                    network_output_dict["static_logit"],
                    network_output_dict["dynamic_logit"],
                ]
            )
            + 100.0 * ones
        )
    elif out_mod_cfg.ground_logit is False:
        network_output_dict["ground_logit"] = tf.stop_gradient(
            tf.reduce_min(
                [
                    network_output_dict["static_logit"],
                    network_output_dict["dynamic_logit"],
                ]
            )
            - 100.0 * ones
        )
    else:
        raise ValueError("unknown output mode: %s" % str(out_mod_cfg.ground_logit))
    # #endregion ground_logit
    return network_output_dict


def artificial_network_output(
    *,
    network_output_dict: Dict[str, tf.Tensor],
    dynamicness_threshold: tf.Tensor,
    cfg: ConfigAttrDict,
    ohe_gt_stat_dyn_ground_label_bev_map: tf.Tensor,
    gt_flow_bev: tf.Tensor,
    gt_static_flow: tf.Tensor,
    filled_pillar_mask: tf.Tensor,
    pc: tf.Tensor,
    pointwise_voxel_coordinates_fs: tf.Tensor,
    pointwise_valid_mask: tf.Tensor,
    voxel_center_metric_coordinates: np.array,
    overwrite_non_filled_pillars_with_default_flow: bool = True,
    overwrite_non_filled_pillars_with_default_logits: bool = True,
):
    model_cfg = cfg.model
    out_mod_cfg = model_cfg.output_modification

    assert len(network_output_dict["static_flow"].shape) == len(
        filled_pillar_mask.shape
    ), (
        network_output_dict["static_flow"].shape,
        filled_pillar_mask.shape,
    )

    with tf.name_scope("artificial_network_output"):

        with tf.name_scope("flow"):
            network_output_dict = artificial_flow_network_output(
                network_output_dict=network_output_dict,
                model_cfg=model_cfg,
                gt_flow_bev=gt_flow_bev,
                gt_static_flow=gt_static_flow,
            )

        with tf.name_scope("logits"):
            network_output_dict = artificial_logit_network_output(
                network_output_dict=network_output_dict,
                model_cfg=model_cfg,
                ohe_gt_stat_dyn_ground_label_bev_map=ohe_gt_stat_dyn_ground_label_bev_map,
                gt_flow_bev=gt_flow_bev,
                gt_static_flow=gt_static_flow,
            )

    with tf.name_scope("mask_nonfilled_pillars"):
        default_values_for_nonfilled_pillars = {
            "disappearing_logit": -100.0,
            "static_logit": -100.0 if out_mod_cfg.static_logit is False else 0.0,
            "dynamic_logit": 0.0 if out_mod_cfg.dynamic_logit is True else -100.0,
            "ground_logit": 0.0 if out_mod_cfg.ground_logit is True else -100.0,
            "static_flow": 0.0,
            "dynamic_flow": 0.0,
            "static_aggr_flow": 0.0,
        }

        modification_taboo_keys = []
        if not overwrite_non_filled_pillars_with_default_flow:
            modification_taboo_keys += [
                "static_flow",
                "dynamic_flow",
                "static_aggr_flow",
            ]
        if not overwrite_non_filled_pillars_with_default_logits:
            modification_taboo_keys += [
                "disappearing_logit",
                "static_logit",
                "dynamic_logit",
                "ground_logit",
            ]

        for k in network_output_dict:
            if k == "weight_logits_for_static_aggregation":
                continue
            assert len(network_output_dict[k].shape) == len(filled_pillar_mask.shape), (
                k,
                network_output_dict[k].shape,
                filled_pillar_mask.shape,
            )

            if k in modification_taboo_keys:
                continue

            network_output_dict[k] = tf.where(
                filled_pillar_mask,
                network_output_dict[k],
                default_values_for_nonfilled_pillars[k]
                * tf.ones_like(network_output_dict[k]),
            )

    with tf.name_scope("construct_class_probs"):
        network_output_dict["class_logits"] = tf.concat(
            [
                network_output_dict["static_logit"],
                network_output_dict["dynamic_logit"],
                network_output_dict["ground_logit"],
            ],
            axis=-1,
        )
        network_output_dict["class_probs"] = tf.nn.softmax(
            network_output_dict["class_logits"]
        )
        network_output_dict["staticness"] = network_output_dict["class_probs"][..., 0]
        network_output_dict["dynamicness"] = network_output_dict["class_probs"][..., 1]
        network_output_dict["groundness"] = network_output_dict["class_probs"][..., 2]
        network_output_dict["is_dynamic"] = (
            network_output_dict["dynamicness"] >= dynamicness_threshold
        )
        network_output_dict["is_static"] = (
            network_output_dict["staticness"] >= network_output_dict["groundness"]
        ) & (~network_output_dict["is_dynamic"])
        network_output_dict["is_ground"] = ~(
            network_output_dict["is_static"] | network_output_dict["is_dynamic"]
        )

    with tf.name_scope("construct_static_aggregation"):
        static_aggr_weight_map = network_output_dict["staticness"] * castf(
            filled_pillar_mask[..., 0]
        )
        if model_cfg.predict_weight_for_static_aggregation is not False:
            mode = model_cfg.predict_weight_for_static_aggregation
            assert mode in {"sigmoid", "softmax"}
            if mode == "softmax":
                network_output_dict["masked_weights_for_static_aggregation"] = tf.where(
                    filled_pillar_mask[..., 0],
                    network_output_dict["weight_logits_for_static_aggregation"],
                    tf.ones_like(
                        network_output_dict["weight_logits_for_static_aggregation"]
                    )
                    * (
                        tf.reduce_min(
                            network_output_dict["weight_logits_for_static_aggregation"]
                        )
                        - 1000.0
                    ),
                )
                curshape = network_output_dict[
                    "masked_weights_for_static_aggregation"
                ].shape
                assert len(curshape) == 3, curshape
                prodshape = curshape[-1] * curshape[-2]
                network_output_dict[
                    "masked_weights_for_static_aggregation"
                ] = tf.reshape(
                    tf.nn.softmax(
                        tf.reshape(
                            network_output_dict[
                                "masked_weights_for_static_aggregation"
                            ],
                            (-1, prodshape),
                        )
                    ),
                    (-1, *curshape[-2:]),
                )
            else:
                assert mode == "sigmoid"
                grid_size = filled_pillar_mask.shape[-3:-1]
                prod_size = grid_size[0] * grid_size[1]
                network_output_dict[
                    "masked_weights_for_static_aggregation"
                ] = tf.reshape(
                    normalized_sigmoid_sum(
                        logits=tf.reshape(
                            network_output_dict["weight_logits_for_static_aggregation"],
                            [-1, prod_size],
                        ),
                        mask=tf.reshape(filled_pillar_mask[..., 0], [-1, prod_size]),
                    ),
                    [-1, *grid_size],
                )
            static_aggr_weight_map = (
                static_aggr_weight_map
                * network_output_dict["masked_weights_for_static_aggregation"]
            )
        (
            network_output_dict["static_aggr_flow"],
            static_aggr_trafo,
            not_enough_points,
        ) = compute_static_aggregated_flow(
            network_output_dict["static_flow"],
            static_aggr_weight_map,
            pc,
            pointwise_voxel_coordinates_fs,
            pointwise_valid_mask,
            voxel_center_metric_coordinates,
            use_eps_for_weighted_pc_alignment=cfg.losses.unsupervised.use_epsilon_for_weighted_pc_alignment,
        )
        network_output_dict["masked_static_aggr_flow"] = tf.where(
            filled_pillar_mask,
            network_output_dict["static_aggr_flow"],
            tf.zeros_like(network_output_dict["static_aggr_flow"]),
        )
        network_output_dict["masked_gt_static_flow"] = tf.where(
            filled_pillar_mask,
            gt_static_flow,
            tf.zeros_like(network_output_dict["masked_static_aggr_flow"]),
        )

    return network_output_dict, static_aggr_trafo, not_enough_points


class HeadDecoder(tf.keras.layers.Layer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, autocast=False, **kwargs)
        self.cfg = cfg

    def concat2network_output(
        self,
        logits,
        static_flow,
        dynamic_flow,
        weight_logits_for_static_aggregation=None,
    ):
        assert logits.shape[-1] == 4
        assert static_flow.shape[-1] == 2
        assert dynamic_flow.shape[-1] == 2
        assert (weight_logits_for_static_aggregation is None) == (
            not self.cfg.model.predict_weight_for_static_aggregation
        )
        if weight_logits_for_static_aggregation is None:
            return tf.concat([logits, static_flow, dynamic_flow], axis=-1)
        assert weight_logits_for_static_aggregation.shape[-1] == 1
        return tf.concat(
            [
                logits,
                static_flow,
                dynamic_flow,
                weight_logits_for_static_aggregation,
            ],
            axis=-1,
        )

    def apply_output_modification(
        self,
        network_output,
        dynamicness_threshold,
        *,
        pc,
        pointwise_voxel_coordinates_fs,
        pointwise_valid_mask,
        filled_pillar_mask,
        inv_odom,
        gt_flow_bev=None,
        ohe_gt_stat_dyn_ground_label_bev_map=None,
        dynamic_flow_is_non_rigid_flow=False,
        overwrite_non_filled_pillars_with_default_flow: bool = True,
        overwrite_non_filled_pillars_with_default_logits: bool = True,
    ):
        flow_dim = 2
        assert 3 == len(filled_pillar_mask.shape) == len(network_output.shape) - 1, (
            filled_pillar_mask.shape,
            network_output.shape,
        )
        assert (
            filled_pillar_mask.shape[-2:].as_list()
            == network_output.shape[-3:-1].as_list()
        ), (filled_pillar_mask.shape, network_output.shape)
        filled_pillar_mask = filled_pillar_mask[..., None]

        with tf.name_scope("network_output_slicing"):
            network_output_dict = {}
            if self.cfg.model.predict_weight_for_static_aggregation is not False:
                network_output_dict[
                    "weight_logits_for_static_aggregation"
                ] = network_output[..., -1]
                network_output = network_output[..., :-1]
            assert network_output.shape[-1] == 4 + 2 * flow_dim
            network_output_dict.update(
                {
                    "disappearing_logit": network_output[..., 0:1],
                    "static_logit": network_output[..., 1:2],
                    "dynamic_logit": network_output[..., 2:3],
                    "ground_logit": network_output[..., 3:4],
                    "static_flow": network_output[..., 4 : 4 + flow_dim],
                    "dynamic_flow": network_output[
                        ..., 4 + flow_dim : 4 + 2 * flow_dim
                    ],
                }
            )
            final_grid_size = network_output.shape[1:3]

        assert pointwise_voxel_coordinates_fs.shape[-1] == 2

        if "gt_label_based" in self.cfg.model.output_modification.values():
            assert ohe_gt_stat_dyn_ground_label_bev_map is not None
            assert ohe_gt_stat_dyn_ground_label_bev_map.shape[-3:-1] == final_grid_size
            assert ohe_gt_stat_dyn_ground_label_bev_map.shape[-1] == 3
            assert ohe_gt_stat_dyn_ground_label_bev_map.dtype == tf.bool

        # #region resize gt_flow_bev to same HW dimensions as output
        if gt_flow_bev is not None:
            gt_flow_bev = max_pool_2d_flow_map(
                gt_flow_bev[..., 0:2],
                (self.cfg.model.u_net.final_scale, self.cfg.model.u_net.final_scale),
            )
        # #endregion resize gt_flow_bev to same HW dimensions as output

        # #region precompute gt_static_flow
        bev_extent = np.array(self.cfg.data.bev_extent)
        net_output_shape = final_grid_size
        voxel_center_metric_coordinates = (
            np.stack(
                np.meshgrid(
                    np.arange(net_output_shape[0]),
                    np.arange(net_output_shape[1]),
                    indexing="ij",
                ),
                axis=-1,
            )
            + 0.5
        )
        voxel_center_metric_coordinates /= net_output_shape
        voxel_center_metric_coordinates *= bev_extent[2:] - bev_extent[:2]
        voxel_center_metric_coordinates += bev_extent[:2]
        homog_metric_voxel_center_coords = np.concatenate(
            [
                voxel_center_metric_coordinates,
                np.zeros_like(voxel_center_metric_coordinates[..., :1]),
                np.ones_like(voxel_center_metric_coordinates[..., :1]),
            ],
            axis=-1,
        )
        gt_static_flow = cast32(
            tf.einsum(
                "bij,hwj->bhwi",
                cast64(inv_odom[:, :2, :])
                - tf.eye(2, num_columns=4, dtype=tf.float64)[None],
                tf.constant(homog_metric_voxel_center_coords, dtype=tf.float64),
            )
        )
        # #endregion precompute gt_static_flow

        (
            network_output_dict,
            static_aggr_trafo,
            not_enough_points,
        ) = artificial_network_output(
            network_output_dict=network_output_dict,
            dynamicness_threshold=dynamicness_threshold,
            cfg=self.cfg,
            ohe_gt_stat_dyn_ground_label_bev_map=ohe_gt_stat_dyn_ground_label_bev_map,
            gt_flow_bev=gt_flow_bev,
            gt_static_flow=gt_static_flow,
            filled_pillar_mask=filled_pillar_mask,
            pc=pc,
            pointwise_voxel_coordinates_fs=pointwise_voxel_coordinates_fs,
            pointwise_valid_mask=pointwise_valid_mask,
            voxel_center_metric_coordinates=voxel_center_metric_coordinates,
            overwrite_non_filled_pillars_with_default_flow=overwrite_non_filled_pillars_with_default_flow,
            overwrite_non_filled_pillars_with_default_logits=overwrite_non_filled_pillars_with_default_logits,
        )

        with tf.name_scope("slice_output"):
            disappearing_logit = network_output_dict["disappearing_logit"][..., 0]
            disappearing = tf.nn.sigmoid(disappearing_logit)
            class_logits = network_output_dict["class_logits"]
            class_probs = network_output_dict["class_probs"]
            staticness = network_output_dict["staticness"]
            dynamicness = network_output_dict["dynamicness"]
            groundness = network_output_dict["groundness"]
            is_static = network_output_dict["is_static"]
            is_dynamic = network_output_dict["is_dynamic"]
            is_ground = network_output_dict["is_ground"]
            static_flow = network_output_dict["static_flow"]
            static_aggr_flow = network_output_dict["static_aggr_flow"]
            dynamic_flow = network_output_dict["dynamic_flow"]
            dynamic_aggr_flow = network_output_dict.get("dynamic_aggr_flow", None)
            masked_dynamic_aggr_flow = network_output_dict.get(
                "masked_dynamic_aggr_flow", None
            )
            masked_static_aggr_flow = network_output_dict["masked_static_aggr_flow"]
            if flow_dim == 2:
                dynamic_flow = tf.concat(
                    [dynamic_flow, tf.zeros_like(dynamic_flow[..., :1])], axis=-1
                )
                static_flow = tf.concat(
                    [static_flow, tf.zeros_like(static_flow[..., :1])], axis=-1
                )
                static_aggr_flow = tf.concat(
                    [static_aggr_flow, tf.zeros_like(static_aggr_flow[..., :1])],
                    axis=-1,
                )
                masked_static_aggr_flow = tf.concat(
                    [
                        masked_static_aggr_flow,
                        tf.zeros_like(masked_static_aggr_flow[..., :1]),
                    ],
                    axis=-1,
                )
                if dynamic_aggr_flow is not None:
                    dynamic_aggr_flow = tf.concat(
                        [
                            dynamic_aggr_flow,
                            tf.zeros_like(dynamic_aggr_flow[..., :1]),
                        ],
                        axis=-1,
                    )
                if masked_dynamic_aggr_flow is not None:
                    masked_dynamic_aggr_flow = tf.concat(
                        [
                            masked_dynamic_aggr_flow,
                            tf.zeros_like(masked_dynamic_aggr_flow[..., :1]),
                        ],
                        axis=-1,
                    )

            if self.cfg.model.use_static_aggr_flow_for_aggr_flow:
                static_flow_for_aggr = masked_static_aggr_flow
            else:
                static_flow_for_aggr = static_flow

            assert len(is_static.shape) == 3
            assert len(groundness.shape) == 3
            if dynamic_flow_is_non_rigid_flow:
                aggregated_flow = tf.where(
                    tf.tile(
                        is_static[..., None],
                        [1, 1, 1, static_flow_for_aggr.shape[-1]],
                    ),
                    static_flow_for_aggr,
                    (static_flow_for_aggr + dynamic_flow)
                    * (1.0 - groundness[..., None]),
                )
            else:
                aggregated_flow = tf.where(
                    tf.tile(
                        is_static[..., None],
                        [1, 1, 1, static_flow_for_aggr.shape[-1]],
                    ),
                    static_flow_for_aggr,
                    dynamic_flow * (1.0 - groundness[..., None]),
                )
            if (
                self.cfg.model.use_dynamic_aggr_flow_for_aggr_flow
                and dynamic_aggr_flow is not None
                and "mask_has_dynamic_aggr_output" in network_output_dict.keys()
            ):
                aggregated_flow = tf.where(
                    network_output_dict["mask_has_dynamic_aggr_output"],
                    dynamic_aggr_flow,
                    aggregated_flow,
                )
            # now we have:
            # disappearing, disappearing_logit
            # class_probs, class_logits, is_static, is_dynamic, is_ground
            # dynamic_flow, static_flow, aggregated_flow
        modified_output_bev_img = AttrDict(
            disappearing=disappearing,
            disappearing_logit=disappearing_logit,
            class_probs=class_probs,
            class_logits=class_logits,
            staticness=staticness,
            dynamicness=dynamicness,
            groundness=groundness,
            is_static=is_static,
            is_dynamic=is_dynamic,
            is_ground=is_ground,
            dynamic_flow=dynamic_flow,
            static_flow=static_flow,
            aggregated_flow=aggregated_flow,
            static_aggr_flow=static_aggr_flow,
            dynamic_aggr_flow=dynamic_aggr_flow,
        )
        return (
            modified_output_bev_img,
            network_output_dict,
            gt_flow_bev,
            static_aggr_trafo,
        )

    def apply_flow_to_points(
        self,
        *,
        modified_output_bev_img,
        pointwise_voxel_coordinates_fs,
        pointwise_valid_mask,
    ):
        with tf.name_scope("apply_flow2points"):
            concat_bool_vals = tf.stack(
                [
                    modified_output_bev_img.is_static,
                    modified_output_bev_img.is_dynamic,
                    modified_output_bev_img.is_ground,
                ],
                axis=-1,
            )
            concat_flt_vals = tf.stack(
                [
                    modified_output_bev_img.disappearing,
                    modified_output_bev_img.disappearing_logit,
                    modified_output_bev_img.staticness,
                    modified_output_bev_img.dynamicness,
                    modified_output_bev_img.groundness,
                ],
                axis=-1,
            )
            concat_flt_vals = tf.concat(
                [
                    concat_flt_vals,
                    modified_output_bev_img.class_probs,
                    modified_output_bev_img.class_logits,
                    modified_output_bev_img.dynamic_flow,
                    modified_output_bev_img.static_flow,
                    modified_output_bev_img.aggregated_flow,
                    modified_output_bev_img.static_aggr_flow,
                ],
                axis=-1,
            )
            if modified_output_bev_img.dynamic_aggr_flow is not None:
                concat_flt_vals = tf.concat(
                    [concat_flt_vals, modified_output_bev_img.dynamic_aggr_flow],
                    axis=-1,
                )
                num_required_concat_vals = 26
            else:
                num_required_concat_vals = 23

            tf.Assert(
                tf.reduce_all(pointwise_voxel_coordinates_fs >= 0),
                data=["negative pixel coordinates found"],
            )
            tf.Assert(
                tf.reduce_all(
                    pointwise_voxel_coordinates_fs < tf.shape(concat_bool_vals)[1:3]
                ),
                data=["too large pixel coordinates found"],
            )
            tf.Assert(
                tf.reduce_all(
                    pointwise_voxel_coordinates_fs < tf.shape(concat_flt_vals)[1:3]
                ),
                data=["too large pixel coordinates found"],
            )
            assert tf.__version__[0] == "2"
            pointwise_concat_bool_vals = tf.gather_nd(
                concat_bool_vals, pointwise_voxel_coordinates_fs, batch_dims=1
            )
            pointwise_concat_flt_vals = tf.gather_nd(
                concat_flt_vals, pointwise_voxel_coordinates_fs, batch_dims=1
            )
            pointwise_concat_bool_vals = tf.where(
                tf.tile(
                    pointwise_valid_mask[..., None],
                    [1, 1, pointwise_concat_bool_vals.shape[-1]],
                ),
                pointwise_concat_bool_vals,
                tf.zeros_like(pointwise_concat_bool_vals),
            )
            pointwise_concat_flt_vals = tf.where(
                tf.tile(
                    pointwise_valid_mask[..., None],
                    [1, 1, pointwise_concat_flt_vals.shape[-1]],
                ),
                pointwise_concat_flt_vals,
                np.nan * tf.ones_like(pointwise_concat_flt_vals),
            )

            assert (
                pointwise_concat_bool_vals.shape[-1] == 3
            ), pointwise_concat_bool_vals.shape
            pointwise_is_static = pointwise_concat_bool_vals[..., 0]
            pointwise_is_dynamic = pointwise_concat_bool_vals[..., 1]
            pointwise_is_ground = pointwise_concat_bool_vals[..., 2]
            assert (
                pointwise_concat_flt_vals.shape[-1] == num_required_concat_vals
            ), pointwise_concat_flt_vals.shape
            pointwise_disappearing = pointwise_concat_flt_vals[..., 0]
            pointwise_disappearing_logit = pointwise_concat_flt_vals[..., 1]
            pointwise_staticness = pointwise_concat_flt_vals[..., 2]
            pointwise_dynamicness = pointwise_concat_flt_vals[..., 3]
            pointwise_groundness = pointwise_concat_flt_vals[..., 4]
            pointwise_class_probs = pointwise_concat_flt_vals[..., 5:8]
            pointwise_class_logits = pointwise_concat_flt_vals[..., 8:11]
            pointwise_dynamic_flow = pointwise_concat_flt_vals[..., 11:14]
            pointwise_static_flow = pointwise_concat_flt_vals[..., 14:17]
            pointwise_aggregated_flow = pointwise_concat_flt_vals[..., 17:20]
            pointwise_static_aggregated_flow = pointwise_concat_flt_vals[..., 20:23]
            if modified_output_bev_img.dynamic_aggr_flow is not None:
                pointwise_dynamic_aggregated_flow = pointwise_concat_flt_vals[
                    ..., 23:26
                ]
            else:
                pointwise_dynamic_aggregated_flow = None
            retval = AttrDict(
                disappearing_logit=pointwise_disappearing_logit,
                disappearing=pointwise_disappearing,
                class_logits=pointwise_class_logits,
                class_probs=pointwise_class_probs,
                staticness=pointwise_staticness,
                dynamicness=pointwise_dynamicness,
                groundness=pointwise_groundness,
                is_static=pointwise_is_static,
                is_dynamic=pointwise_is_dynamic,
                is_ground=pointwise_is_ground,
                dynamic_flow=pointwise_dynamic_flow,
                static_flow=pointwise_static_flow,
                aggregated_flow=pointwise_aggregated_flow,
                static_aggr_flow=pointwise_static_aggregated_flow,
                dynamic_aggr_flow=pointwise_dynamic_aggregated_flow,
            )
            return retval

    def call(
        self,
        network_output,
        dynamicness_threshold,
        *,
        pc,
        pointwise_voxel_coordinates,
        pointwise_valid_mask,
        filled_pillar_mask,
        odom,
        inv_odom,
        summaries,
        gt_flow_bev=None,
        ohe_gt_stat_dyn_ground_label_bev_map=None,
        dynamic_flow_is_non_rigid_flow=False,
    ):
        pointwise_voxel_coordinates_fs = (
            pointwise_voxel_coordinates // self.cfg.model.u_net.final_scale
        )
        assert pointwise_voxel_coordinates_fs.shape[-1] == 2

        (
            modified_output_bev_img,
            network_output_dict,
            gt_flow_bev,
            static_aggr_trafo,
        ) = self.apply_output_modification(
            network_output,
            dynamicness_threshold,
            pc=pc,
            pointwise_voxel_coordinates_fs=pointwise_voxel_coordinates_fs,
            pointwise_valid_mask=pointwise_valid_mask,
            filled_pillar_mask=filled_pillar_mask,
            inv_odom=inv_odom,
            gt_flow_bev=gt_flow_bev,
            ohe_gt_stat_dyn_ground_label_bev_map=ohe_gt_stat_dyn_ground_label_bev_map,
            dynamic_flow_is_non_rigid_flow=dynamic_flow_is_non_rigid_flow,
        )

        pointwise_output = self.apply_flow_to_points(
            modified_output_bev_img=modified_output_bev_img,
            pointwise_voxel_coordinates_fs=pointwise_voxel_coordinates_fs,
            pointwise_valid_mask=pointwise_valid_mask,
        )

        retval = self.output_decoder_summaries(
            network_output_dict=network_output_dict,
            pointwise_output=pointwise_output,
            modified_output_bev_img=modified_output_bev_img,
            filled_pillar_mask=filled_pillar_mask,
        )

        retval["static_aggr_trafo"] = static_aggr_trafo
        retval["dynamicness_threshold"] = dynamicness_threshold
        return retval

    def output_decoder_summaries(
        self,
        *,
        network_output_dict,
        pointwise_output,
        modified_output_bev_img,
        filled_pillar_mask,
    ):
        assert (
            3
            == len(filled_pillar_mask.shape)
            == len(network_output_dict["is_static"].shape)
        ), (filled_pillar_mask.shape, network_output_dict["is_static"].shape)
        assert (
            filled_pillar_mask.shape[-2:].as_list()
            == network_output_dict["is_static"].shape[-2:].as_list()
        ), (filled_pillar_mask.shape, network_output_dict["is_static"].shape)

        return AttrDict(
            **pointwise_output,
            dense_maps=AttrDict(
                aggregated_flow=modified_output_bev_img.aggregated_flow,
            ),
            modified_network_output=AttrDict(network_output_dict),
        )
