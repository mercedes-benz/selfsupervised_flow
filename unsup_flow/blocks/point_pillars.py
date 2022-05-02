#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


from typing import Tuple

import numpy as np
import tensorflow as tf


class PointPillarsLayer(tf.keras.layers.Layer):
    def __init__(self, cfg, bn_kwargs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        # #region default values
        self.cfg.set_default("nbr_point_feats", 64)
        self.cfg.set_default("inf_distance", 1000.0)
        self.cfg.set_default("nbr_pillars", (640, 640))
        self.cfg.set_default("max_points_per_pillar", 32)
        self.cfg.set_default(
            "point_feat_mask",
            (False, False, True, True, True, True, True, True, True, True, False),
            # boolean mask for 11 features [
            #     (x y z)/(L W H) r
            #     (x_rel_mean y_rel_mean z_rel_mean)/(voxel size) r_rel_mean
            #     (x_rel_center y_rel_center z_rel_center)/(voxel size)
            # ]
        )
        self.cfg.set_default(
            "voxel_feat_mask",
            (False, False, True, True, True, True, False, False, False, False, True),
            # boolean mask for 11 features [
            #     (x_mean y_mean z_mean)/(L W H) r_mean
            #     (x_mean_rel_center y_mean_rel_center z_mean_rel_center)
            #     /(voxel size)
            #     (x_center y_center z_center)/(L W H)
            #     nbr_valid_points/max_nbr_points
            # ]
        )
        # #endregion default values
        # #region create layers
        self.dense_layer = tf.keras.layers.Dense(
            units=self.cfg.nbr_point_feats, use_bias=False
        )
        self.batch_norm_layer = tf.keras.layers.BatchNormalization(
            center=False, scale=False, **bn_kwargs
        )

    @staticmethod
    def get_voxel_config(
        *,
        inf_distance: float,
        bev_extent: Tuple[float, float, float, float],
        nbr_pillars: Tuple[int, int],
        max_points_per_pillar: int,
    ):
        # construct voxel op config from point pillars config
        voxel_cfg = {
            "extent": [
                *bev_extent[0:2],
                -inf_distance,
                *bev_extent[2:4],
                inf_distance,
            ],
            "resolution": [*nbr_pillars, 1],
            "max_points_per_voxel": max_points_per_pillar,
        }
        return voxel_cfg

    def build(self, input_shape):
        # #region create layers
        # self.dense_layer.build(sum(self.cfg.point_feat_mask))
        # self.batch_norm_layer.build([None] + [self.cfg.nbr_point_feats])

        self.beta_offset = self.add_weight(
            "bn_beta_var",
            shape=(1, self.cfg.nbr_point_feats + sum(self.cfg.voxel_feat_mask)),
            initializer=tf.zeros_initializer(),
            # dtype=tf.float32,
            trainable=True,
        )
        self.gamma_scale = self.add_weight(
            "bn_gamma_var",
            shape=(1, self.cfg.nbr_point_feats + sum(self.cfg.voxel_feat_mask)),
            initializer=tf.ones_initializer(),
            # dtype=tf.float32,
            trainable=True,
        )
        # #endregion create layers
        super().build(input_shape)

    def call(self, inputs, training=None):
        # p_feats is a ragged tensor that makes problems
        v_feats, p_feats_row_splits, p_feats_values, v_coors, v_count, nbr_pcls = inputs

        p_feats = tf.RaggedTensor.from_row_splits(
            values=p_feats_values, row_splits=p_feats_row_splits
        )

        self.v_feats, self.p_feats, self.v_coors, self.v_count, self.nbr_pcls = (
            v_feats,
            p_feats,
            v_coors,
            v_count,
            nbr_pcls,
        )
        # #region apply feature masks
        v_feats_masked = tf.boolean_mask(
            tensor=v_feats,
            mask=self.cfg.voxel_feat_mask,
            name="boolean_mask_voxel_features",
            axis=1,
        )
        v_feats_masked.set_shape(
            list(v_feats_masked.shape[:-1]) + [sum(self.cfg.voxel_feat_mask)]
        )
        p_feats_masked = p_feats.with_flat_values(
            tf.boolean_mask(
                tensor=p_feats.flat_values,
                mask=self.cfg.point_feat_mask,
                name="boolean_mask_point_features",
                axis=1,
            )
        )
        p_feats_masked.values.set_shape(
            list(p_feats_masked.values.shape[:-1]) + [sum(self.cfg.point_feat_mask)]
        )
        # #endregion apply feature masks
        p_feats_masked = p_feats_masked.with_flat_values(
            self.dense_layer(p_feats_masked.flat_values)
        )
        # it is more efficient to scale and center with beta and gamma
        # after max pooling, while retaining same results
        # => introducing non-trainable bn here
        p_feats_masked = p_feats_masked.with_flat_values(
            self.batch_norm_layer(p_feats_masked.flat_values, training=training)
        )
        # NOTE: the following workaround with `to_tensor` is filed on github
        # under https://github.com/tensorflow/tensorflow/issues/35802
        p_feats_pooled = tf.reduce_max(
            p_feats_masked.to_tensor(default_value=np.nan),
            axis=1,
            keepdims=False,
            name="max_pool_over_points",
        )
        all_v_feats = tf.concat(
            values=[p_feats_pooled, v_feats_masked], axis=-1, name="concat_feats"
        )
        all_v_feats = all_v_feats * self.gamma_scale + self.beta_offset
        bev_features = tf.scatter_nd(
            indices=v_coors[:, :3],  # remove z coordinate for pillars
            updates=all_v_feats,
            shape=[
                nbr_pcls,
                *self.cfg.nbr_pillars,
                self.cfg.nbr_point_feats + sum(self.cfg.voxel_feat_mask),
            ],
            name="scatter_feats",
        )
        return bev_features
