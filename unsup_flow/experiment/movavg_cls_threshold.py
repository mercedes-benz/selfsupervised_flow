#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


from typing import Tuple

import tensorflow as tf


class MovingAverageThreshold(tf.keras.Model):
    def __init__(
        self,
        unsupervised: bool,
        num_train_samples,
        num_moving,
        num_still=None,
        resolution: int = 100000,
        start_value: float = 0.5,
        value_range: Tuple[float, float] = (0.0, 1.0),
        *args,
        **kwargs,
    ):
        super().__init__(*args, autocast=False, **kwargs)
        self.value_range = (value_range[0], value_range[1] - value_range[0])
        self.resolution = resolution
        self.num_moving = num_moving
        assert unsupervised == (num_still is None), (
            "training unsupervised requires num_still to be set to None, "
            "supervised requires num_still to be set"
        )
        self.num_still = num_still
        self.start_value = tf.constant(start_value, dtype=tf.float32)
        self.total = num_moving
        if num_still is not None:
            self.total += num_still
        assert num_train_samples > 0, num_train_samples
        avg_points_per_sample = self.total / num_train_samples
        self.update_weight = 1.0 / min(
            2.0 * self.total, 5_000.0 * avg_points_per_sample
        )  # update buffer roughly every 5k iterations, so 5k * points per sample for denominator
        self.update_weight = tf.constant(self.update_weight, dtype=tf.float64)

        if num_still is not None:
            self.moving_counter = self.add_weight(
                "moving_counter",
                shape=(),
                dtype=tf.int64,
                initializer=tf.keras.initializers.Constant(self.num_moving),
                trainable=False,
            )
            self.still_counter = self.add_weight(
                "still_counter",
                shape=(),
                dtype=tf.int64,
                initializer=tf.keras.initializers.Constant(self.num_still),
                trainable=False,
            )

        self.bias_counter = self.add_weight(
            "bias_counter",
            shape=(),
            dtype=tf.float64,
            initializer="zeros",
            trainable=False,
        )
        self.moving_average_importance = self.add_weight(
            "value_buffer",
            shape=(self.resolution,),
            dtype=tf.float32,
            initializer="zeros",
            trainable=False,
        )

    def value(self):
        return tf.where(
            self.bias_counter > 0.0,
            self._compute_optimal_score_threshold(),
            self.start_value,
        )

    def _compute_bin_idxs(self, dynamicness_scores):
        idxs = tf.cast(
            self.resolution
            * (dynamicness_scores - self.value_range[0])
            / self.value_range[1],
            tf.int32,
        )
        tf.debugging.assert_less_equal(idxs, self.resolution)
        tf.debugging.assert_greater_equal(idxs, 0)
        idxs = tf.minimum(idxs, self.resolution - 1)
        tf.debugging.assert_less(idxs, self.resolution)
        return idxs

    def _compute_improvements(
        self,
        epes_stat_flow,
        epes_dyn_flow,
        moving_mask,
    ):
        if self.num_still is None:
            assert moving_mask is None
            improvements = epes_stat_flow - epes_dyn_flow
        else:
            assert moving_mask is not None
            assert len(moving_mask.shape) == 1
            improvement_weight = tf.constant(1.0, tf.float32) / tf.cast(
                tf.where(moving_mask, self.moving_counter, self.still_counter),
                tf.float32,
            )
            improvements = (epes_stat_flow - epes_dyn_flow) * improvement_weight
        return improvements

    def _compute_optimal_score_threshold(self):
        improv_over_thresh = tf.concat(
            [[0], tf.cumsum(self.moving_average_importance)], axis=0
        )
        best_improv = tf.reduce_min(improv_over_thresh)
        avg_optimal_idx = tf.reduce_mean(
            tf.cast(tf.where(tf.equal(best_improv, improv_over_thresh)), tf.float32)
        )
        optimal_score_threshold = (
            self.value_range[0]
            + avg_optimal_idx * self.value_range[1] / self.resolution
        )
        return optimal_score_threshold

    def _update_values(self, cur_value, cur_weight):
        cur_update_weight = (1.0 - self.update_weight) ** tf.cast(
            cur_weight, tf.float64
        )
        self.moving_average_importance.assign(
            self.moving_average_importance * tf.cast(cur_update_weight, tf.float32)
        )
        self.moving_average_importance.assign_add(
            tf.cast(1.0 - cur_update_weight, tf.float32) * cur_value
        )
        self.bias_counter.assign(self.bias_counter * cur_update_weight)
        self.bias_counter.assign_add(1.0 - cur_update_weight)

    def update(
        self,
        epes_stat_flow,
        epes_dyn_flow,
        moving_mask,
        dynamicness_scores,
        summaries,
        training,
    ):
        assert isinstance(training, bool)
        if training:
            assert len(epes_stat_flow.shape) == 1
            assert len(epes_dyn_flow.shape) == 1
            assert len(dynamicness_scores.shape) == 1
            improvements = self._compute_improvements(
                epes_stat_flow,
                epes_dyn_flow,
                moving_mask,
            )
            bin_idxs = self._compute_bin_idxs(dynamicness_scores)
            cur_result = tf.scatter_nd(
                bin_idxs[:, None], improvements, shape=(self.resolution,)
            )
            self._update_values(cur_result, tf.size(epes_stat_flow))
            if self.num_still is not None:
                self.moving_counter.assign_add(tf.math.count_nonzero(moving_mask))
                self.still_counter.assign_add(tf.math.count_nonzero(~moving_mask))
            result = self.value()
            if summaries["metrics_eval"]:
                tf.summary.scalar("dynamicness_threshold", result)
                tf.summary.scalar("dynamicness_update_amount", self.bias_counter)
                if self.num_still is not None:
                    tf.summary.scalar(
                        "moving_percentage",
                        tf.cast(self.moving_counter, tf.float32)
                        / tf.cast(self.moving_counter + self.still_counter, tf.float32),
                    )
            return result
        return self.value()


if __name__ == "__main__":
    dynamicness = tf.constant([0.1, 0.2, 0.4, 0.5, 0.6, 0.8])
    epes_stat_flow = tf.constant([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    epes_dyn_flow = tf.constant([0.6, 0.4, 0.0, 0.8, 0.4, 0.0])
    threshold_layer = MovingAverageThreshold(
        unsupervised=True,
        num_train_samples=2,
        num_moving=6091776000 // 3,
        num_still=6091776000 * 2 // 3,
    )
    # threshold_layer = MovingAverageThreshold(4, 8)
    for _i in range(10):
        print(threshold_layer.value())
        opt_thresh = threshold_layer.update(
            epes_stat_flow,
            epes_dyn_flow,
            tf.constant([True, False, True, False, False, False], dtype=tf.bool),
            dynamicness,
            {"metrics_eval": True},
            training=True,
        )
        print(opt_thresh)
        opt_thresh = threshold_layer.update(
            epes_stat_flow,
            epes_stat_flow + 1,
            tf.constant([True, False, True, False, False, False], dtype=tf.bool),
            dynamicness,
            {"metrics_eval": True},
            training=True,
        )
        print(opt_thresh)
        opt_thresh = threshold_layer.update(
            epes_stat_flow,
            epes_stat_flow - 1,
            tf.constant([True, False, True, False, False, False], dtype=tf.bool),
            dynamicness,
            {"metrics_eval": True},
            training=False,
        )
        print(opt_thresh)
