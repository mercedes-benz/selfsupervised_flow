#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import itertools as it
import os
import os.path as osp
import sys
import typing as t
from time import time

import numpy as np
import tensorflow as tf
from flatten_dict import flatten, unflatten
from tqdm import tqdm

from cfgattrdict import convert_camel2snake, get_current_cpu_memory_usage, is_power_of_2
from labelmap import get_label_map_from_file
from unsup_flow.blocks.point_pillars import PointPillarsLayer
from unsup_flow.experiment.losses import (
    compute_custom_class_iou_metrics,
    new_compute_custom_conf_mat,
)
from unsup_flow.experiment.raft.raft_model import OurPillarModel
from unsup_flow.layers.schedules import step_decay, warm_up
from unsup_flow.losses.static_points_loss import trafo_distance
from unsup_flow.tf import rank, relaxed_tf_function  # , shape
from usfl_io import tcolor
from usfl_io.io_tools import get_unsupervised_flow_dataset


def turn_list_of_dicts_to_dict_of_lists(list_of_dicts):

    result = {k: [] for k in flatten(list_of_dicts[0]).keys()}
    for curdict in list_of_dicts:
        flatdict = flatten(curdict)
        assert flatdict.keys() == result.keys()
        for k in flatdict:
            result[k].append(flatdict[k])
    return unflatten(result)


def print_on_fail(value):
    def decorator(func):
        def wrapped(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
            except Exception:  # noqa:B902
                print(value)
                raise
            return result

        return wrapped

    return decorator


class TBFactory:
    noop_writer = tf.summary.create_noop_writer()

    def __init__(self, base_path: str):
        super().__init__()
        self.base_path = base_path
        self._tb_writers: t.Dict[t.Optional[str], t.Any] = {None: self.noop_writer}

    def __call__(self, tb_name: str = None):
        if tb_name not in self._tb_writers:
            assert isinstance(tb_name, str)
            assert len(tb_name) >= 1
            self._tb_writers[tb_name] = tf.summary.create_file_writer(
                osp.join(self.base_path, tb_name)
            )
        return self._tb_writers[tb_name]


class UnsupervisedFlow:
    version = "v0.5.0"

    def __init__(self, cfg):
        self.cfg = cfg
        self.initial_cfg_check()
        self.set_version()

    def initial_cfg_check(self):
        # print(self.cfg.model.output_modification)
        if "supervised" in self.cfg.losses:
            if self.cfg.losses.supervised.mode == "total_flow":
                assert self.cfg.model.output_modification.disappearing_logit is False
                assert self.cfg.model.output_modification.static_logit is True
                assert self.cfg.model.output_modification.dynamic_logit is False
                assert self.cfg.model.output_modification.ground_logit is False
                assert self.cfg.model.output_modification.static_flow == "net"
                assert self.cfg.model.output_modification.dynamic_flow == "zero"

        assert "pretrain" in self.cfg.iterations
        if set(self.cfg.phases.keys()) == {"train"}:
            assert self.cfg.iterations.pretrain == 0
        elif set(self.cfg.phases.keys()) == {"train", "pretrain"}:
            assert self.cfg.iterations.pretrain > 0
        else:
            raise ValueError(
                "unknown combination of training phases: %s" % self.cfg.phases.keys()
            )

    def set_version(self):
        if hasattr(self, "version"):
            if "version" in self.cfg:
                self.version = self.cfg.version + "_" + self.version
                del self.cfg["version"]
        elif "version" in self.cfg:
            self.version = self.cfg.version
            del self.cfg["version"]
        else:
            self.version = "unversioned"

    def setup_experiment_path(self):
        self.cfg.set_default("debug")
        self.cfg.debug.set_default("prod", False)
        self.cfg.debug.set_default("profile", False)
        self.experiment_path = os.getenv("OUTPUT_DATADIR", "/tmp")
        if not self.cfg.debug.prod:
            self.experiment_path = osp.join(self.experiment_path, "test")
        self.experiment_path = osp.join(
            self.experiment_path,
            (convert_camel2snake(self.__class__.__name__) + "_" + self.version),
        )
        self.experiment_path = self.cfg.initialize_hashed_path(
            self.experiment_path, exist_ok=True, timestamp=True, verbose="Experiment"
        )
        # self.cfg.debug.dump_train_spec(self.experiment_path + ".cfg")

    def prepare(self):
        print("START PREPARING EXPERIMENT!")
        self.global_step = tf.Variable(1, trainable=False, dtype=tf.int64)

        carla_labelmap = get_label_map_from_file(
            "carla", "static_dynamic_ground", "static_dynamic_ground"
        )
        if self.cfg.data.name == "carla" or self.cfg.data.name == "kitti_lidar_raw":
            # this is necessary so that eval of kitti_lidar_raw on kitti_lidar_sf works
            self.label_mapping = carla_labelmap
        elif self.cfg.data.name == "nuscenes":
            self.label_mapping = get_label_map_from_file(
                "nuscenes",
                "nuscenes2static_dynamic_ground",
                "nuscenes_static_dynamic_ground",
            )
        else:
            raise NotImplementedError(
                "Unknown cfg.data.name {0}".format(self.cfg.data.name)
            )
        if self.cfg.optimizer == "adam":
            print("Using Adam Optimizer")
            self.optimizer = tf.optimizers.Adam(self.cfg.learning_rate.initial)
        elif self.cfg.optimizer == "rmsprop":
            print("Using RMSProp Optimizer")
            self.optimizer = tf.optimizers.RMSprop(self.cfg.learning_rate.initial)
        else:
            raise AssertionError("Please select optimizer adam or rmsprop")

        self.metrics_label_dict = {
            key: tf.constant(idx, tf.int32)
            for idx, key in enumerate(self.label_mapping.mnames)
        }

        # this is necessary so that eval of kitti_lidar_raw on kitti_lidar_sf works
        train_ds_lm = (
            None if self.cfg.data.name == "kitti_lidar_raw" else self.label_mapping
        )
        ignore_contradicting_flow_semseg = self.cfg.data.name == "carla"
        pp_voxel_cfg = PointPillarsLayer.get_voxel_config(
            bev_extent=self.cfg.data.bev_extent,
            **{
                k: v
                for k, v in self.cfg.model.point_pillars.items()
                if k in {"nbr_pillars", "inf_distance", "max_points_per_pillar"}
            },
        )
        self.train_dataset, train_fnames = get_unsupervised_flow_dataset(
            cfg=self.cfg,
            labelmap=train_ds_lm,
            voxel_cfg=pp_voxel_cfg,
            data_source=self.cfg.data.name,
            ignore_contradicting_flow_semseg=ignore_contradicting_flow_semseg,
            data_dir=os.path.join(
                os.getenv("INPUT_DATADIR", "INPUT_DATADIR_ENV_NOT_DEFINED"),
                "prepped_datasets",
                self.cfg.data.name,
            ),
        )
        print("Working with ", len(train_fnames), " train files")

        if self.cfg.data.name == "kitti_lidar_raw":
            self.valid_dataset, _ = get_unsupervised_flow_dataset(
                cfg=self.cfg,
                labelmap=carla_labelmap,
                voxel_cfg=pp_voxel_cfg,
                valid=True,
                data_source="kitti_stereo_sf",
                data_dir=os.path.join(
                    os.getenv("INPUT_DATADIR", "INPUT_DATADIR_ENV_NOT_DEFINED"),
                    "prepped_datasets",
                    "kitti_stereo_sf",
                ),
            )
        else:
            self.valid_dataset, val_fnames = get_unsupervised_flow_dataset(
                cfg=self.cfg,
                labelmap=self.label_mapping,
                voxel_cfg=pp_voxel_cfg,
                valid=True,
                data_source=self.cfg.data.name,
                ignore_contradicting_flow_semseg=ignore_contradicting_flow_semseg,
                data_dir=os.path.join(
                    os.getenv("INPUT_DATADIR", "INPUT_DATADIR_ENV_NOT_DEFINED"),
                    "prepped_datasets",
                    self.cfg.data.name,
                ),
            )
            print("Working with ", len(val_fnames), " val files")
        self.valid_dataset_kitti_stereo_sf, _ = get_unsupervised_flow_dataset(
            cfg=self.cfg,
            labelmap=carla_labelmap,
            voxel_cfg=pp_voxel_cfg,
            valid=True,
            data_source="kitti_stereo_sf",
            data_dir=os.path.join(
                os.getenv("INPUT_DATADIR", "INPUT_DATADIR_ENV_NOT_DEFINED"),
                "prepped_datasets",
                "kitti_stereo_sf",
            ),
        )
        self.valid_dataset_kitti_stereo_sf_8k, _ = get_unsupervised_flow_dataset(
            cfg=self.cfg,
            labelmap=carla_labelmap,
            voxel_cfg=pp_voxel_cfg,
            valid=True,
            data_source="kitti_stereo_sf",
            num_input_points=8192,
            data_dir=os.path.join(
                os.getenv("INPUT_DATADIR", "INPUT_DATADIR_ENV_NOT_DEFINED"),
                "prepped_datasets",
                "kitti_stereo_sf",
            ),
        )
        self.datasets = {
            "train": self.train_dataset,
            "valid": self.valid_dataset,
            "valid_kitti_stereo_sf": self.valid_dataset_kitti_stereo_sf,
            "valid_kitti_stereo_sf_8k": self.valid_dataset_kitti_stereo_sf_8k,
        }

        self.update_num_mov_and_still_points_based_on_num_input_points_and_train_samples()
        self.model = OurPillarModel(cfg=self.cfg, label_dict=self.metrics_label_dict)

        if self.cfg.set_default(["debug", "eager"], False):
            self.train_one_step = train_one_step_eager
            self.valid_one_step = valid_one_step_eager
        else:
            self.train_one_step = train_one_step
            self.valid_one_step = valid_one_step

        self.setup_experiment_path()

        ckpt = tf.train.Checkpoint(
            step=self.global_step,
            optimizer=self.optimizer,
            net=self.model,
        )
        self.chkpt_manager = tf.train.CheckpointManager(
            ckpt,
            osp.join(self.experiment_path, "checkpoints", "train_checkpointmanager"),
            max_to_keep=20,
        )
        print("FINISHED PREPARING EXPERIMENT!")

    def update_num_mov_and_still_points_based_on_num_input_points_and_train_samples(
        self,
    ):
        num_moving = self.cfg.data.train.num_moving_points
        num_still = self.cfg.data.train.num_still_points
        nbr_train_samples = self.cfg.data.nbr_samples.train
        if self.cfg.data.name == "kitti_lidar_raw":
            nbr_train_samples = self.cfg.data.nbr_samples.kitti_lidar_raw
        if self.cfg.data.num_input_points > 0:
            if num_still is None:
                num_moving = nbr_train_samples * self.cfg.data.num_input_points * 2
            else:
                ratio = num_moving / (num_moving + num_still) * 2
                num_moving = nbr_train_samples * self.cfg.data.num_input_points * ratio
                num_still = (
                    nbr_train_samples * self.cfg.data.num_input_points * (2.0 - ratio)
                )
        self.cfg.data.train.num_moving_points = num_moving
        self.cfg.data.train.num_still_points = num_still

    def run(self):
        print("START RUNNING EXPERIMENT!")
        tb_factory = TBFactory(osp.join(self.experiment_path, "tb"))

        for phase, el in zip(
            ["pretrain"] * self.cfg.iterations.pretrain
            + ["train"] * self.cfg.iterations.train,
            it.chain(
                *[
                    self.datasets[self.cfg.phases[phase].dataset].take(
                        self.cfg.iterations[phase]
                    )
                    for phase in ["pretrain", "train"]
                    if phase in self.cfg.phases
                ]
            ),
        ):
            t_start_iter = time()
            tf.summary.experimental.set_step(self.global_step)
            assert el["pcl_t0"]["pc"].shape[1] > 3
            assert el["pcl_t1"]["pc"].shape[1] > 3

            with tb_factory(phase).as_default():
                if self.cfg.set_default(["debug", "profile"], False):
                    tf.summary.trace_on(graph=True, profiler=True)
            t0 = time()
            _ = print_on_fail(el["name"])(self.train_one_step)(
                el,
                self.global_step,
                cfg=self.cfg,
                optimizer=self.optimizer,
                model=self.model,
                summaries={
                    "writer": tb_factory(phase),
                    "imgs_eval": self.global_step.numpy()
                    % self.cfg.iterations.eval_every
                    == 0,
                    "metrics_eval": self.global_step.numpy()
                    % self.cfg.iterations.train_metrics_every
                    == 0,
                    "aggregated_metrics": False,
                    "metrics_label_dict": self.metrics_label_dict,
                    "label_mapping": self.label_mapping,
                },
                cur_train_mode=self.cfg.phases[phase].mode,
            )
            time_per_train_step = time() - t0
            with tb_factory(phase).as_default():
                if self.cfg.set_default(["debug", "profile"], False):
                    tf.summary.trace_export(
                        name="model_trace",
                        step=self.global_step,
                        profiler_outdir=osp.join(self.experiment_path, "tb"),
                    )
                tf.summary.scalar("time_per_train_step", time_per_train_step)

            if (
                self.global_step
                % self.cfg.iterations.set_default("full_eval_every", 10_000)
                == 0
            ):
                self.save_model2disk()

                self.run_complete_validation(
                    self.valid_dataset,
                    tb_factory("%s_valid" % phase),
                    tb_factory(),
                )
                self.run_complete_validation(
                    self.valid_dataset_kitti_stereo_sf,
                    tb_factory("%s_valid_kitti_stereo_sf" % phase),
                    tb_factory(),
                )
                self.run_complete_validation(
                    self.valid_dataset_kitti_stereo_sf_8k,
                    tb_factory("%s_valid_kitti_stereo_sf_8k" % phase),
                    tb_factory(),
                )

            with tb_factory(phase).as_default():
                if self.global_step == 1:
                    tf.summary.text(
                        "cfg",
                        "Configuration-Hash: %s\n\nCommand: `$ %s`\n\n    %s"
                        % (
                            self.cfg.get_hash_value(),
                            osp.abspath(sys.argv[0]) + " " + " ".join(sys.argv[1:]),
                            self.cfg.get_train_cfg_as_string().replace("\n", "\n    "),
                        ),
                    )

                if is_power_of_2(self.global_step.numpy()):
                    tf.summary.scalar("cpu_memory", get_current_cpu_memory_usage())
                tf.summary.scalar("time_per_iteration", time() - t_start_iter)
            self.global_step.assign_add(1)

    def save_model2disk(self):
        self.model.save_weights(
            osp.join(
                self.experiment_path,
                "checkpoints",
                "%09d" % self.global_step,
                "model_save_weights",
            )
        )
        self.chkpt_manager.save()

    def run_complete_validation(
        self,
        valid_dataset,
        valid_writer,
        noop_writer,
        compute_losses=True,
    ):
        list_of_metrics = []
        with noop_writer.as_default():
            for val_elem in tqdm(valid_dataset):
                list_of_metrics.append(
                    print_on_fail(val_elem["name"])(self.valid_one_step)(
                        val_elem,
                        self.global_step,
                        cfg=self.cfg,
                        model=self.model,
                        summaries={
                            "writer": noop_writer,
                            "imgs_eval": False,
                            "metrics_eval": True,
                            "metrics_label_dict": self.metrics_label_dict,
                            "label_mapping": self.label_mapping,
                        },
                        compute_losses=compute_losses,
                    )
                )
                list_of_metrics[-1] = list_of_metrics[-1][0]

        metrics_lists = turn_list_of_dicts_to_dict_of_lists(list_of_metrics)
        agg_metrics = {}
        do_not_average = ["conf_mat", "num_pts_used"]
        for type_select in metrics_lists:
            if type_select == "losses":
                continue
            for mask in metrics_lists[type_select]:
                num_pts_used = np.array(
                    metrics_lists[type_select][mask]["num_pts_used"]
                )
                assert num_pts_used.dtype == np.int64
                assert num_pts_used.ndim == 1
                assert np.all(num_pts_used >= 0)
                mask_out_zero_points_bc_they_are_averaged_NaNs = num_pts_used > 0
                sum_num_pts_used = np.sum(
                    num_pts_used[mask_out_zero_points_bc_they_are_averaged_NaNs], axis=0
                )
                for k in metrics_lists[type_select][mask]:
                    if k in do_not_average:
                        agg_metrics[osp.join(k, type_select, mask, "sum")] = np.sum(
                            np.array(metrics_lists[type_select][mask][k]), axis=0
                        )
                    else:
                        agg_metrics[osp.join(k, type_select, mask, "pointwise_avg")] = (
                            np.sum(
                                (
                                    np.array(metrics_lists[type_select][mask][k])
                                    * num_pts_used
                                )[mask_out_zero_points_bc_they_are_averaged_NaNs],
                                axis=0,
                            )
                            / sum_num_pts_used
                        )

        # #region add AEE 5050 to agg_metrics
        for k in list(agg_metrics):
            if "AEE/still" != k[:9]:
                continue
            stillkey = k
            movkey = k.replace("still", "moving")
            resultkey = k.replace("AEE/still", "AEE/5050")
            agg_metrics[resultkey] = 0.5 * (agg_metrics[stillkey] + agg_metrics[movkey])
        # #endregion add AEE 5050 to agg_metrics

        with valid_writer.as_default():
            for k, v in agg_metrics.items():
                if "conf_mat" in k:
                    continue
                tf.summary.scalar(k, v)
        assert agg_metrics["conf_mat/overall/all/sum"].shape == (4, 4)
        ious = compute_custom_class_iou_metrics(agg_metrics["conf_mat/overall/all/sum"])

        with valid_writer.as_default():
            for k, v in ious.items():
                if len(v.shape) == 0 and v.dtype != tf.string:
                    tf.summary.scalar("classification/%s" % k, v)
                else:
                    tf.summary.text("classification/%s" % k, v)

        valid_writer.flush()


def average_cosine_distance(flow_pred, flow_gt, mask):
    flow_pred = tf.math.l2_normalize(flow_pred, axis=-1)
    flow_gt = tf.math.l2_normalize(flow_gt, axis=-1)
    cs = tf.reduce_sum(tf.multiply(flow_pred, flow_gt), axis=-1)
    acd = 1.0 - tf.reduce_mean(tf.boolean_mask(cs, mask))
    return acd


def get_inlier_outlier_ratios(pred_flow, gt_flow, inspect_these_points_mask):
    end_point_errors = tf.norm(pred_flow - gt_flow, axis=-1)

    acc_3d_0_05 = get_ratio_for_thresh(
        end_point_errors,
        abs_thresh=0.05,
        rel_thresh=0.05,
        gt_flow=gt_flow,
        inspect_these_points_mask=inspect_these_points_mask,
        mode="inliers",
        abs_AND_rel=False,
    )
    acc_3d_0_1 = get_ratio_for_thresh(
        end_point_errors,
        abs_thresh=0.1,
        rel_thresh=0.1,
        gt_flow=gt_flow,
        inspect_these_points_mask=inspect_these_points_mask,
        mode="inliers",
        abs_AND_rel=False,
    )
    outliers_3d = get_ratio_for_thresh(
        end_point_errors,
        abs_thresh=0.3,
        rel_thresh=0.1,
        gt_flow=gt_flow,
        inspect_these_points_mask=inspect_these_points_mask,
        mode="outliers",
        abs_AND_rel=False,
    )
    robust_outliers_3d = get_ratio_for_thresh(
        end_point_errors,
        abs_thresh=0.3,
        rel_thresh=0.3,
        gt_flow=gt_flow,
        inspect_these_points_mask=inspect_these_points_mask,
        mode="outliers",
        abs_AND_rel=True,
    )

    return {
        "ACC3D_0_05": acc_3d_0_05,
        "ACC3D_0_1": acc_3d_0_1,
        "Outliers3D": outliers_3d,
        "RobustOutliers3D": robust_outliers_3d,
    }


def get_ratio_for_thresh(
    end_point_errors,
    abs_thresh,
    rel_thresh,
    gt_flow,
    inspect_these_points_mask,
    mode,
    abs_AND_rel: bool,
):
    # if we are woring in mode=="inlier", variable names with "inlier" refer to inliers
    # otherwise its outliers
    assert mode in ["inliers", "outliers"]

    relative_error = end_point_errors / tf.norm(gt_flow, axis=-1)
    relative_error = tf.where(
        tf.math.is_finite(relative_error),
        relative_error,
        tf.ones_like(relative_error) * 2.0 * rel_thresh,
    )
    if mode == "inliers":
        point_is_inlier_absolute = end_point_errors < abs_thresh
        point_is_inlier_relative = relative_error < rel_thresh
    else:
        point_is_inlier_absolute = end_point_errors > abs_thresh
        point_is_inlier_relative = relative_error > rel_thresh

    if abs_AND_rel:
        point_is_inlier = tf.logical_and(
            point_is_inlier_absolute, point_is_inlier_relative
        )
    else:
        point_is_inlier = tf.logical_or(
            point_is_inlier_absolute, point_is_inlier_relative
        )
    num_inliers = tf.math.count_nonzero(
        tf.logical_and(point_is_inlier, inspect_these_points_mask)
    )
    num_pts_total = tf.math.count_nonzero(inspect_these_points_mask)
    ratio = num_inliers / num_pts_total
    if mode == "inliers":
        ratio = tf.where(
            tf.equal(num_pts_total, tf.zeros_like(num_pts_total)),
            tf.ones_like(ratio),
            ratio,
        )
    else:
        assert mode == "outliers"
        ratio = tf.where(
            tf.equal(num_pts_total, tf.zeros_like(num_pts_total)),
            tf.zeros_like(ratio),
            ratio,
        )
    return ratio


def create_train_summaries(
    aggr_flow,
    prediction_is_dynamic,
    el,
    losses,
    learning_rate,
    summaries,
    backward_static_trafo=None,
    plot_directly_instead_of_accumulation=False,
):
    metrics_label_dict = summaries["metrics_label_dict"]

    if plot_directly_instead_of_accumulation:
        tf.summary.scalar("learning_rate", learning_rate)
        for k in losses:
            tf.summary.scalar("loss/%s" % k, losses[k])
        tf.summary.text("gt_forward_static_trafo", tf.as_string(el["odom_t0_t1"][0]))

    if backward_static_trafo is not None:
        point_avg_odom_error = tf.reduce_mean(
            trafo_distance(
                el["odom_t0_t1"] - backward_static_trafo,
                tf.concat([el["pcl_t0"]["pc"], el["pcl_t1"]["pc"]], axis=1),
            )
        )
        if plot_directly_instead_of_accumulation:
            tf.summary.scalar("PAOE", point_avg_odom_error)
    else:
        point_avg_odom_error = None

    def metrics_to_tb(type_name, mask_name, metrics):
        if plot_directly_instead_of_accumulation:
            for k, v in metrics.items():
                if len(v.shape) > 0 or v.dtype == tf.string:
                    continue
                if k == "num_pts_used":
                    agg_type = "sum"
                else:
                    agg_type = "pointwise_avg"
                tf.summary.scalar(osp.join(k, type_name, mask_name, agg_type), v)

    res_mask_metrics = {"overall": {}}

    required_fields_for_eval = ["flow_t0_t1", "flow_gt_t0_t1"]
    if all(key in el.keys() for key in required_fields_for_eval):
        endpoint_errors = tf.linalg.norm(aggr_flow - el["flow_t0_t1"], axis=-1)
        base_mask = tf.logical_and(
            tf.logical_not(tf.math.is_nan(endpoint_errors)),
            el["flow_gt_t0_t1"]["exact_gt_mask"],
        )

        for mask, mask_name in [
            (base_mask, "all"),
            (base_mask & el["nearest_neighbor_points_mask_t0"], "nn_filtered"),
        ]:
            # OVERALL
            overall_metrics_dict = compute_scene_flow_metrics_for_points_in_this_mask(
                aggr_flow,
                el["flow_t0_t1"],
                mask,
            )
            conf_mat = new_compute_custom_conf_mat(
                gt_moving=el["flow_gt_t0_t1"]["moving_mask"],
                semantics_gt=el.get("semantics_t0", None),
                label_dict=metrics_label_dict,
                mask=mask,
                prediction_is_dynamic=prediction_is_dynamic,
            )

            overall_metrics = {
                **overall_metrics_dict,
                "conf_mat": conf_mat,
            }
            metrics_to_tb("overall", mask_name, overall_metrics)
            if point_avg_odom_error is not None and mask_name == "":
                overall_metrics["PAOE"] = point_avg_odom_error

            res_mask_metrics["overall"][mask_name] = overall_metrics

            possible_submasks = [
                (el["flow_gt_t0_t1"]["moving_mask"], "moving"),
                (~el["flow_gt_t0_t1"]["moving_mask"], "still"),
            ]
            if "semantics_t0" in el:
                possible_submasks += [
                    (
                        tf.equal(el["semantics_t0"], metrics_label_dict["static"]),
                        "static",
                    ),
                    (
                        tf.equal(el["semantics_t0"], metrics_label_dict["dynamic"]),
                        "dynamic",
                    ),
                    (
                        tf.equal(el["semantics_t0"], metrics_label_dict["ground"]),
                        "ground",
                    ),
                ]

            for submask, subname in possible_submasks:
                cur_mask = mask & submask
                cur_metrics = compute_scene_flow_metrics_for_points_in_this_mask(
                    aggr_flow,
                    el["flow_t0_t1"],
                    cur_mask,
                )
                metrics_to_tb(subname, mask_name, cur_metrics)
                if subname not in res_mask_metrics:
                    res_mask_metrics[subname] = {}
                res_mask_metrics[subname][mask_name] = cur_metrics

            # custom 5050 AEE
            subname = "overall"
            aee_5050 = 0.5 * (
                res_mask_metrics["moving"][mask_name]["AEE"]
                + res_mask_metrics["still"][mask_name]["AEE"]
            )
            # does not make sense to aggregate AEE_5050 over multiple samples
            # res_mask_metrics[subname][mask_name]["AEE_5050"] = aee_5050
            metrics_to_tb(subname, mask_name, {"AEE_5050": aee_5050})
    return res_mask_metrics


def compute_scene_flow_metrics_for_points_in_this_mask(pred_flow, gt_flow, mask):
    endpoint_errors = tf.linalg.norm(pred_flow - gt_flow, axis=-1)
    masked_endpoint_errors = tf.boolean_mask(endpoint_errors, mask)
    in_out_liers_dict = get_inlier_outlier_ratios(pred_flow, gt_flow, mask)
    avg_endpoint_error = tf.reduce_mean(masked_endpoint_errors)
    avg_cosine_dist = average_cosine_distance(pred_flow, gt_flow, mask)
    num_pts_used_for_metric = tf.math.count_nonzero(mask)
    return {
        **in_out_liers_dict,
        "ACD": avg_cosine_dist,
        "AEE": avg_endpoint_error,
        "num_pts_used": num_pts_used_for_metric,
    }


def train_one_step_eager(el, step, *, cfg, optimizer, model, summaries, cur_train_mode):
    print(
        "################################### Tracing train_one_step_eager ###################################"
    )
    learning_rate = (
        cfg.learning_rate.initial
        * warm_up(step, **cfg.learning_rate.warm_up)
        * step_decay(step, **cfg.learning_rate.step_decay)
    )
    optimizer.lr = learning_rate
    with tf.GradientTape() as tape:
        with summaries["writer"].as_default():
            res_losses = model(
                el, summaries=summaries, compute_losses=True, training=True
            )
        gradients = tape.gradient(res_losses[cur_train_mode], model.trainable_variables)

    if not all(g is None for g in gradients):
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    else:
        print(
            tcolor.WARNING
            + "Warning: There were no gradients computed and no optimizing takes place!"
            + tcolor.ENDC
        )
        any_output_by_net = any(
            v == "net" for v in cfg.model.output_modification.values()
        )
        if any_output_by_net:
            raise AssertionError(
                "No gradients, despite calling train function. You surely do not want this!"
            )

    backward_static_trafo = None
    if "unsupervised" in model.loss_layers and hasattr(
        model.loss_layers["unsupervised"].symmetric_static_points_loss, "trafo_bw"
    ):
        backward_static_trafo = model.loss_layers[
            "unsupervised"
        ].symmetric_static_points_loss.trafo_bw

    with summaries["writer"].as_default():
        mask_metrics = create_train_summaries(
            aggr_flow=model.prediction_fw["aggregated_flow"],
            prediction_is_dynamic=model.prediction_fw["is_dynamic"],
            el=el,
            losses=res_losses,
            learning_rate=optimizer.lr,
            summaries=summaries,
            backward_static_trafo=backward_static_trafo,
            plot_directly_instead_of_accumulation=True,
        )

    return {"losses": res_losses, **mask_metrics}


def relax_shape(tensor):
    if (
        rank(tensor) in [3, 4]
        and tensor.shape[1] == tensor.shape[2]
        and tensor.shape[1] >= 8
    ):
        # this is very likely a image like tensor with/without channel dimension
        result = [s if i > 0 else None for i, s in enumerate(tensor.shape)]
    else:
        result = [
            s if s in {2, 3, 4, 11} and i > 0 else None
            for i, s in enumerate(tensor.shape)
        ]
    # print(tensor.shape, "\tconverted to\t", result)
    return result


train_one_step = relaxed_tf_function(relax_shape)(train_one_step_eager)


def valid_one_step_eager(
    el, step, *, cfg, model, summaries, compute_losses=True, return_bev_fmaps=False
):
    print(
        "################################### Tracing valid_one_step_eager ###################################"
    )
    learning_rate = (
        cfg.learning_rate.initial
        * warm_up(step, **cfg.learning_rate.warm_up)
        * step_decay(step, **cfg.learning_rate.step_decay)
    )
    res_losses = model(
        el, summaries=summaries, compute_losses=compute_losses, training=False
    )

    if summaries["metrics_eval"]:

        backward_static_trafo = None
        if "unsupervised" in model.loss_layers and hasattr(
            model.loss_layers["unsupervised"].symmetric_static_points_loss, "trafo_bw"
        ):
            backward_static_trafo = model.loss_layers[
                "unsupervised"
            ].symmetric_static_points_loss.trafo_bw

        mask_metrics = create_train_summaries(
            aggr_flow=model.prediction_fw["aggregated_flow"],
            prediction_is_dynamic=model.prediction_fw["is_dynamic"],
            el=el,
            losses=res_losses,
            learning_rate=learning_rate,
            summaries=summaries,
            backward_static_trafo=backward_static_trafo,
            plot_directly_instead_of_accumulation=False,
        )
        dict_retvals = {"losses": res_losses, **mask_metrics}
    else:
        dict_retvals = {"losses": res_losses}
    if return_bev_fmaps:
        bev_fmaps = {
            "bev_pillar_feature_map_t0": model.network.bev_img_t0,
            "bev_pillar_feature_map_t1": model.network.bev_img_t1,
            "bev_enc_feature_map_t0": model.network.fmap_t0,
            "bev_enc_feature_map_t1": model.network.fmap_t1,
        }
        pred_fw = model.prediction_fw
        pred_fw["bev_feature_maps"] = bev_fmaps
        return dict_retvals, pred_fw
    else:
        return dict_retvals, model.prediction_fw


valid_one_step = relaxed_tf_function(relax_shape)(valid_one_step_eager)
