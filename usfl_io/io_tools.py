#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import os
import os.path as osp
from typing import List, Tuple

import h5py
import numpy as np
import tensorflow as tf

from cfgattrdict import ConfigAttrDict
from dl_np_tools import cast32, cast64
from npimgtools import Transform
from tfrecutils import get_filenames_and_feature_format, tfrecord_parser

from .batching import padded_batch, tcolor

INPUT_DATADIR = os.getenv("INPUT_DATADIR", "INPUT_DATADIR_ENV_NOT_DEFINED")
CFG_DIR = os.getenv("CFG_DIR", "CFG_DIR_ENV_NOT_DEFINED")


def load_kitti_stereo_sf_hdf5_file(
    filepath: np.array,
):

    pcl_t0_id = "pts_t0"
    pcl_t1_id = "pts_t0_plus_flow"  # or use "pts_t0" if you want the full stereo data
    flow_t0_t1_id = "flow_gt_t0_t1"
    odom_t0_t1_id = "odometry_t0_t1"
    semantics_list_t0_id = "labels_t0"

    with h5py.File(filepath, mode="r", driver="core") as f:
        pcl_t0 = np.ascontiguousarray(f[pcl_t0_id], dtype=np.float32)
        pcl_t1 = np.ascontiguousarray(f[pcl_t1_id], dtype=np.float32)
        flow_t0_t1 = np.ascontiguousarray(f[flow_t0_t1_id], dtype=np.float32)
        odom_t0_t1 = np.ascontiguousarray(f[odom_t0_t1_id], dtype=np.float32).astype(
            np.float64
        )
        semantics_list_t0 = np.ascontiguousarray(
            f[semantics_list_t0_id], dtype=np.int32
        )

    return [
        pcl_t0,
        pcl_t1,
        flow_t0_t1,
        odom_t0_t1,
        semantics_list_t0,
        osp.basename(osp.dirname(filepath))
        + b"_"
        + osp.splitext(osp.basename(filepath))[0],
    ]


def sample_first_k_of_list(values: List, k: int) -> List:
    n = len(values)
    assert k == -1 or 0 <= k <= n
    if k == -1:
        return values
    return values[:k]


def sample_by_name(values: List[str], names: List[str]) -> List:
    result = [v for v in values if any(n in v for n in names)]
    return result


def drop_data_from_dataset_where_mask_is_false(sample, t, mask):
    assert t in {"t0", "t1"}, t
    keys = [
        "pcl_%s" % t,
        "ref_%s" % t,
        "semantics_%s" % t,
        "flow_annotation_%s" % t,
    ]
    keys += ["flow_t0_t1"] if t == "t0" else ["flow_t1_t0"]
    for k in keys:
        if k in sample:
            sample[k] = tf.boolean_mask(sample[k], mask)

    flow_dict_key = "flow_gt_%s" % keys[-1][-5:]
    if flow_dict_key in sample:
        for subkey in [
            "flow",
            "annotation_mask",
            "nn_interpolated_mask",
            "exact_gt_mask",
            "ego_flow_mask",
        ]:
            if subkey in sample[flow_dict_key]:
                sample[flow_dict_key][subkey] = tf.boolean_mask(
                    sample[flow_dict_key][subkey], mask
                )

    return sample


def filter_using_bev(sample, bev_extent=(-50.0, -50.0, 50.0, 50.0)):
    bev_extent = np.array(bev_extent)
    for t in ["t0", "t1"]:
        pc = sample["pcl_%s" % t][..., :2]
        mask = tf.reduce_all(
            tf.concat([pc >= bev_extent[:2], pc <= bev_extent[2:]], axis=-1), axis=-1
        )
        tf.Assert(
            tf.reduce_sum(tf.cast(mask, tf.int32)) >= 3,
            data=[
                "too many points bev filtered",
                sample["name"],
                tf.shape(mask),
                tf.reduce_sum(tf.cast(mask, tf.int32)),
                mask,
            ],
        )

        sample = drop_data_from_dataset_where_mask_is_false(sample, t, mask)
    return sample


def infer_ground_label_using_cone(
    pcl, cone_z_threshold__m: float = -1.70, cone_angle__deg: float = 0.8
):
    assert 0.0 <= cone_angle__deg <= 10.0  # 10 deg arbitrary high value as sanity check
    if cone_angle__deg > 0.0:
        cone_angle = cone_angle__deg / 180.0 * np.pi
        d_xy = tf.linalg.norm(pcl[..., 0:2], axis=-1)
        z_t_thresh = cone_z_threshold__m + np.tan(cone_angle) * d_xy
        is_ground = pcl[..., 2] < z_t_thresh
    else:
        is_ground = pcl[..., 2] < cone_z_threshold__m
    return is_ground


def filter_ground_by_cone(sample, cone_z_threshold__m: float, cone_angle__deg: float):
    for t in ["pcl_t0", "pcl_t1"]:
        is_ground = infer_ground_label_using_cone(
            sample[t],
            cone_z_threshold__m=cone_z_threshold__m,
            cone_angle__deg=cone_angle__deg,
        )
        sample = drop_data_from_dataset_where_mask_is_false(sample, t[-2:], ~is_ground)
    return sample


def filter_by_angle_fov(
    sample, min_opening_angle__deg: float, max_opening_angle__deg: float
):
    for t in ["pcl_t0", "pcl_t1"]:
        pc = sample[t]
        assert len(pc.shape) == 2
        assert pc.shape[1] == 3
        pc_x, pc_y = pc[:, 0], pc[:, 1]
        angles = tf.atan2(pc_y, pc_x)
        min_angle = min_opening_angle__deg / 180.0 * np.pi
        max_angle = max_opening_angle__deg / 180.0 * np.pi
        filter_mask = (angles >= min_angle) & (angles <= max_angle)
        sample = drop_data_from_dataset_where_mask_is_false(sample, t[-2:], filter_mask)
    return sample


def filter_ground_using_mapped_semantics(sample, labelmap):
    assert "ground" in labelmap.mnames
    ground_idx = labelmap.mnames.index("ground")

    for t in ["pcl_t0", "pcl_t1"]:
        is_not_ground_mask = tf.logical_not(
            sample["semantics_%s" % t[-2:]] == ground_idx
        )
        tf.Assert(
            tf.reduce_sum(tf.cast(is_not_ground_mask, tf.int32)) >= 3,
            data=["too many points ground filtered"],
        )
        sample = drop_data_from_dataset_where_mask_is_false(
            sample, t[-2:], is_not_ground_mask
        )
    return sample


def drop_ground_points(
    sample,
    *,
    method: str,
    labelmap=None,
    cone_angle__deg: float = None,
    cone_z_threshold__m: float = None,
):
    assert method in {"cone", "semantics", "semantics_cone"}, method

    if "semantics" in method:
        assert labelmap is not None, "need labelmap to filter ground using semantics!"
        sample = filter_ground_using_mapped_semantics(sample, labelmap)
    if "cone" in method:
        assert cone_angle__deg is not None
        assert cone_z_threshold__m is not None
        sample = filter_ground_by_cone(
            sample,
            cone_z_threshold__m=cone_z_threshold__m,
            cone_angle__deg=cone_angle__deg,
        )
    else:
        assert cone_angle__deg is None
        assert cone_z_threshold__m is None

    return sample


def apply_subset_sampling(sample, num_points: int):
    for t in ["t0", "t1"]:
        n = tf.shape(sample["pcl_%s" % t])[0]
        uniform_distribution = tf.random.uniform(shape=[n], dtype=tf.float32)
        _, sampled_indices = tf.math.top_k(
            uniform_distribution, tf.minimum(n, num_points)
        )
        # sampled_indices1 = np.random.choice(
        #     indices1, size=self.num_points, replace=False, p=None
        # )
        mask = tf.scatter_nd(
            sampled_indices[:, None], tf.ones_like(sampled_indices, dtype=tf.bool), [n]
        )
        assert len(mask.shape) == 1
        drop_data_from_dataset_where_mask_is_false(sample, t, mask)
    return sample


def apply_labelmap(sample, labelmap, perform_checks=True):
    if not perform_checks:
        CRED = "\033[91m"
        CEND = "\033[0m"
        print(
            CRED
            + "Skipping Label Map Checks - (could be because of unlabeled KITTI Dataset?!)"
            + CEND
        )
    for sem in ["semantics_t0", "semantics_t1"]:
        if sem not in sample:
            continue
        map_idx_tensor = -np.ones(
            (max(labelmap.ridx_midx_dict.keys()) + 1), dtype=np.int32
        )
        for raw_idx, mapped_index in labelmap.ridx_midx_dict.items():
            map_idx_tensor[raw_idx] = mapped_index
        sample[sem] = tf.gather(map_idx_tensor, sample[sem])
        tf.Assert(tf.reduce_all(sample[sem] >= 0), data=["some raw labels not mapped"])
        if perform_checks:
            for i, n in enumerate(labelmap.mnames):
                fraction_of_points = tf.reduce_sum(
                    tf.cast(tf.equal(sample[sem], i), tf.int32)
                ) / tf.size(sample[sem])
                if n == "ignore":
                    tf.Assert(
                        fraction_of_points < 0.9,
                        data=[
                            sample["name"]
                            + ": more than 90% unlabeled: "
                            + tf.as_string(100.0 * fraction_of_points)
                            + "%"
                        ],
                    )
                if n == "dynamic":
                    tf.Assert(
                        fraction_of_points < 0.99,
                        data=[
                            sample["name"]
                            + ": more than 99% dynamic: "
                            + tf.as_string(100.0 * fraction_of_points)
                            + "%"
                        ],
                    )
                # if n == "ground":
                #     tf.Assert(
                #         tf.equal(fraction_of_points, 0.0),
                #         data=[
                #             sample["name"]
                #             + ": not all ground points filtered: "
                #             + tf.as_string(100.0 * fraction_of_points)
                #             + "%"
                #         ],
                #     )
                if n == "static":
                    tf.Assert(
                        fraction_of_points > 0.01,
                        data=[
                            sample["name"]
                            + ": less than 1% static: "
                            + tf.as_string(100.0 * fraction_of_points)
                            + "%"
                        ],
                    )
    return sample


def add_moving_mask(
    sample,
    labelmap,
    non_rigid_flow_threshold: float = 0.05,
    ignore_contradicting_flow_semseg: bool = False,
):
    if "odom_t0_t1" not in sample:
        # we need odometry to compute non rigid flow component
        return sample

    inv_odom = {
        "t0": tf.linalg.inv(sample["odom_t0_t1"]),
        "t1": sample["odom_t0_t1"],
    }

    for t0, t1 in [("t0", "t1"), ("t1", "t0")]:
        flow_key = "flow_gt_%s_%s" % (t0, t1)
        pc_key = "pcl_%s" % t0
        sem_key = "semantics_%s" % t0
        if flow_key not in sample:
            continue
        static_flow = cast32(
            cast64(sample[pc_key])
            @ tf.transpose(inv_odom[t0][:3, :3] - tf.eye(3, dtype=tf.float64))
            + inv_odom[t0][:3, 3]
        )
        non_rigid_flow = sample[flow_key]["flow"] - static_flow
        non_rigid_flow_mag = tf.linalg.norm(non_rigid_flow, axis=-1)
        moving_mask = (
            non_rigid_flow_mag >= non_rigid_flow_threshold
        ) & ~tf.math.is_nan(non_rigid_flow_mag)

        if sem_key in sample:
            semseg_not_dynamic = sample[sem_key] != labelmap.mnames.index("dynamic")
            moving_but_not_dynamic = semseg_not_dynamic & moving_mask
            count_moving_but_not_dynamic = tf.math.count_nonzero(moving_but_not_dynamic)
            count_moving = tf.math.count_nonzero(moving_mask)
            if not ignore_contradicting_flow_semseg:
                tf.Assert(
                    count_moving_but_not_dynamic
                    <= tf.math.count_nonzero(moving_mask) // 10,
                    data=[
                        sample["name"],
                        "from_%s_to_%s" % (t0, t1),
                        "found sample with dynamic looking flow but semantics say non-dynamic",
                        "count_moving_but_not_dynamic:",
                        count_moving_but_not_dynamic,
                        "total_count_moving:",
                        count_moving,
                    ],
                )
            moving_mask = moving_mask & ~semseg_not_dynamic

        sample[flow_key]["moving_mask"] = moving_mask
    return sample


def create_configurable_voxelization_map(voxel_cfg):
    from unsup_flow.tf_user_ops import points_to_voxel

    def apply_pointcloud_voxelization(sample):
        # # [batch_dim, hist_dim, point_dim, feat_dim]
        # pointcloud_tensor = sample["pointclouds"]
        # pointcloud_tensor, _ = temp_axis_squash(pointcloud_tensor, 2)
        for pc_name in ["pcl_t0", "pcl_t1"]:
            pointcloud_tensor = sample[pc_name]
            (
                voxel_feats,
                point_feats,
                voxel_coors,
                voxel_count,
                _,
                _,
                _,
            ) = points_to_voxel(
                # NOTE: add one dimension as dummy reflectiviy
                tf.concat([pointcloud_tensor, pointcloud_tensor[..., :1]], axis=-1),
                voxel_cfg["extent"],
                voxel_cfg["resolution"],
                voxel_cfg["max_points_per_voxel"],
            )
            flat_point_feats = tf.boolean_mask(
                point_feats,
                tf.sequence_mask(voxel_count, maxlen=voxel_cfg["max_points_per_voxel"]),
            )
            point_feats = tf.RaggedTensor.from_row_lengths(
                values=flat_point_feats, row_lengths=tf.cast(voxel_count, tf.int64)
            )
            sample[pc_name] = {}
            sample[pc_name]["pc"] = pointcloud_tensor
            sample[pc_name]["voxel_feats"] = voxel_feats
            sample[pc_name]["point_feats"] = {
                "values": point_feats.values,
                "row_splits": point_feats.row_splits,
            }
            sample[pc_name]["voxel_coors"] = voxel_coors
            sample[pc_name]["voxel_count"] = voxel_count
        sample["batch_size"] = tf.shape(pointcloud_tensor)[0]
        return sample

    return apply_pointcloud_voxelization


def compute_pointwise_voxel_coordinates(
    sample, grid_size=(320, 320), bev_extent=(-50.0, -50.0, 50.0, 50.0)
):
    bev_extent = np.array(bev_extent)
    for pc_name in ["pcl_t0", "pcl_t1"]:
        v_coors = sample[pc_name]["pc"][..., :2]
        v_coors -= bev_extent[:2]
        v_coors /= bev_extent[2:] - bev_extent[:2]
        valid_mask = tf.logical_not(tf.math.is_nan(v_coors[..., 0]))
        # view_mask = tf.logical_or(
        #     tf.reduce_all(tf.concat([v_coors >= 0.0, v_coors < 1.0], axis=-1), axis=-1),
        #     tf.logical_not(valid_mask),
        # )
        v_coors = tf.clip_by_value(v_coors, 0, 1.0)
        v_coors = tf.where(
            tf.tile(valid_mask[..., None], [1, 1, v_coors.shape[-1]]),
            v_coors,
            tf.zeros_like(v_coors),
        )
        grid_size_float = np.array(grid_size).astype(np.float32)
        v_coors = tf.cast(
            tf.minimum(
                v_coors * grid_size_float[None, None, :],
                grid_size_float[None, None, :] - 0.5,
            ),
            tf.int32,
        )
        sample[pc_name]["pointwise_voxel_coors"] = v_coors
        sample[pc_name]["pointwise_valid_mask"] = valid_mask
    return sample


def infer_flow_map(flow_gt_t0_t1, grid_size, pillar_coords_t0, point_valid_mask_t0):
    pillar_coords_t0 = tf.boolean_mask(pillar_coords_t0, point_valid_mask_t0)
    flow_gt_t0_t1 = tf.boolean_mask(flow_gt_t0_t1, point_valid_mask_t0)
    bev_aggr_flow_target_shape = [*grid_size, 3]
    flow_map_gt_t0_t1 = tf.scatter_nd(
        indices=pillar_coords_t0,
        updates=flow_gt_t0_t1,
        shape=bev_aggr_flow_target_shape,
    )
    counter_shape = [*grid_size]
    points_counter_t0 = tf.scatter_nd(
        indices=pillar_coords_t0,
        updates=tf.ones_like(flow_gt_t0_t1[..., 0], dtype=tf.int32),
        shape=counter_shape,
    )
    flow_map_gt_t0_t1 = tf.math.divide_no_nan(
        flow_map_gt_t0_t1, tf.cast(points_counter_t0, dtype=tf.float32)[..., None]
    )
    return flow_map_gt_t0_t1, points_counter_t0


def compute_gt_flow_bev_maps(ds, grid_size=(320, 320)):
    def infer_flow_map_wrap(elems):
        pillar_coords_t0, point_valid_mask_t0, flow_gt_t0_t1 = elems
        flow_map_gt_t0_t1, points_counter_t0 = infer_flow_map(
            flow_gt_t0_t1, grid_size, pillar_coords_t0, point_valid_mask_t0
        )
        return flow_map_gt_t0_t1, points_counter_t0

    pillar_coords = {}
    pillar_coords["t0"] = ds["pcl_t0"]["pointwise_voxel_coors"]
    pillar_coords["t1"] = ds["pcl_t1"]["pointwise_voxel_coors"]
    point_valid_mask = {}
    point_valid_mask["t0"] = ds["pcl_t0"]["pointwise_valid_mask"]
    point_valid_mask["t1"] = ds["pcl_t1"]["pointwise_valid_mask"]

    for flow_key in ["t0_t1", "t1_t0"]:
        if "flow_%s" % flow_key not in ds:
            continue
        flow_map_gt, points_counter = tf.map_fn(
            infer_flow_map_wrap,
            [
                pillar_coords[flow_key[:2]],
                point_valid_mask[flow_key[:2]],
                ds["flow_%s" % flow_key],
            ],
            dtype=(tf.float32, tf.int32),
        )
        ds["pcl_%s" % flow_key[:2]]["flow_map_bev_gt"] = flow_map_gt
        ds["pcl_%s" % flow_key[:2]]["bev_points_counter_map"] = points_counter

    return ds


def infer_semseg_map(
    ohe_semantics_gt_t0, grid_size, pillar_coords_t0, point_valid_mask_t0
):
    assert ohe_semantics_gt_t0.dtype == tf.bool
    ignore_semantic = ohe_semantics_gt_t0[..., 0]
    point_mask = point_valid_mask_t0 & (~ignore_semantic)
    pillar_coords_t0 = tf.boolean_mask(pillar_coords_t0, point_mask)
    ohe_semantics_gt_t0 = tf.boolean_mask(ohe_semantics_gt_t0, point_mask)
    bev_aggr_flow_target_shape = [*grid_size, ohe_semantics_gt_t0.shape[-1] - 1]
    gt_ohe_semantics_bev = tf.scatter_nd(
        indices=pillar_coords_t0,
        updates=tf.cast(ohe_semantics_gt_t0[..., 1:], tf.int32),
        shape=bev_aggr_flow_target_shape,
    )
    return gt_ohe_semantics_bev


def add_ohe_gt_stat_dyn_ground_label_bev_maps(
    ds,
    grid_size=(320, 320),
    labelmap=None,
    final_scale=2,
):
    def wrap_infer_semseg_map(elems, cur_grid_size):
        pillar_coords_t0, point_valid_mask_t0, ohe_semantics_gt_t0 = elems
        gt_ohe_semseg_bev_t0 = infer_semseg_map(
            ohe_semantics_gt_t0, cur_grid_size, pillar_coords_t0, point_valid_mask_t0
        )
        return gt_ohe_semseg_bev_t0

    for t in ["t0", "t1"]:
        pcl_t = "pcl_%s" % t
        if "semantics_%s" % t not in ds:
            continue

        ohe_semantics_gt = tf.one_hot(
            ds["semantics_%s" % t],
            depth=len(labelmap.mnames),
            on_value=True,
            off_value=False,
            dtype=tf.bool,
        )
        assert labelmap.mnames == ["ignore", "dynamic", "ground", "static"]
        gt_ohe_sum_semantics_bev = tf.map_fn(
            lambda x: wrap_infer_semseg_map(x, [gs // final_scale for gs in grid_size]),
            [
                ds[pcl_t]["pointwise_voxel_coors"] // final_scale,
                ds[pcl_t]["pointwise_valid_mask"],
                ohe_semantics_gt,
            ],
            dtype=tf.int32,
        )
        static_idx = labelmap.mnames.index("static") - 1
        dynamic_idx = labelmap.mnames.index("dynamic") - 1
        ground_idx = labelmap.mnames.index("ground") - 1
        sum_static_bev = gt_ohe_sum_semantics_bev[..., static_idx]
        sum_dynamic_bev = gt_ohe_sum_semantics_bev[..., dynamic_idx]
        sum_ground_bev = gt_ohe_sum_semantics_bev[..., ground_idx]
        static_bev = sum_static_bev > tf.maximum(sum_dynamic_bev, sum_ground_bev)
        ground_bev = (sum_ground_bev > sum_dynamic_bev) & ~static_bev
        dynamic_bev = ~ground_bev & ~static_bev
        ds["ohe_gt_stat_dyn_ground_label_bev_map_%s" % t] = tf.stack(
            [static_bev, dynamic_bev, ground_bev],
            axis=-1,
        )
    return ds


def nusc_add_nn_segmentation_flow_for_t1(sample, add_semseg=False, add_flow=False):

    assert add_semseg or add_flow
    if "semantics_t1" in sample.keys() and "flow_gt_t1_t0" in sample.keys():
        return sample

    from unsup_flow.knn.knn_wrapper import (
        get_idx_dists_for_knn,
    )

    idxs_t1_into_t0 = tf.squeeze(
        get_idx_dists_for_knn(
            sample["pcl_t0"] + sample["flow_gt_t0_t1"]["flow"], sample["pcl_t1"]
        ),
        axis=1,
    )
    tf.Assert(
        tf.reduce_all(idxs_t1_into_t0 < tf.shape(sample["pcl_t0"])[0]),
        data=["invalid points found in pcl_t1"],
    )
    if add_semseg:
        sample["semantics_t1"] = tf.gather(sample["semantics_t0"], idxs_t1_into_t0)
    if add_flow:
        ego_flow_mask_t1_t0 = tf.gather(
            sample["flow_gt_t0_t1"]["ego_flow_mask"], idxs_t1_into_t0
        )
        odom_flow = cast32(
            cast64(sample["pcl_t1"])
            @ tf.transpose(sample["odom_t0_t1"][:3, :3] - tf.eye(3, dtype=tf.float64))
            + sample["odom_t0_t1"][:3, 3]
        )
        sample["flow_gt_t1_t0"] = {
            "flow": tf.where(
                ego_flow_mask_t1_t0[:, None],
                odom_flow,
                -tf.gather(sample["flow_gt_t0_t1"]["flow"], idxs_t1_into_t0),
            ),
            "annotation_mask": tf.ones_like(ego_flow_mask_t1_t0),
            "nn_interpolated_mask": tf.ones_like(ego_flow_mask_t1_t0),
            "exact_gt_mask": tf.zeros_like(ego_flow_mask_t1_t0),
            "ego_flow_mask": ego_flow_mask_t1_t0,
        }
    return sample


def get_nuscenes_flow_dataset(
    split: str,
    keep_plain: bool,
    nbr_samples_cfg: ConfigAttrDict,
    data_params: ConfigAttrDict,
    data_dir: str,
):

    assert set(data_params.keys()) == {
        "add_nn_segmentation_for_t1",
        "add_nn_flow_for_t1",
    }, data_params
    filenames, feature_format = get_filenames_and_feature_format(data_dir)
    nusc_split = ConfigAttrDict().from_file(osp.join(CFG_DIR, "splits.yml")).nuscenes

    assert split in {"train", "valid"}
    if split == "train":
        selected_scenes = nusc_split.train
        deselected_scenes = nusc_split.valid
    else:
        selected_scenes = nusc_split.valid
        deselected_scenes = nusc_split.train

    split_filenames = []
    for fname in filenames:
        scene_name, id_name = osp.basename(fname).split("_")[:2]
        if scene_name in selected_scenes:
            if split == "train":
                split_filenames.append(fname)
            else:
                assert split == "valid"
                split_filenames.append(fname)
        else:
            assert scene_name in deselected_scenes

    # #region get only number of samples specified
    if not keep_plain:
        np.random.shuffle(split_filenames)
    if isinstance(nbr_samples_cfg[split], int):
        split_filenames = sample_first_k_of_list(
            split_filenames, nbr_samples_cfg[split]
        )
        nbr_samples_cfg[split] = len(split_filenames)
    else:
        split_filenames = sample_by_name(filenames, nbr_samples_cfg[split])
    # #endregion get only number of samples specified

    dataset, meta = tfrecord_parser(
        split_filenames, feature_format, keep_plain=keep_plain
    )

    def restruct_input(sample):
        sample["ref_t0"] = sample["pcl_t0"][:, 3]
        sample["ref_t1"] = sample["pcl_t1"][:, 3]
        sample["pcl_t0"] = sample["pcl_t0"][:, :3]
        sample["pcl_t1"] = sample["pcl_t1"][:, :3]
        assert "odom_t0_t1" in sample
        sample["flow_gt_t0_t1"] = {
            "flow": sample["flow_t0_t1"],
            "annotation_mask": tf.ones_like(
                sample["stat_possibly_dyn_on_object_box_edges_mask"]
            ),
            "nn_interpolated_mask": tf.zeros_like(
                sample["stat_possibly_dyn_on_object_box_edges_mask"]
            ),
            "exact_gt_mask": ~sample["stat_possibly_dyn_on_object_box_edges_mask"],
            "ego_flow_mask": sample["ego_flow_mask"],
        }
        del sample["flow_t0_t1"]
        del sample["ego_flow_mask"]
        del sample["stat_possibly_dyn_on_object_box_edges_mask"]
        del sample["filename"]
        return sample

    def move_sample_into_lidar_frame_pose(sample):
        lidar_frame = Transform()
        lidar_frame.set_trans(np.array([0.95, 0.0, 1.73]))
        lidar_frame = lidar_frame.as_htm()
        lidar_frame_inv = tf.constant(np.linalg.inv(lidar_frame))
        lidar_frame = tf.constant(lidar_frame)

        sample["pcl_t0"] = cast32(
            cast64(sample["pcl_t0"]) @ tf.transpose(lidar_frame_inv[:3, :3])
            + lidar_frame_inv[:3, 3]
        )
        sample["pcl_t1"] = cast32(
            cast64(sample["pcl_t1"]) @ tf.transpose(lidar_frame_inv[:3, :3])
            + lidar_frame_inv[:3, 3]
        )
        sample["flow_gt_t0_t1"]["flow"] = cast32(
            cast64(sample["flow_gt_t0_t1"]["flow"])
            @ tf.transpose(lidar_frame_inv[:3, :3])
        )
        if "flow_gt_t1_t0" in sample.keys():
            sample["flow_gt_t1_t0"]["flow"] = cast32(
                cast64(sample["flow_gt_t1_t0"]["flow"])
                @ tf.transpose(lidar_frame_inv[:3, :3])
            )
        sample["odom_t0_t1"] = lidar_frame_inv @ sample["odom_t0_t1"] @ lidar_frame
        return sample

    dataset = dataset.map(restruct_input)
    dataset = dataset.map(move_sample_into_lidar_frame_pose)

    add_semseg_t1 = data_params.add_nn_segmentation_for_t1
    add_flow_t1 = data_params.add_nn_flow_for_t1
    if add_semseg_t1 or add_flow_t1:
        dataset = dataset.map(
            lambda sample: nusc_add_nn_segmentation_flow_for_t1(
                sample, add_semseg=add_semseg_t1, add_flow=add_flow_t1
            )
        )

    # dataset = dataset.map(lambda x: filter_ground_by_semantics(x, labelmap=labelmap))

    return dataset, split_filenames


def get_kitti_lidar_raw_dataset(
    keep_plain: bool,
    nbr_samples_cfg: ConfigAttrDict,
    data_dir: str,
    exclude_kitti_stereo_frames=True,
):

    filenames, feature_format = get_filenames_and_feature_format(data_dir)
    num_samples = nbr_samples_cfg["kitti_lidar_raw"]

    if exclude_kitti_stereo_frames:
        kitti_stereo_flow_frames = (
            ConfigAttrDict()
            .from_file(
                osp.join(
                    os.getenv("CFG_DIR", "config"),
                    "kitti_raw_dont_use_these_samples.yml",
                )
            )
            .stereo_scene_flow_frames
        )

        filtered_filenames = []
        for fname in filenames:
            use_this_sample = True
            for sf_frame in kitti_stereo_flow_frames:
                if sf_frame in fname:
                    use_this_sample = False
                    break
            if use_this_sample:
                filtered_filenames.append(fname)

        if len(filenames) >= 14200:
            # i.e. we are not in local debug scenario
            assert (
                len(filenames) - len(filtered_filenames) == 142
            ), "unable to filter kitti stereo flow scenes from kitti raw"
        filenames = filtered_filenames

    assert len(filenames) > 0, "no samples found"

    if not keep_plain:
        np.random.shuffle(filenames)
    filenames = sample_first_k_of_list(filenames, num_samples)
    try:
        nbr_samples_cfg["kitti_lidar_raw"] = len(filenames)
    except AssertionError:
        print("Failed to set nbr_samples_cfg for kitti_lidar_raw")
    ds, meta = tfrecord_parser(
        filenames, feature_format=feature_format, keep_plain=keep_plain
    )

    def restruct_input(sample):
        # sample["ref_t1"] = sample["pcl_t1"][:, 3]
        sample["pcl_t0"] = sample["pcl_t0"][:, :3]
        sample["pcl_t1"] = sample["pcl_t1"][:, :3]
        assert "odom_t0_t1" in sample
        if sample["odom_t0_t1"].dtype == tf.float32:
            sample["odom_t0_t1"] = tf.cast(sample["odom_t0_t1"], tf.float64)
        del sample["filename"]
        return sample

    ds = ds.map(restruct_input)

    return ds, filenames


def load_kitti_stereo_sf_to_dict_of_tensors(filename: tf.Tensor):
    data = tf.py_function(
        lambda x: load_kitti_stereo_sf_hdf5_file(x.numpy()),
        inp=[filename],
        Tout=[tf.float32, tf.float32, tf.float32, tf.float64, tf.int32, tf.string],
        name="kitti_sf_stereo_hdf5_load",
    )

    (pcl_t0, pcl_t1, flow_t0_t1, odom_t0_t1, semantics_t0, name) = data
    pcl_shape = [None, 4]
    flow_shape = [None, 3]
    semantics_shape = [None]
    pcl_t0.set_shape(pcl_shape)
    pcl_t1.set_shape(pcl_shape)

    flow_t0_t1.set_shape(flow_shape)
    semantics_t0.set_shape(semantics_shape)

    flow_annotation_mask_t0 = tf.ones_like(semantics_t0, dtype=tf.bool)

    exact_flow_mask_t0 = tf.ones_like(semantics_t0, dtype=tf.bool)

    odom_t0_t1.set_shape([4, 4])
    name.set_shape([])

    print(
        tcolor.WARNING
        + "Warning: Kitti Stereo Labels t0 are transfered to t1 - explicit 1:1 matches required"
        + tcolor.ENDC
    )

    return tf.data.Dataset.from_tensors(
        {
            "pcl_t0": pcl_t0,
            "pcl_t1": pcl_t1,
            "flow_t0_t1": flow_t0_t1,
            "flow_t1_t0": -flow_t0_t1,
            "odom_t0_t1": odom_t0_t1,
            "semantics_t0": semantics_t0,
            "semantics_t1": semantics_t0,
            "flow_annotation_mask_t0": flow_annotation_mask_t0,
            "flow_annotation_mask_t1": flow_annotation_mask_t0,
            "exact_flow_mask_t0": exact_flow_mask_t0,
            "name": name,
        }
    )


def get_kitti_stereo_flow_dataset(
    keep_plain: bool, nbr_samples_cfg: ConfigAttrDict, data_dir: str
):
    fnames = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
    fnames = [fname for fname in fnames if fname.endswith(".hdf5")]

    # #region get only number of samples specified
    if not keep_plain:
        np.random.shuffle(fnames)
    fnames = sample_first_k_of_list(fnames, nbr_samples_cfg["kitti"])
    nbr_samples_cfg["kitti"] = len(fnames)
    # #endregion get only number of samples specified

    ds = tf.data.Dataset.from_tensor_slices(fnames)

    if not keep_plain:
        ds = ds.shuffle(20000)
        ds = ds.repeat()

    ds = ds.interleave(
        lambda x: load_kitti_stereo_sf_to_dict_of_tensors(x),
        cycle_length=20,
        block_length=1,
        # num_parallel_calls=40,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    sensor_height_above_ground = 1.73
    cutoff_height_above_ground = 0.3
    z_thresh = -sensor_height_above_ground + cutoff_height_above_ground
    ds = ds.map(
        lambda x: filter_ground_by_cone(
            x, cone_z_threshold__m=z_thresh, cone_angle__deg=0.0
        )
    )

    def restruct_input(sample):
        sample["ref_t0"] = sample["pcl_t0"][:, 3]
        sample["ref_t1"] = sample["pcl_t1"][:, 3]
        sample["pcl_t0"] = sample["pcl_t0"][:, :3]
        sample["pcl_t1"] = sample["pcl_t1"][:, :3]
        # assert "odom_t0_t1" in sample
        sample["flow_gt_t0_t1"] = {
            "flow": sample["flow_t0_t1"],
            "annotation_mask": sample["flow_annotation_mask_t0"],
            "nn_interpolated_mask": sample["flow_annotation_mask_t0"]
            & (~sample["exact_flow_mask_t0"]),
            "exact_gt_mask": sample["flow_annotation_mask_t0"]
            & sample["exact_flow_mask_t0"],
        }
        sample["flow_gt_t1_t0"] = {
            "flow": sample["flow_t1_t0"],
            "annotation_mask": sample["flow_annotation_mask_t1"],
            "nn_interpolated_mask": sample["flow_annotation_mask_t1"],
            "exact_gt_mask": tf.zeros_like(sample["flow_annotation_mask_t1"]),
        }
        del sample["semantics_t0"]
        del sample["semantics_t1"]
        del sample["flow_t0_t1"]
        del sample["flow_t1_t0"]
        del sample["flow_annotation_mask_t0"]
        del sample["flow_annotation_mask_t1"]
        del sample["exact_flow_mask_t0"]
        return sample

    ds = ds.map(restruct_input)

    return ds, fnames


def check_dataset_interface(dataset):
    from flatten_dict import flatten

    from .batching import get_output_shapes, get_output_types

    out_types = get_output_types(dataset)
    out_shapes = get_output_shapes(dataset)

    target_types_shapes_opt_group = {
        "name": (tf.string, [], 0),
        "pcl_t0": (tf.float32, [None, 3], 0),
        "pcl_t1": (tf.float32, [None, 3], 0),
        "ref_t0": (tf.float32, [None], 101),
        "ref_t1": (tf.float32, [None], 102),
        "semantics_t0": (tf.int32, [None], 103),
        "semantics_t1": (tf.int32, [None], 104),
        "odom_t0_t1": (tf.float64, [4, 4], 105),
        "flow_gt_t0_t1": {
            "flow": (tf.float32, [None, 3], 1),
            "annotation_mask": (tf.bool, [None], 1),
            "nn_interpolated_mask": (tf.bool, [None], 1),
            "exact_gt_mask": (tf.bool, [None], 1),
            "ego_flow_mask": (tf.bool, [None], 10),
        },
        "flow_gt_t1_t0": {
            "flow": (tf.float32, [None, 3], 2),
            "annotation_mask": (tf.bool, [None], 2),
            "nn_interpolated_mask": (tf.bool, [None], 2),
            "exact_gt_mask": (tf.bool, [None], 2),
            "ego_flow_mask": (tf.bool, [None], 20),
        },
    }
    flat_out_types = flatten(out_types, reducer="path")
    flat_out_shapes = flatten(out_shapes, reducer="path")
    flat_target_types_shapes_opt_group = flatten(
        target_types_shapes_opt_group, reducer="path"
    )
    opt_groups_found = set()
    opt_groups_not_found = set()
    for k in flat_out_types:
        if k in flat_target_types_shapes_opt_group:
            continue
        print(
            tcolor.WARNING
            + "Warning: dataset has additional data %s not defined by interface declaration"
            % k
            + tcolor.ENDC
        )
        raise ValueError(
            "dataset has additional data %s not defined by interface declaration" % k
        )
    for k in flat_target_types_shapes_opt_group:
        dtype, shape, opt_group = flat_target_types_shapes_opt_group[k]
        if k not in flat_out_types:
            opt_groups_not_found.add(opt_group)
        else:
            opt_groups_found.add(opt_group)
            assert flat_out_types[k] == dtype, (k, flat_out_types[k])
            assert flat_out_shapes[k].as_list() == shape, (k, flat_out_shapes[k])
    assert 0 in opt_groups_found
    assert opt_groups_found.isdisjoint(
        opt_groups_not_found
    ), opt_groups_found.intersection(opt_groups_not_found)
    for opt in opt_groups_found:
        if opt > 100:
            continue
        assert (opt // 10) in opt_groups_found


def get_unsupervised_flow_dataset(
    *,
    cfg,
    labelmap,
    voxel_cfg=None,
    valid=False,
    data_source="carla",
    return_before_batch_and_voxelization: bool = False,
    ignore_contradicting_flow_semseg: bool = False,
    num_input_points: int = None,
    keep_plain: bool = None,
    exclude_kitti_stereo_frames=True,
    filter_bev: bool = True,
    data_dir: str = None,
):
    if keep_plain is None:
        keep_plain = valid

    batch_size = cfg.batch_size

    if data_source == "nuscenes":
        assert not hasattr(cfg.data.params, "carla")
        ds, filenames = get_nuscenes_flow_dataset(
            split=["train", "valid"][valid],
            keep_plain=keep_plain,
            nbr_samples_cfg=cfg.data.nbr_samples,
            data_params=cfg.data.params.nuscenes,
            data_dir=data_dir,
        )
    elif data_source == "kitti_lidar_raw":
        ds, filenames = get_kitti_lidar_raw_dataset(
            keep_plain=keep_plain,
            nbr_samples_cfg=cfg.data.nbr_samples,
            data_dir=data_dir,
            exclude_kitti_stereo_frames=exclude_kitti_stereo_frames,
        )
    elif data_source == "kitti_stereo_sf":
        ds, filenames = get_kitti_stereo_flow_dataset(
            keep_plain=keep_plain,
            nbr_samples_cfg=cfg.data.nbr_samples,
            data_dir=data_dir,
        )
    else:
        raise ValueError(
            "don't know what to do with data_source {0}".format(data_source)
        )

    check_dataset_interface(ds)

    is_kitti_data = data_source in [
        "kitti_stereo_sf",
        "kitti_lidar_raw",
    ]

    ds = ds.map(
        lambda x: apply_labelmap(x, labelmap=labelmap, perform_checks=not is_kitti_data)
    )

    if is_kitti_data and cfg.data.ground_filtering.kitti.method is not None:
        ds = ds.map(
            lambda sample: drop_ground_points(sample, **cfg.data.ground_filtering.kitti)
        )
    elif cfg.data.ground_filtering.base_data.method is not None:
        if (
            data_source == "nuscenes"
            and "semantics" in cfg.data.ground_filtering.base_data.method
        ):
            assert (
                cfg.data.params.nuscenes.add_nn_segmentation_for_t1
            ), "we need semantics to drop ground points via semantics"

        ds = ds.map(
            lambda sample: drop_ground_points(
                sample, labelmap=labelmap, **cfg.data.ground_filtering.base_data
            )
        )
    if data_source != "kitti_stereo_sf" and cfg.data.stereo_fov_filtering is not None:
        ds = ds.map(
            lambda sample: filter_by_angle_fov(sample, **cfg.data.stereo_fov_filtering)
        )

    if filter_bev:
        ds = ds.map(lambda x: filter_using_bev(x, bev_extent=cfg.data.bev_extent))
    if num_input_points is None:
        num_input_points = cfg.data.num_input_points
    if num_input_points > 0:
        ds = ds.map(lambda x: apply_subset_sampling(x, num_points=num_input_points))

    def for_compatability_add_flow_t0_t1_back_to_top_level_dict(sample):
        for t in ["t0_t1", "t1_t0"]:
            if "flow_gt_%s" % t in sample:
                sample["flow_%s" % t] = sample["flow_gt_%s" % t]["flow"]
        return sample

    check_dataset_interface(ds)
    ds = ds.map(for_compatability_add_flow_t0_t1_back_to_top_level_dict)
    ds = ds.map(
        lambda x: add_moving_mask(
            x,
            labelmap=labelmap,
            non_rigid_flow_threshold=cfg.data.non_rigid_flow_threshold,
            ignore_contradicting_flow_semseg=ignore_contradicting_flow_semseg,
        )
    )

    def add_nearest_neighbor_points_to_center(
        sample, *, nbr_points: int, center: Tuple[float, float, float]
    ):
        center = np.array(center).astype(np.float32)
        for t in ["t0", "t1"]:
            point_dists = tf.linalg.norm(sample["pcl_%s" % t] - center, axis=-1)
            assert len(point_dists.shape) == 1
            _, indices = tf.math.top_k(
                -point_dists,  # -dists as top_k looks for largest, but we want smallest dists
                k=tf.minimum(nbr_points, tf.shape(point_dists)[0]),
                sorted=False,
            )
            sample["nearest_neighbor_points_mask_%s" % t] = tf.scatter_nd(
                indices[:, None],
                tf.ones_like(indices, dtype=tf.bool),
                shape=tf.shape(point_dists),
            )
        return sample

    ds = ds.map(
        lambda sample: add_nearest_neighbor_points_to_center(
            sample, **cfg.data.nn_filter_for_metrics
        )
    )

    if return_before_batch_and_voxelization:
        return ds, filenames

    ds = padded_batch(
        ds, batch_size, padding_value_int=-1, padding_value_bool=False, verbose=True
    )

    if voxel_cfg is None:
        # this mode is for data loading in external repos which don't need voxelization
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds, filenames

    ds = ds.map(create_configurable_voxelization_map(voxel_cfg))
    ds = ds.map(
        lambda x: compute_pointwise_voxel_coordinates(
            x,
            grid_size=cfg.model.point_pillars.nbr_pillars,
            bev_extent=cfg.data.bev_extent,
        )
    )

    def add_bev_mask(sample, *, grid_size: Tuple[int, int], final_scale: int):
        for t in ["t0", "t1"]:
            pclkey = "pcl_%s" % t
            shape = tf.shape(sample[pclkey]["pc"])
            bs, maxN = shape[0], shape[1]
            bidxs = tf.tile(tf.range(bs)[:, None, None], [1, maxN, 1])
            sample[pclkey]["bev_pillar_mask"] = tf.scatter_nd(
                indices=tf.concat(
                    [bidxs, sample[pclkey]["pointwise_voxel_coors"]], axis=2
                ),
                updates=sample[pclkey]["pointwise_valid_mask"],
                shape=[bs, *grid_size],
            )
            assert sample[pclkey]["bev_pillar_mask"].dtype == tf.bool
            if final_scale > 1:
                sample[pclkey]["bev_pillar_fs_mask"] = tf.scatter_nd(
                    indices=tf.concat(
                        [bidxs, sample[pclkey]["pointwise_voxel_coors"] // final_scale],
                        axis=2,
                    ),
                    updates=sample[pclkey]["pointwise_valid_mask"],
                    shape=[bs, *[g // final_scale for g in grid_size]],
                )
                assert sample[pclkey]["bev_pillar_fs_mask"].dtype == tf.bool
            else:
                assert final_scale == 1
                sample[pclkey]["bev_pillar_fs_mask"] = sample[pclkey]["bev_pillar_mask"]
        return sample

    ds = ds.map(
        lambda x: add_bev_mask(
            x,
            grid_size=cfg.model.point_pillars.nbr_pillars,
            final_scale=cfg.model.u_net.final_scale,
        )
    )

    ds = ds.map(
        lambda x: compute_gt_flow_bev_maps(
            x,
            grid_size=cfg.model.point_pillars.nbr_pillars,
        )
    )
    ds = ds.map(
        lambda x: add_ohe_gt_stat_dyn_ground_label_bev_maps(
            x,
            grid_size=cfg.model.point_pillars.nbr_pillars,
            labelmap=labelmap,
            final_scale=cfg.model.u_net.final_scale,
        )
    )

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds, filenames
