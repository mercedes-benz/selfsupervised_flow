#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import os.path as osp

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from labelmap import get_label_map_from_file
from tfrecutils import write_tfrecord
from unsup_flow.datasets.nuscenes.nuscenes_parser import NuScenesParser
from usfl_io.io_tools import nusc_add_nn_segmentation_flow_for_t1


def create_and_write_sample(sample, path: str, nusc: NuScenesParser) -> str:
    meta = {"framerate__Hz": 10.0}

    scene = nusc.get("scene", sample["scene_token"])
    sample_tokens = nusc.get_token_list("sample", sample["token"], recurse_by=-1)
    sample_idx = sample_tokens.index(sample["token"])
    filename = "%s_%02d_%s" % (scene["name"], sample_idx, sample["token"])
    filename = osp.join(path, filename)

    if osp.isfile(filename + ".tfrecords"):
        return "exists"

    nusc2carla_labelmap = get_label_map_from_file("nuscenes", "nuscenes2carla")
    nusc2statdynground_labelmap = get_label_map_from_file(
        "nuscenes", "nuscenes2static_dynamic_ground"
    )

    # #region compute data
    cur_sd_tok = sample["data"]["LIDAR_TOP"]
    cur_sd = nusc.get("sample_data", cur_sd_tok)

    skip = 20 / meta["framerate__Hz"]
    assert np.allclose(skip, int(round(skip))), skip
    skip = int(round(skip))
    assert skip > 0

    # take skip-times next to have the target framerate__Hz like in KITTI/CARLA
    next_sd_tokens = nusc.get_token_list(
        "sample_data", cur_sd_tok, check_if_start=False
    )
    if len(next_sd_tokens) <= skip:
        return "not enough follow up sample datas"
    next_sd_tok = next_sd_tokens[skip]
    next_sd = nusc.get("sample_data", next_sd_tok)

    t0 = sample["timestamp"]
    t1 = next_sd["timestamp"]
    frame_diff_0_1 = (t1 - t0) / (1e6 / 20.0)
    frame_diff_0_1_int = int(round(frame_diff_0_1))
    assert np.abs(frame_diff_0_1 - frame_diff_0_1_int) < 0.3
    if frame_diff_0_1_int > skip:
        assert not np.allclose(
            t1 - t0, 1e6 / meta["framerate__Hz"], rtol=0.1, atol=5000
        ), (
            t1 - t0,
            sample["token"],
        )
        tqdm.write(sample["token"] + ": missing sample data for correct frequency")
        return "irregular temporal sampling"
        # skip a few samples as there are sample data frames missing and would
        # fail the following assert
    assert frame_diff_0_1_int == skip
    assert np.allclose(t1 - t0, 1e6 / meta["framerate__Hz"], rtol=0.1, atol=5000), (
        t1 - t0,
        sample["token"],
    )

    ep_t0 = nusc.get_ego_pose_at_timestamp(sample["scene_token"], t0)
    ep_t1 = nusc.get_ego_pose_at_timestamp(sample["scene_token"], t1)
    odom_t0_t1 = ep_t0.copy().invert() * ep_t1
    odom_t1_t0 = ep_t1.copy().invert() * ep_t0
    try:
        pcl_t0, ego_mask_t0 = nusc.get_pointcloud(cur_sd, ref_frame="ego")
        unmasked_pcl_t0, _ = nusc.get_pointcloud(cur_sd, remove_ego_points=False)
        pcl_t1, ego_mask_t1 = nusc.get_pointcloud(next_sd, ref_frame="ego")
    except AttributeError:
        print("Failed to load point clouds due to missing ego_points_decisions")
        return "missing ego_points_decisions"
    semantics_t0 = nusc.get_lidar_semseg(sample)

    semantics_t0 = semantics_t0[ego_mask_t0]
    semantics_t0[
        semantics_t0 == nusc2carla_labelmap.rname_ridx_dict["vehicle.ego"]
    ] = nusc2carla_labelmap.rname_ridx_dict["vehicle.car"]

    points_found_norig = np.zeros_like(semantics_t0, dtype=np.bool)
    point_belongs_to_this_object_idx = -np.ones_like(semantics_t0, dtype=np.int32)

    assert pcl_t0.ndim == 2
    assert pcl_t0.shape[0] == 5
    pcl_t0__m_R = pcl_t0.T[:, :4]
    pcl_t1__m_R = pcl_t1.T[:, :4]
    pcl_t0 = pcl_t0__m_R[:, :3]
    pcl_t1 = pcl_t1__m_R[:, :3]

    flow_t0_t1 = pcl_t0 @ (odom_t1_t0.rot_mat() - np.eye(3)).T + odom_t1_t0.trans()

    for ann_idx, ann_tok in enumerate(sample["anns"]):
        ann = nusc.get("sample_annotation", ann_tok)
        size = np.array(ann["size"])[[1, 0, 2]]
        instance = nusc.get("instance", ann["instance_token"])
        if (
            ann["category_name"]
            not in nusc2statdynground_labelmap.mname_rnames_dict["dynamic"]
        ):
            continue
        obj_pose_EGO0_t0 = nusc.get_annotation_pose_EGO__m(ann_tok)
        inv_obj_pose_EGO0_t0 = obj_pose_EGO0_t0.copy().invert()
        try:
            obj_pose_EGO1_t1 = (
                ep_t1.copy().invert()
                * nusc.get_interpolated_instance_poses__m(
                    instance, [next_sd["timestamp"]]
                )[0]
            )
        except AssertionError:
            print("Interpolation failed!")
            return "pose interpolation failed"
        pcl_t0_OBJ0 = (
            pcl_t0 @ inv_obj_pose_EGO0_t0.rot_mat().T + inv_obj_pose_EGO0_t0.trans()
        )
        cur_obj_points_mask_t0 = (np.abs(pcl_t0_OBJ0) < size / 2.0).all(axis=-1)
        cur_semseg_mask_t0 = (
            semantics_t0 == nusc2carla_labelmap.rname_ridx_dict[ann["category_name"]]
        )
        cur_obj_points_mask_t0 = cur_obj_points_mask_t0 & cur_semseg_mask_t0

        cur_dyn_flow_trafo = obj_pose_EGO1_t1 * obj_pose_EGO0_t0.copy().invert()
        cur_dyn_flow = (
            pcl_t0 @ (cur_dyn_flow_trafo.rot_mat() - np.eye(3)).T
            + cur_dyn_flow_trafo.trans()
        )
        flow_t0_t1[cur_obj_points_mask_t0] = cur_dyn_flow[cur_obj_points_mask_t0]
        point_belongs_to_this_object_idx[cur_obj_points_mask_t0] = ann_idx
        points_found_norig[cur_obj_points_mask_t0] = True
    # #endregion compute data

    # #region check that all dynamic symantics non rigid flow got
    dynamic_semantics = np.zeros_like(semantics_t0, dtype=np.bool)
    for rname in nusc2statdynground_labelmap.mname_rnames_dict["dynamic"]:
        ridx = nusc2statdynground_labelmap.rname_ridx_dict[rname]
        assert not (dynamic_semantics & (semantics_t0 == ridx)).any()
        dynamic_semantics[semantics_t0 == ridx] = True
    assert (dynamic_semantics >= points_found_norig).all()
    stat_possibly_dyn_on_object_box_edges_mask = dynamic_semantics & (
        ~points_found_norig
    )
    # #endregion check that all dynamic symantics non rigid flow got

    tf_adaptor_sample = {
        "pcl_t0": tf.convert_to_tensor(pcl_t0__m_R[..., 0:3].astype(np.float32)),
        "pcl_t1": tf.convert_to_tensor(pcl_t1__m_R[..., 0:3].astype(np.float32)),
        "flow_gt_t0_t1": {
            "flow": tf.convert_to_tensor(flow_t0_t1.astype(np.float32)),
            "ego_flow_mask": tf.convert_to_tensor(~points_found_norig),
        },
        "odom_t0_t1": tf.convert_to_tensor(odom_t0_t1.as_htm().astype(np.float64)),
        "semantics_t0": tf.convert_to_tensor(semantics_t0.astype(np.int32)),
    }

    nn_annotated_sample = nusc_add_nn_segmentation_flow_for_t1(
        tf_adaptor_sample, add_semseg=True, add_flow=True
    )

    def recursive_dict_to_numpy(some_el):
        if isinstance(some_el, dict):
            return {k: recursive_dict_to_numpy(v) for k, v in some_el.items()}
        else:
            return some_el.numpy()

    nn_annotated_sample = recursive_dict_to_numpy(nn_annotated_sample)

    data_dict = {
        "flow_gt_t1_t0": nn_annotated_sample["flow_gt_t1_t0"],
        "pcl_t0": pcl_t0__m_R.astype(np.float32),
        "pcl_t1": pcl_t1__m_R.astype(np.float32),
        "flow_t0_t1": flow_t0_t1.astype(np.float32),
        "odom_t0_t1": odom_t0_t1.as_htm().astype(np.float64),
        "semantics_t0": semantics_t0.astype(np.int32),
        "stat_possibly_dyn_on_object_box_edges_mask": stat_possibly_dyn_on_object_box_edges_mask,
        "ego_flow_mask": ~points_found_norig,
        "semantics_t1": nn_annotated_sample["semantics_t1"],
        "name": sample["token"],
    }
    dimension_dict = {
        "pcl_t0": (-1, 4),
        "pcl_t1": (-1, 4),
        "flow_t0_t1": (-1, 3),
        "flow_t1_t0_gt": {"flow": (-1, 3)},
        "odom_t0_t1": (4, 4),
    }
    write_tfrecord(
        data_dict, filename, verbose=True, dimension_dict=dimension_dict, meta=meta
    )
    return "fine"


def main(path_out: str, nusc_root: str):
    nusc = NuScenesParser(
        version="v1.0-trainval",
        dataroot=nusc_root,
        verbose=True,
    )
    results = {}
    count_results = {}
    for sample in tqdm(nusc.sample):
        cur_result = create_and_write_sample(sample, path_out, nusc)
        results[sample["token"]] = cur_result
        if cur_result not in count_results:
            count_results[cur_result] = 0
        count_results[cur_result] += 1
        tqdm.write(str(count_results))


if __name__ == "__main__":
    import os

    INPUT_DATADIR = os.getenv("INPUT_DATADIR", "INPUT_DATADIR_ENV_NOT_DEFINED")
    main(
        osp.join(INPUT_DATADIR, "prepped_datasets", "nuscenes"),
        osp.join(INPUT_DATADIR, "nuscenes"),
    )
