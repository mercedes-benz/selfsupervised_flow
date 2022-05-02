#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import functools
import os.path as osp
import warnings
from typing import Any, Dict, List, Union

import numpy as np
import scipy
from nuscenes import NuScenes

from npimgtools import Transform


def plerp(t0__us, t1__us, p0, p1, *args, max_rotation_angle__deg_per_fr=90.0):
    assert t0__us < t1__us
    max_rotation_angle__deg = max(
        min(max_rotation_angle__deg_per_fr * (t1__us - t0__us) / 5e5, 179.0), 1.0
    )
    trafo = np.matmul(p1, np.linalg.inv(p0))
    ln_trafo = scipy.linalg.logm(trafo)
    assert (p0[:3, 0] * p1[:3, 0]).sum() >= np.cos(
        max_rotation_angle__deg / 180.0 * np.pi
    ) * np.linalg.norm(p0[:3, 0]) * np.linalg.norm(p1[:3, 0]), print(
        "\n\nmax_rotation_angle__deg_per_fr\n\n",
        max_rotation_angle__deg_per_fr,
        "\n\nmax_rotation_angle__deg\n\n",
        max_rotation_angle__deg,
        "\n\nmeasured angle\n\n",
        np.arccos(
            np.clip(
                (p0[:3, 0] * p1[:3, 0]).sum()
                / (np.linalg.norm(p0[:3, 0]) * np.linalg.norm(p1[:3, 0])),
                -1.0,
                1.0,
            )
        )
        / np.pi
        * 180.0,
        "\n\nt0\n\n",
        t0__us,
        "\n\nt1\n\n",
        t1__us,
        "\n\np0\n\n",
        p0,
        "\n\np1\n\n",
        p1,
        "\n\ntrafo\n\n",
        trafo,
        "\n\nln_trafo\n\n",
        ln_trafo,
        *args
    )

    def plerp_function(t):
        f1 = (t - t0__us) / (t1__us - t0__us)
        result_trafo = scipy.linalg.expm(f1 * ln_trafo)
        abs_result_trafo = np.abs(result_trafo)
        assert np.allclose(abs_result_trafo, np.abs(result_trafo.real)), print(
            abs_result_trafo,
            result_trafo.real,
            result_trafo,
            "\n\nt0\n\n",
            t0__us,
            "\n\nt1\n\n",
            t1__us,
            "\n\np0\n\n",
            p0,
            "\n\np1\n\n",
            p1,
            "\n\ntrafo\n\n",
            trafo,
            "\n\nln_trafo\n\n",
            ln_trafo,
            "\n\nf1\n\n",
            f1,
            *args
        )
        assert np.allclose(result_trafo.imag, 0.0)
        return np.matmul(result_trafo.real, p0)

    return plerp_function


class NuScenesParser(NuScenes):
    def get_annotation_pose_EGO__m(self, ann_tok):
        if isinstance(ann_tok, str):
            return self.get_annotation_pose_EGO__m(
                self.get("sample_annotation", ann_tok)
            )

        assert "token" in ann_tok
        ann = ann_tok

        sample = self.get("sample", ann["sample_token"])
        scene_token = sample["scene_token"]
        ep = self.get_ego_pose_at_timestamp(scene_token, sample["timestamp"])
        ep_inv = ep.copy().invert()
        return ep_inv * Transform(ann)

    def is_annotation_dynamic(self, annotation):
        ann_toks = [annotation["prev"], annotation["token"], annotation["next"]]
        ann_toks = [t for t in ann_toks if t != ""]
        ann_recs = [self.get("sample_annotation", tok) for tok in ann_toks]
        category = self.get(
            "category",
            self.get("instance", annotation["instance_token"])["category_token"],
        )["name"]

        max_loc_err = 0.0
        for prev_rec, rec in zip(ann_recs[:-1], ann_recs[1:]):
            prev_pose = Transform(prev_rec)
            cur_pose = Transform(rec).as_htm()
            loc_err = np.linalg.norm(prev_pose.as_htm()[:2, 3] - cur_pose[:2, 3])
            max_loc_err = max(max_loc_err, loc_err)
        if (
            category
            in {
                "static_object.bicycle_rack",
                "movable_object.trafficcone",
                "movable_object.debris",
                "movable_object.barrier",
                "movable_object.pushable_pullable",
            }
            and max_loc_err < 0.1
        ):
            return False
        attr_toks = functools.reduce(
            lambda a, b: a + b,
            [rec["attribute_tokens"] for rec in ann_recs],
            [],
        )
        for attr_tok in attr_toks:
            attr = self.get("attribute", attr_tok)["name"]
            if attr in {
                "pedestrian.moving",
                "vehicle.moving",
            }:
                return True
        return max_loc_err > 0.1

    def get_token_list(
        self,
        table_name: str,
        start_token: str,
        recurse_by: int = 0,
        reverse: bool = False,
        check_if_start: bool = True,
        return_records: bool = False,
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """Returns a list of tokens corresponding to table name.

        Parameters
        ----------
        table_name : str
            The name of the table which shall be iterated.
        start_token : str
            The start or reference token (see recurse_by) to start off.
        recurse_by : int
            If >=0 it goes back by this many elements or when at start
            before starting the list. When negative, recurses to the beginning
            of the corresponding sequence (the default is 0).
        reverse : bool
            If true, the sequence is traversed in the other way around.
            recurse_by and check_if_start refer to the reversed sequence (the default is False).
        check_if_start : bool
            Checks if the start_token (after recursing) is
            the start token of the sequence (the default is True).
        return_records : bool
            Specifies if the corresponding records should be returned
            (the default is False).

        Returns
        -------
        List[str]
            Returns list of table tokens.

        Raises
        -------
        AssertionError
            When check_if_start is True but the computed start_token
            (based on recurse_by) is not the start of the sequence.
        """
        next_kw = "next"
        prev_kw = "prev"
        if reverse:
            next_kw = "prev"
            prev_kw = "next"

        cur_rec = self.get(table_name, start_token)
        while recurse_by != 0 and cur_rec[prev_kw] != "":
            cur_rec = self.get(table_name, cur_rec[prev_kw])
            recurse_by -= 1
        assert recurse_by <= 0
        start_token = cur_rec["token"]

        if check_if_start:
            assert cur_rec[prev_kw] == "", "not start of the sequence"

        tokens = [start_token]
        recs = [cur_rec]
        while cur_rec[next_kw] != "":
            tokens.append(cur_rec[next_kw])
            cur_rec = self.get(table_name, tokens[-1])
            recs.append(cur_rec)

        assert all(t == r["token"] for t, r in zip(tokens, recs))
        assert recs[-1][next_kw] == ""

        if return_records:
            return recs
        return tokens

    def get_interpolated_instance_poses__m(self, instance, timestamps__us):
        start_ann_token = instance["first_annotation_token"]
        ann_toks = self.get_token_list("sample_annotation", start_ann_token)
        ann_recs = [self.get("sample_annotation", tok) for tok in ann_toks]

        gt_timestamps__us = [
            self.get("sample", rec["sample_token"])["timestamp"] for rec in ann_recs
        ]
        gt_poses = [Transform(rec) for rec in ann_recs]

        # if not self.is_instance_dynamic(instance) or len(gt_timestamps__us) == 1:
        if len(gt_timestamps__us) == 1:
            return [gt_poses[0].copy() for _ in timestamps__us]

        if not self.is_annotation_dynamic(ann_recs[0]):
            t0__us = gt_timestamps__us[0]
            t1__us = gt_timestamps__us[1]
            gt_timestamps__us = [2 * t0__us - t1__us] + gt_timestamps__us
            gt_poses = [gt_poses[0].copy()] + gt_poses
            ann_toks = [None] + ann_toks
        if not self.is_annotation_dynamic(ann_recs[-1]):
            t0__us = gt_timestamps__us[-1]
            t1__us = gt_timestamps__us[-2]
            gt_timestamps__us = gt_timestamps__us + [2 * t0__us - t1__us]
            gt_poses = gt_poses + [gt_poses[-1].copy()]
            ann_toks = ann_toks + [None]
        gt_timestamps__us = np.array(gt_timestamps__us)
        assert all(
            t0__us < t1__us
            for t0__us, t1__us in zip(gt_timestamps__us[:-1], gt_timestamps__us[1:])
        ), (gt_timestamps__us * 1e-6 - gt_timestamps__us[0] * 1e-6)

        assert len(gt_timestamps__us) >= 2

        result = []
        all_plerp_funcs = {}
        for t in timestamps__us:
            idx_mask = (t >= gt_timestamps__us[:-1]) & (t < gt_timestamps__us[1:])
            assert idx_mask.sum() <= 1
            if idx_mask.sum() == 1:
                idx = np.where(idx_mask)[0][0]
            else:
                assert t < gt_timestamps__us[0] or t >= gt_timestamps__us[-1]
                warnings.warn("interpolating instance pose far outside time window")
                # warnings.warn(
                #     "interpolating instance pose far outside time window %.2f: %.2f"
                #     % (
                #         (gt_timestamps__us[-1] - gt_timestamps__us[0]) * 1e-6,
                #         (t - gt_timestamps__us[0]) * 1e-6,
                #     )
                # )
                if t < gt_timestamps__us[0]:
                    idx = 0
                else:
                    idx = len(gt_timestamps__us) - 2
            assert idx <= len(gt_timestamps__us) - 2
            t0__us = gt_timestamps__us[idx]
            t1__us = gt_timestamps__us[idx + 1]
            p0 = gt_poses[idx]
            p1 = gt_poses[idx + 1]
            assert t0__us < t1__us, (t0__us, t1__us)
            assert len(gt_timestamps__us) == len(ann_toks), print(
                len(gt_timestamps__us), len(ann_toks)
            )

            if idx not in all_plerp_funcs:
                all_plerp_funcs[idx] = plerp(
                    t0__us,
                    t1__us,
                    p0.as_htm(),
                    p1.as_htm(),
                    "\n\nann_tok\n\n",
                    ann_toks[idx],
                    max_rotation_angle__deg_per_fr=360.0
                    if any(
                        s in ann_recs[0]["category_name"]
                        for s in {"movable_object", "human", "animal"}
                    )
                    else 65.0,
                )

            result.append(Transform().set_htm(all_plerp_funcs[idx](t)))

        return result

    def get_lidar_semseg(self, sample):
        if isinstance(sample, str):
            sample = self.get("sample", sample)
        lidarseg = self.get("lidarseg", sample["data"]["LIDAR_TOP"])
        semseg = np.fromfile(
            osp.join(self.dataroot, lidarseg["filename"]), dtype=np.uint8
        )
        return semseg

    def get_pointcloud(self, sample_data, ref_frame=None, remove_ego_points=True):
        """Returns a pointcloud from the nuscenes dataset with an invalid/ego vehicle points mask.

        Parameters
        ----------
        sample_data : dict
            A nuscenes sample_data dict, not the token!
        ref_frame : str [Optional]
            One of 'ego' or 'world'. Is the reference frame in which the pointcloud is returned.
            If not provided (default=None), then no change of frame is done, the pointcloud is then measured relative to the sensor.
        remove_ego_points : bool
            Specifies if you want the pointcloud to be filtered. The filtering removes ego vehicle points and invalid laser returns.

        Returns
        -------
        np.ndarray [5, num_valid_points] (float)
            The pointcloud with x,y,z,intensity,channel_id(row) in the first dimension, and the potentially masked number of points along the second dimension.
        np.ndarray [num_all_points] (bool)
            The mask for the raw pointcloud, therefore the number does not match num_valid_points in case of remove_ego_points=True.

        Examples
        -------
        >>> nusc = NuscenesParser(version="v1.0-trainval", dataroot="/path/to/nuscenes", verbose=True)
        >>> sd = ds.get("sample_data", ds.get_sample_data_tokens()[0])
        >>> pc, mask = pc.get_pointcloud(sd, ref_frame="ego")
        """

        # if not hasattr(self, "_ego_point_decisions"):
        #     with open(
        #         osp.join(
        #             self.dataroot,
        #             "precompute",
        #             "remove_ego_points",
        #             "ego_point_decision.yml",
        #         ),
        #         "r",
        #     ) as fin:
        #         self._ego_point_decisions = yaml.safe_load(fin)
        assert sample_data["channel"] == "LIDAR_TOP"
        pcl = (
            np.fromfile(
                osp.join(self.dataroot, sample_data["filename"]), dtype=np.float32
            )
            .astype(np.float64)
            .reshape([-1, 5])
            .T
        )
        if remove_ego_points:
            sample = self.get("sample", sample_data["sample_token"])
            desc = self.get("scene", sample["scene_token"])["description"]
            rain = "rain" in desc or "Rain" in desc
            # ego point region is taken from the measurements from renault zoe
            width = 1.5
            height = 1.1
            head = 2.308
            tail = -1.419
            delta_margin = 0.1
            ego_point_mask = np.abs(pcl[0, :]) < width / 2.0
            ego_point_mask = np.logical_and(ego_point_mask, pcl[1, :] < head)
            ego_point_mask = np.logical_and(ego_point_mask, pcl[1, :] > tail)
            ego_point_mask = np.logical_and(ego_point_mask, pcl[2, :] < 0.0)
            ego_point_mask = np.logical_and(ego_point_mask, pcl[2, :] > -height)
            invalid_point_mask = np.abs(pcl[0, :]) < 0.1
            invalid_point_mask = np.logical_and(
                invalid_point_mask, np.abs(pcl[1, :]) < 1.5
            )
            invalid_point_mask = np.logical_and(
                invalid_point_mask, np.abs(pcl[2, :]) < 0.1
            )
            ego_point_mask = np.logical_or(ego_point_mask, invalid_point_mask)

            r_ego = np.max(np.sqrt(np.sum(pcl[:3, ego_point_mask] ** 2, axis=0)))
            ego_shadow_mask = (
                np.sqrt(np.sum(pcl[:3, :] ** 2, axis=0)) <= r_ego + delta_margin
            )
            ego_shadow_mask = np.logical_and(
                ego_shadow_mask, np.logical_not(ego_point_mask)
            )
            if not np.all(np.logical_not(ego_shadow_mask)):
                r_min_shadow = np.min(
                    np.sqrt(np.sum(pcl[:3, ego_shadow_mask] ** 2, axis=0))
                )
                ego_point_edge_mask = (
                    np.sqrt(np.sum(pcl[:3, :] ** 2, axis=0))
                    >= r_min_shadow - delta_margin
                )
                ego_point_edge_mask = np.logical_and(
                    ego_point_edge_mask, ego_point_mask
                )

                min_dists = np.min(
                    np.sqrt(
                        np.sum(
                            (
                                pcl[:3, ego_shadow_mask, None]
                                - pcl[:3, None, ego_point_edge_mask]
                            )
                            ** 2,
                            axis=0,
                        )
                    ),
                    axis=-1,
                )
                ego_shadow_mask[ego_shadow_mask] = min_dists < delta_margin

            if np.sum(ego_shadow_mask) > int(rain):
                assert sample_data["token"] in self._ego_point_decisions
                decs = self._ego_point_decisions[sample_data["token"]]
                point_idxs = np.array(decs["point_idxs"], dtype=np.int32)
                ego_point_mask[point_idxs] = decs["belong_to_ego_vehicle"]
                # else:
                #     import pdb; pdb.set_trace()
                #     if osp.exists(osp.join(
                #         self.dataroot, 'crazy_ego_points.yml'
                #     )):
                #         with open(osp.join(
                #             self.dataroot, 'crazy_ego_points.yml'
                #         ), 'r') as fin:
                #             log_failure = yaml.safe_load(fin)
                #     else:
                #         log_failure = {}
                #     log_failure[sample_data['token']] = (
                #         'something undocumented in the shadows'
                #     )
                #     with open(osp.join(
                #         self.dataroot, 'crazy_ego_points.yml'
                #     ), 'w') as fout:
                #         fout.write(yaml.safe_dump(log_failure))
            removed_ego_points_mask = np.logical_not(ego_point_mask)
            pcl = pcl[:, removed_ego_points_mask]
        else:
            removed_ego_points_mask = np.ones((pcl.shape[1],), dtype=np.bool)
        if ref_frame is None:
            return pcl, removed_ego_points_mask
        sensor_pose = self.get(
            "calibrated_sensor", sample_data["calibrated_sensor_token"]
        )
        Transform(sensor_pose).apply(pcl, coord_dim=0)
        if ref_frame == "ego":
            return pcl, removed_ego_points_mask
        ego_pose = self.get("ego_pose", sample_data["ego_pose_token"])
        Transform(ego_pose).apply(pcl, coord_dim=0)
        assert ref_frame == "world"
        return pcl, removed_ego_points_mask

    def get_ego_pose_at_timestamp(
        self, scene_token: str, timestamp__us: int
    ) -> Transform:
        # #region init cache for fast retrieval
        if not hasattr(self, "_scene_time_windows__us"):
            self._scene_time_windows__us = {}
            for scene in self.scene:
                cur_scene_ts_ep_dict = {}
                sample = self.get("sample", scene["first_sample_token"])
                for channel in sample["data"]:
                    sample_data = self.get("sample_data", sample["data"][channel])
                    assert sample_data["prev"] == ""
                    sample_data_token = sample_data["token"]
                    while sample_data_token != "":
                        sample_data = self.get("sample_data", sample_data_token)
                        ts = sample_data["timestamp"]
                        cur_scene_ts_ep_dict[ts] = sample_data["ego_pose_token"]
                        sample_data_token = sample_data["next"]
                self._scene_time_windows__us[scene["token"]] = {
                    "sorted_ts": np.sort(np.array(list(cur_scene_ts_ep_dict.keys()))),
                    "ts_ep_dict": cur_scene_ts_ep_dict,
                }
        # #endregion init cache for fast retrieval
        sorted_ts = self._scene_time_windows__us[scene_token]["sorted_ts"]
        ts_ep_dict = self._scene_time_windows__us[scene_token]["ts_ep_dict"]
        assert sorted_ts[0] <= timestamp__us <= sorted_ts[-1]
        idx = np.argmin(np.abs(sorted_ts - timestamp__us))
        if sorted_ts[idx] > timestamp__us or idx + 1 >= len(sorted_ts):
            idx -= 1
        t1 = sorted_ts[idx]
        t2 = sorted_ts[idx + 1]
        assert (
            t1 <= timestamp__us <= t2
        ), "timestamp %d was not inside possible timestamp range %d-%d" % (
            timestamp__us,
            sorted_ts[0],
            sorted_ts[-1],
        )
        ep1 = Transform(self.get("ego_pose", ts_ep_dict[t1])).as_htm()
        ep2 = Transform(self.get("ego_pose", ts_ep_dict[t2])).as_htm()
        assert t1 < t2
        return Transform().set_htm(plerp(t1, t2, ep1, ep2)(timestamp__us))
