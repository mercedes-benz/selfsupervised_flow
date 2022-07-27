# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT

import os
import sys
from argparse import ArgumentParser
from collections import namedtuple
from functools import partial

import h5py
import numpy as np
import png
import pykitti
from joblib import Parallel, delayed
from tqdm import tqdm
from unsup_flow.datasets.kitti_sf.weighted_pc_alignment import (
    weighted_pc_alignment_wrapper_homog_trafo,
)

__alpha_t_deg = np.deg2rad(0.8)


def infer_ground_label(pcl: np.array, z_0=-1.70, alpha_t_deg=__alpha_t_deg):
    d_xy = np.linalg.norm(pcl[..., 0:2], axis=-1)
    z_t_thresh = z_0 + np.tan(alpha_t_deg) * d_xy
    is_ground = pcl[..., 2] < z_t_thresh
    return is_ground


def get_labels(static_mask, dynamic_mask, ground_mask, mapping):
    labels = -np.ones_like(static_mask, dtype=np.int32)
    labels[static_mask] = mapping["STATIC"]
    labels[dynamic_mask] = mapping["DYNAMIC"]
    labels[ground_mask] = mapping["GROUND"]
    unknown_mask = np.logical_and(
        ~ground_mask, np.logical_and(~static_mask, ~dynamic_mask)
    )
    labels[unknown_mask] = mapping["UNKNOWN"]
    assert np.all(labels >= 0)
    return labels


def infer_label_map(artificial_dense_flow, odom_t0_t1, pts_t0, valid_mask_02_t0):

    LabelMapping = {"UNKNOWN": 0, "GROUND": 7, "STATIC": 1, "DYNAMIC": 10}

    homog_pts_t0 = pts_t0.copy()
    homog_pts_t0[..., -1] = 1.0
    static_flow_t0_t1, _ = static_flow_and_proj_points_from_homogenous_points(
        homog_pts_t0, odom_t0_t1
    )
    pts_t0_is_static_mask = (
        np.linalg.norm(artificial_dense_flow - static_flow_t0_t1, axis=-1) <= 0.05
    )
    pts_t0_ground_mask = infer_ground_label(pts_t0)
    pts_t0_dynamic_mask = np.logical_and(~pts_t0_is_static_mask, ~pts_t0_ground_mask)
    pts_t0_is_static_mask[~valid_mask_02_t0] = False
    pts_t0_dynamic_mask[~valid_mask_02_t0] = False
    labels_t0 = get_labels(
        pts_t0_is_static_mask, pts_t0_dynamic_mask, pts_t0_ground_mask, LabelMapping
    )
    return labels_t0


def parse_calibration_cam2cam(calib_dir, flow_file_idx):
    sidx = "{0:06d}".format(flow_file_idx)
    calib_path = os.path.join(calib_dir, sidx + ".txt")
    with open(calib_path) as fd:
        lines = fd.readlines()
        assert len([line for line in lines if line.startswith("P_rect_02")]) == 1
        P_rect_left = np.array(
            [
                float(item)
                for item in [line for line in lines if line.startswith("P_rect_02")][
                    0
                ].split()[1:]
            ],
            dtype=np.float32,
        ).reshape(3, 4)

    return P_rect_left


def parse_mapping(kitti_sf_dir):
    FlowFrame = namedtuple(
        "FlowFrame",
        ["file_idx", "drive_str", "folder", "date", "raw_frame", "raw_frame_str"],
    )
    with open(
        os.path.join(kitti_sf_dir, "devkit", "mapping", "train_mapping.txt"), "r"
    ) as mapping_file:
        file_mapping = []
        for file_idx, line in enumerate(mapping_file):
            line = line.rstrip()
            if line:
                elements = line.split(" ")
                date = elements[0]
                folder = elements[1]
                raw_frame_str = elements[2]
                raw_frame = int(raw_frame_str)
                drive_str = folder.split("_")[4]
                mapping_tuple = FlowFrame(
                    file_idx=file_idx,
                    date=date,
                    drive_str=drive_str,
                    folder=folder,
                    raw_frame=raw_frame,
                    raw_frame_str=raw_frame_str,
                )
            else:
                mapping_tuple = None
            file_mapping.append(mapping_tuple)

        assert len(file_mapping) > 0
    return file_mapping


def load_optical_flow_from_png(fpath):
    optical_flow_img = load_png(fpath, dtype=np.uint16)
    valid = optical_flow_img[..., -1] == 1
    optical_flow_img = optical_flow_img.astype(np.float32)
    flow = (optical_flow_img[..., :-1] - 2 ** 15) / 64.0
    return flow, valid


def load_png(path_to_png, dtype=np.uint8):
    assert dtype in (np.uint8, np.uint16)
    img = png.Reader(path_to_png).read()
    flat_pixels = np.vstack(tuple(map(dtype, img[2])))
    if img[3]["planes"] == 3:
        width, height = img[:2]
        flat_pixels = flat_pixels.reshape(height, width, 3)
    return flat_pixels


def load_disparity_from_png(fpath):
    raw_disparity = load_png(fpath, dtype=np.uint16)
    valid_mask = raw_disparity > 0
    disparity = 1 / 256.0 * raw_disparity.astype(np.float32)
    disparity[~valid_mask] = -1.0
    return disparity, valid_mask


def static_flow_and_proj_points_from_homogenous_points(
    points_homog: np.array, odom_t0_t1: np.array
):
    assert points_homog.shape[-1] == 4
    assert np.all(points_homog[..., -1] == 1.0)
    inv_odom = np.linalg.inv(odom_t0_t1)
    proj_points_homog = np.einsum("ij,nj->ni", inv_odom, points_homog)

    flow = proj_points_homog - points_homog
    flow = flow[:, 0:3]
    return flow, proj_points_homog


def convert_stereo_disparity_to_3d_pcl(
    disparity: np.array,
    focal_length_pixels,
    stereo_baseline_meters,
    cam_principal_point_y_pixel,
    cam_principal_point_x_pixel,
):
    row_indices, col_indices = np.mgrid[0 : disparity.shape[0], 0 : disparity.shape[1]]

    x = (col_indices - cam_principal_point_x_pixel) / disparity
    y = (row_indices - cam_principal_point_y_pixel) / disparity
    z = focal_length_pixels / disparity
    pcl_3d = stereo_baseline_meters * np.stack([x, y, z], axis=-1)
    return pcl_3d


def stereo_flow_to_3d_flow_cam_coords(
    optical_flow_u: np.array,  # column flow
    optical_flow_v: np.array,  # row flow
    disparity: np.array,
    disparity_change: np.array,
    focal_length_pixels,
    stereo_baseline_meters,
    cam_principal_point_y_pixel,
    cam_principal_point_x_pixel,
):
    row_indices, col_indices = np.mgrid[
        0 : optical_flow_u.shape[0], 0 : optical_flow_u.shape[1]
    ]

    v_x = (col_indices + optical_flow_u - cam_principal_point_x_pixel) / (
        disparity + disparity_change
    ) - (col_indices - cam_principal_point_x_pixel) / disparity
    v_y = (row_indices + optical_flow_v - cam_principal_point_y_pixel) / (
        disparity + disparity_change
    ) - (row_indices - cam_principal_point_y_pixel) / disparity
    v_z = (
        focal_length_pixels / (disparity + disparity_change)
        - focal_length_pixels / disparity
    )
    flow_3d = stereo_baseline_meters * np.stack([v_x, v_y, v_z], axis=-1)
    return flow_3d


def refine_odometry(pcl_t0, flow_t0_t1, odom_t0_t1):
    assert pcl_t0.shape[-1] == 4
    assert np.all(pcl_t0[..., -1] == 1.0)
    pcl_t0_homog = np.copy(pcl_t0)
    initial_thresh = 0.05

    static_flow_t0_t1, _ = static_flow_and_proj_points_from_homogenous_points(
        pcl_t0_homog, odom_t0_t1
    )

    num_static_pts = np.count_nonzero(
        (np.linalg.norm(flow_t0_t1 - static_flow_t0_t1, axis=-1) <= initial_thresh)
    )

    if num_static_pts < 0.3 * pcl_t0_homog.shape[0]:
        thresh = initial_thresh
        while num_static_pts < 0.3 * pcl_t0_homog.shape[0]:
            thresh += 0.01
            static_flow_t0_t1, _ = static_flow_and_proj_points_from_homogenous_points(
                pcl_t0_homog, odom_t0_t1
            )
            static_pts_mask = (
                np.linalg.norm(flow_t0_t1 - static_flow_t0_t1, axis=-1) <= thresh
            )
            num_static_pts = np.count_nonzero(static_pts_mask)
        src_pcl = pcl_t0[..., 0:3][static_pts_mask]
        target_pcl = pcl_t0[..., 0:3][static_pts_mask] + flow_t0_t1[static_pts_mask]
        trafo = weighted_pc_alignment_wrapper_homog_trafo(
            src_pcl.T,
            target_pcl.T,
            np.ones_like(src_pcl[..., 0], dtype=np.float),
        )
        odom_t0_t1 = np.linalg.inv(trafo)
        static_flow_t0_t1, _ = static_flow_and_proj_points_from_homogenous_points(
            pcl_t0_homog, odom_t0_t1
        )
        static_pts_mask = (
            np.linalg.norm(flow_t0_t1 - static_flow_t0_t1, axis=-1) <= 0.01
        )
        num_static_pts = np.count_nonzero(static_pts_mask)

    return odom_t0_t1


def compute_and_save_kitti_lidar_scene_flow_hdf5(
    data_element,
    calib_dir,
    kitti_raw_dir,
    op_flow_root,
    disp0_root,
    disp1_root,
    target_folder,
):
    if data_element:
        P_rect_left = parse_calibration_cam2cam(calib_dir, data_element.file_idx)

        dataset = pykitti.raw(
            os.path.join(kitti_raw_dir),
            date=data_element.date,
            drive=data_element.drive_str,
        )

        oxts_t0_index = [
            i
            for i, v in enumerate(dataset.oxts_files)
            if data_element.raw_frame_str in v
        ][0]
        w_T_imu_t0 = dataset.oxts[oxts_t0_index].T_w_imu
        w_T_imu_t1 = dataset.oxts[oxts_t0_index + 1].T_w_imu

        # transform from imu to velo is consistent for all sequences
        Tr_imu_to_velo_kitti = (
            "9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 "
            "-7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 "
            "2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01 "
            "0.0 0.0 0.0 1.0"
        )

        velo_T_imu = np.fromstring(
            Tr_imu_to_velo_kitti, dtype=np.float, sep=" "
        ).reshape((4, 4))
        imu_T_velo = np.linalg.inv(velo_T_imu)

        w_T_velo_t0 = np.matmul(w_T_imu_t0, imu_T_velo)
        w_T_velo_t1 = np.matmul(w_T_imu_t1, imu_T_velo)

        odom_velo_t0_t1 = np.matmul(np.linalg.inv(w_T_velo_t0), w_T_velo_t1)

        png_str = "{0:06d}_10.png".format(data_element.file_idx)

        op_flow, valid_op_flow = load_optical_flow_from_png(
            os.path.join(op_flow_root, png_str)
        )

        focal_length_pixel = P_rect_left[0, 0]
        principal_point_x_pixel = P_rect_left[0, 2]
        principal_point_y_pixel = P_rect_left[1, 2]
        stereo_baseline_m = 0.54

        disp0_path = os.path.join(disp0_root, png_str)
        disp0, valid_disp0 = load_disparity_from_png(disp0_path)
        disp1_path = os.path.join(disp1_root, png_str)
        disp1, valid_disp1 = load_disparity_from_png(disp1_path)
        valid_disp = np.logical_and(valid_disp0, valid_disp1)

        valid_data_mask = np.logical_and(valid_disp, valid_op_flow)
        assert (
            np.count_nonzero(valid_disp0) - np.count_nonzero(valid_data_mask) <= 400
        ), "data loss!"
        assert (
            np.count_nonzero(valid_disp1) - np.count_nonzero(valid_data_mask) <= 400
        ), "data loss!"
        pcl_t0_cam2 = convert_stereo_disparity_to_3d_pcl(
            disp0,
            focal_length_pixel,
            stereo_baseline_m,
            cam_principal_point_y_pixel=principal_point_y_pixel,
            cam_principal_point_x_pixel=principal_point_x_pixel,
        )
        pcl_t0_cam2 = pcl_t0_cam2[valid_data_mask]
        pts_t0_cam2 = np.concatenate(
            [pcl_t0_cam2, np.ones_like(pcl_t0_cam2[..., -1:])], axis=-1
        )

        T_velo_cam2 = np.linalg.inv(dataset.calib.T_cam2_velo)
        pts_t0_velo = np.einsum("ij, nj -> ni", T_velo_cam2, pts_t0_cam2)

        pcl_t1_cam2 = convert_stereo_disparity_to_3d_pcl(
            disp1,
            focal_length_pixel,
            stereo_baseline_m,
            cam_principal_point_y_pixel=principal_point_y_pixel,
            cam_principal_point_x_pixel=principal_point_x_pixel,
        )
        pcl_t1_cam2 = pcl_t1_cam2[valid_data_mask]
        pts_t1_cam2 = np.concatenate(
            [pcl_t1_cam2, np.ones_like(pcl_t1_cam2[..., -1:])], axis=-1
        )
        pts_t1_velo = np.einsum("ij, nj -> ni", T_velo_cam2, pts_t1_cam2)

        if np.count_nonzero(valid_disp1) != np.count_nonzero(valid_disp0):
            print(
                data_element.file_idx,
                ": ",
                np.count_nonzero(valid_disp1),
                " vs ",
                np.count_nonzero(valid_disp0),
                " remaining: ",
                np.count_nonzero(valid_data_mask),
            )

        disparity_change = disp1 - disp0

        flow3d_cam_02 = stereo_flow_to_3d_flow_cam_coords(
            op_flow[..., 0],
            op_flow[..., 1],
            disp0,
            disparity_change,
            focal_length_pixel,
            stereo_baseline_m,
            principal_point_y_pixel,
            principal_point_x_pixel,
        )

        flow3d = flow3d_cam_02[valid_data_mask]
        flow3d_cam_02_homog = np.concatenate(
            [flow3d, np.zeros_like(flow3d[..., -1:])], axis=-1
        )
        flow3d_velo = np.einsum("ij, nj -> ni", T_velo_cam2, flow3d_cam_02_homog)[
            ..., 0:3
        ]
        target_file = os.path.join(
            target_folder, str(data_element.file_idx).zfill(4) + ".hdf5"
        )

        odom_velo_t0_t1 = refine_odometry(pts_t0_velo, flow3d_velo, odom_velo_t0_t1)

        labels_t0 = infer_label_map(
            flow3d_velo,
            odom_velo_t0_t1,
            pts_t0_velo,
            np.ones_like(pts_t0_velo[..., 0], dtype=np.bool),
        )
        pts_t0_plus_flow = np.copy(pts_t0_velo)
        pts_t0_plus_flow[..., 0:3] += flow3d_velo

        with h5py.File(target_file, "w") as f:
            f.create_dataset("pts_t0", data=pts_t0_velo, dtype=np.float32)
            f.create_dataset(
                "pts_t0_plus_flow", data=pts_t0_plus_flow, dtype=np.float32
            )
            f.create_dataset("pts_t1", data=pts_t1_velo, dtype=np.float32)
            f.create_dataset("labels_t0", data=labels_t0, dtype=np.int32)
            # flow t0 -> t1
            f.create_dataset("flow_gt_t0_t1", data=flow3d_velo, dtype=np.float32)
            f.create_dataset("odometry_t0_t1", data=odom_velo_t0_t1, dtype=np.float32)

            f.create_dataset("meta_utf8", data=str(data_element).encode("utf-8"))

        return True
    else:
        return False


def main():
    argparser = ArgumentParser(description="Convert Kitti Stereo Flow")
    argparser.add_argument(
        "-t",
        "--target_dir",
        default=os.path.join(
            os.getenv("INPUT_DATADIR", "INPUT_DATADIR_ENV_NOT_DEFINED"),
            "prepped_datasets",
            "kitti_stereo_sf",
        ),
        help="place where converted data will be dumped",
    )
    argparser.add_argument(
        "--kitti_raw_dir",
        default=os.path.join(
            os.getenv("INPUT_DATADIR", "INPUT_DATADIR_ENV_NOT_DEFINED"), "kitti_raw"
        ),
        help="place where kitti raw data will be loaded from",
    )
    argparser.add_argument(
        "--kitti_sf_dir",
        default=os.path.join(
            os.getenv("INPUT_DATADIR", "INPUT_DATADIR_ENV_NOT_DEFINED"),
            "kitti_sf_data",
        ),
        help="place where kitti scene flow data will be loaded from",
    )
    args = argparser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    disparity_0_root_dir = os.path.join(args.kitti_sf_dir, "training/disp_occ_0")
    disparity_1_root_dir = os.path.join(args.kitti_sf_dir, "training/disp_occ_1")
    optical_flow_root_dir = os.path.join(args.kitti_sf_dir, "training/flow_occ")
    calib_dir = os.path.join(args.kitti_sf_dir, "training/calib_cam_to_cam")
    flow_id_to_raw_mapping = parse_mapping(args.kitti_sf_dir)

    print(
        "Found {0} potential kitti_sf <-> kitti_raw mappings".format(
            len(flow_id_to_raw_mapping)
        )
    )
    print("Writing converted files to {0}".format(args.target_dir))

    save_flow = partial(
        compute_and_save_kitti_lidar_scene_flow_hdf5,
        calib_dir=calib_dir,
        kitti_raw_dir=args.kitti_raw_dir,
        op_flow_root=optical_flow_root_dir,
        disp0_root=disparity_0_root_dir,
        disp1_root=disparity_1_root_dir,
        target_folder=args.target_dir,
    )

    success_stats = Parallel(n_jobs=12)(
        delayed(save_flow)(data_element)
        for data_element in tqdm(flow_id_to_raw_mapping)
    )

    success_mask = np.array(success_stats)

    print(
        "Wrote {0} files to {1}!".format(
            np.count_nonzero(success_mask), args.target_dir
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
