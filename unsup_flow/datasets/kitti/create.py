#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import argparse
import os

import numpy as np
import pykitti
from tqdm import tqdm

from tfrecutils import write_tfrecord


def main():
    parser = argparse.ArgumentParser(
        description="Process Kitti Raw Dataset to tfrecords."
    )
    parser.add_argument(
        "--kitti_raw_dir",
        default=os.path.join(
            os.getenv("INPUT_DATADIR", "INPUT_DATADIR_ENV_NOT_DEFINED"), "kitti_raw"
        ),
        help="location of kitti raw dataset",
    )
    parser.add_argument(
        "--target_dir",
        default=os.path.join(
            os.getenv("INPUT_DATADIR", "INPUT_DATADIR_ENV_NOT_DEFINED"),
            "prepped_datasets",
            "kitti_lidar_raw",
        ),
        help="target dir for tfrecords of kitti raw dataset",
    )

    args = parser.parse_args()
    raw_base_dir = args.kitti_raw_dir
    target_dir = args.target_dir

    dates = ["2011_09_26", "2011_09_28", "2011_09_29", "2011_09_30", "2011_10_03"]

    dimension_dict = {
        "pcl_t0": [-1, 4],
        "pcl_t1": [-1, 4],
        "odom_t0_t1": [4, 4],
    }
    meta = {"framerate__Hz": 10.0}
    skipped_sequences = 0
    success = 0
    for date in tqdm(dates):

        drives_strs = [str(i).zfill(4) for i in range(1000)]

        for drive_str in tqdm(drives_strs):
            try:
                kitti = pykitti.raw(raw_base_dir, date, drive_str)
            except FileNotFoundError:
                skipped_sequences += 1
                # print("Skipping {0} {1}".format(date, drive_str))
                continue

            for idx in range(0, len(kitti.velo_files) - 1, 1):
                pcl_t0 = pykitti.utils.load_velo_scan(kitti.velo_files[idx])
                pcl_t1 = pykitti.utils.load_velo_scan(kitti.velo_files[idx + 1])

                w_T_imu_t0 = kitti.oxts[idx].T_w_imu
                w_T_imu_t1 = kitti.oxts[idx + 1].T_w_imu
                imu_T_velo = np.linalg.inv(kitti.calib.T_velo_imu)

                w_T_velo_t0 = np.matmul(w_T_imu_t0, imu_T_velo)
                w_T_velo_t1 = np.matmul(w_T_imu_t1, imu_T_velo)

                odom_t0_t1 = np.matmul(np.linalg.inv(w_T_velo_t0), w_T_velo_t1)
                sample_name = "{0}_{1}_{2}".format(date, drive_str, str(idx).zfill(10))
                data_dict = {
                    "pcl_t0": pcl_t0.astype(np.float32),
                    "pcl_t1": pcl_t1.astype(np.float32),
                    "odom_t0_t1": odom_t0_t1.astype(np.float64),
                    "name": sample_name,
                }

                write_tfrecord(
                    data_dict,
                    os.path.join(target_dir, sample_name),
                    verbose=False,
                    dimension_dict=dimension_dict,
                    meta=meta,
                )
                success += 1
    print("Skipped: {0} Success: {1}".format(skipped_sequences, success))


if __name__ == "__main__":
    main()
