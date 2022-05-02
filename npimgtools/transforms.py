#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import numpy as np
from scipy.spatial.transform import Rotation as R


class Transform:
    def __init__(self, pose_data=None, data_format: str = "NuScenes"):
        if pose_data is None:
            self.homogenuous_transformation_matrix = np.eye(4, dtype=np.float64)
            return

        if data_format == "NuScenes":
            T = R.from_quat(
                np.array(
                    pose_data["rotation"][1:] + pose_data["rotation"][0:1],
                    dtype=np.float64,
                )
            ).as_dcm()
            T = np.concatenate(
                [T, np.array(pose_data["translation"])[:, np.newaxis]], axis=1
            )
            T = np.concatenate([T, np.zeros((1, 4))], axis=0)
            T[3, 3] = 1.0
            self.homogenuous_transformation_matrix = T
        elif data_format == "matrix":
            assert isinstance(pose_data, np.ndarray)
            assert pose_data.shape in [(3, 4), (4, 4)]
            if pose_data.shape == (3, 4):
                self.homogenuous_transformation_matrix = np.eye(4, dtype=np.float64)
                self.homogenuous_transformation_matrix[:3, :] = pose_data
            else:
                self.homogenuous_transformation_matrix = pose_data.copy()
        else:
            raise ValueError(data_format)

        assert self.homogenuous_transformation_matrix.dtype == np.float64

    def copy(self):
        t = Transform()
        t.set_htm(self.homogenuous_transformation_matrix)
        return t

    def set_htm(self, htm):
        assert isinstance(htm, np.ndarray)
        assert htm.dtype == np.float64
        self.homogenuous_transformation_matrix = htm.copy()
        return self

    def as_htm(self):
        return self.homogenuous_transformation_matrix.copy()

    def set_trans(self, trans):
        assert isinstance(trans, np.ndarray)
        assert trans.dtype == np.float64
        self.homogenuous_transformation_matrix[:3, 3] = trans

    def trans(self):
        return self.homogenuous_transformation_matrix[:3, 3]

    def rot(self):
        return R.from_dcm(self.homogenuous_transformation_matrix[:3, :3])

    def rot_mat(self):
        return self.homogenuous_transformation_matrix[:3, :3]

    def yaw(self):
        xaxis_2d = self.homogenuous_transformation_matrix[:2, 0]
        ori = np.arctan2(xaxis_2d[1], xaxis_2d[0])
        return ori

    def invert(self):
        self.homogenuous_transformation_matrix = np.linalg.inv(
            self.homogenuous_transformation_matrix
        )
        return self

    def apply(self, points, coord_dim: int = -1):
        if coord_dim == 0:
            points[:3, ...] = np.tensordot(
                self.homogenuous_transformation_matrix[:3],
                np.vstack([points[:3, ...], np.ones([1, *points.shape[1:]])]),
                axes=[[1], [0]],
            )
        elif coord_dim == -1:
            points[..., :3] = points[..., :3] @ self.rot_mat().T + self.trans()
        else:
            raise ValueError("unhandled coordinate dimension %d" % coord_dim)
        return points

    def __mul__(self, other_transform):
        return Transform().set_htm(
            self.homogenuous_transformation_matrix
            @ other_transform.homogenuous_transformation_matrix
        )
