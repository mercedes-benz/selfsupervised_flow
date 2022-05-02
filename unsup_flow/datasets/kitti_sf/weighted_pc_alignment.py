#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import numpy as np


def weighted_pc_alignment(cloud_t0, cloud_t1, weights):
    m, n = cloud_t0.shape
    # ones = np.ones(n)
    # X_homog = np.vstack((cloud_t0, ones))
    # Y_homog = np.vstack((cloud_t1, ones))

    # weights = weights[:, None]

    cum_wts = np.sum(weights)

    X_wtd = cloud_t0 * weights
    Y_wtd = cloud_t1 * weights

    mx_wtd = X_wtd.sum(1) / cum_wts
    my_wtd = Y_wtd.sum(1) / cum_wts
    Xc = cloud_t0 - np.tile(mx_wtd, (n, 1)).T
    Yc = cloud_t1 - np.tile(my_wtd, (n, 1)).T

    # sx = np.mean(np.sum(Xc * Yc, 0))

    Sxy_wtd = np.dot(Yc * weights, Xc.T) / cum_wts

    U, D, V = np.linalg.svd(Sxy_wtd, full_matrices=True, compute_uv=True)
    V = V.T.copy()
    # print U,"\n\n",D,"\n\n",V
    # r = np.rank(Sxy)
    r = np.linalg.matrix_rank(Sxy_wtd)
    # d = np.linalg.det(Sxy_wtd)
    S = np.eye(m)

    if r > (m - 1):
        if np.linalg.det(Sxy_wtd) < 0.0:
            S[m - 1, m - 1] = -1.0
    elif r == m - 1:
        det_mul = np.linalg.det(U) * np.linalg.det(V)
        if np.isclose(det_mul, -1.0):
            S[m - 1, m - 1] = -1.0
    else:
        raise RuntimeError("Rank deterioration!")

    R = np.dot(np.dot(U, S), V.T)

    # c = np.trace(np.dot(np.diag(D), S)) / sx
    c = 1.0
    t = my_wtd - c * np.dot(R, mx_wtd)

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return R, c, t


def weighted_pc_alignment_wrapper_homog_trafo(cloud_t0, cloud_t1, weights):
    R, c, t = weighted_pc_alignment(cloud_t0, cloud_t1, weights)
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T


if __name__ == "__main__":
    # Run an example test
    # We have 3 points in 3D. Every point is a column vector of this matrix A
    A = np.array(
        [
            [0.57215, 0.37512, 0.37551, 1.57215, 1.37512, 1.37551],
            [0.23318, 0.86846, 0.98642, 1.23318, 1.86846, 1.98642],
            [0.79969, 0.96778, 0.27493, 1.79969, 1.96778, 1.27493],
        ]
    )
    # Deep copy A to get B
    B = A.copy()
    # and sum a translation on z axis (3rd row) of 10 units
    B[2, :3] = B[2, :3] + 10
    B[2, 3:] = B[2, 3:] + 11
    weights = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    # Reconstruct the transformation with ralign.ralign
    R, c, t = weighted_pc_alignment(A, B, weights)
    print(
        "Rotation matrix=\n", R, "\nScaling coefficient=", c, "\nTranslation vector=", t
    )
