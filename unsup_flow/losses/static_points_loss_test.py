#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import numpy as np
import tensorflow as tf

from unsup_flow.losses.static_points_loss import padded2ragged


def test_padded2ragged():
    padded_tensor = tf.constant(
        np.array(
            [
                [
                    [[[np.nan, np.nan]], [[np.nan, np.nan]]],
                    [[[1.0, 2.0]], [[3.0, 4.0]]],
                ],
                [
                    [[[3.0, 4.0]], [[5.0, 6.0]]],
                    [[[np.nan, np.nan]], [[np.nan, np.nan]]],
                ],
            ]
        )
    )
    ragged_tensor = padded2ragged(padded_tensor)
    assert ragged_tensor.shape.as_list() == [2, None, 2, 1, 2]
