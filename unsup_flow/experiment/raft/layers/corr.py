# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT

import tensorflow as tf

from unsup_flow.tf import shape


def initialize_flow(image, downscale_factor=8):
    bs, h, w, _ = shape(image)
    target_height = h // downscale_factor
    target_width = w // downscale_factor
    gy, gx = tf.meshgrid(
        tf.range(target_width, dtype=tf.float32),
        tf.range(target_height, dtype=tf.float32),
        indexing="ij",
    )
    return tf.tile(tf.stack([gx, gy], axis=-1)[None, ...], (bs, 1, 1, 1))


# @tf.function
def bilinear_img_sampling(img, coords):
    """
    upper_left   upper_right
     ___________
    |           |
    |        +  |
    |           |
    |           |
    |___________|
    lower_left   lower_right
    """
    _, num_img_rows, num_img_cols, _ = tf.unstack(tf.shape(img))

    target_col_idx_f32 = coords[:, :, :, 0]
    target_row_idx_f32 = coords[:, :, :, 1]
    left_col = tf.cast(target_col_idx_f32, tf.int32)
    right_col = left_col + 1
    lower_row = tf.cast(target_row_idx_f32, tf.int32)
    upper_row = lower_row + 1

    left_col = tf.clip_by_value(left_col, 0, num_img_cols - 1)
    right_col = tf.clip_by_value(right_col, 0, num_img_cols - 1)
    lower_row = tf.clip_by_value(lower_row, 0, num_img_rows - 1)
    upper_row = tf.clip_by_value(upper_row, 0, num_img_rows - 1)

    lower_left_val = tf.gather_nd(
        img, tf.stack([lower_row, left_col], axis=-1), batch_dims=1
    )
    upper_left_val = tf.gather_nd(
        img, tf.stack([upper_row, left_col], axis=-1), batch_dims=1
    )
    lower_right_val = tf.gather_nd(
        img, tf.stack([lower_row, right_col], axis=-1), batch_dims=1
    )
    upper_right_val = tf.gather_nd(
        img, tf.stack([upper_row, right_col], axis=-1), batch_dims=1
    )

    col_interpol = tf.cast(right_col, tf.float32) - target_col_idx_f32
    row_interpol = tf.cast(upper_row, tf.float32) - target_row_idx_f32
    weight_lower_left = col_interpol * row_interpol
    weight_upper_left = col_interpol * (1.0 - row_interpol)
    weight_lower_right = (1.0 - col_interpol) * row_interpol
    weight_upper_right = (1.0 - col_interpol) * (1.0 - row_interpol)

    return tf.add_n(
        [
            weight_lower_left[..., None] * lower_left_val,
            weight_upper_left[..., None] * upper_left_val,
            weight_lower_right[..., None] * lower_right_val,
            weight_upper_right[..., None] * upper_right_val,
        ]
    )


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        corr = self.correlation(fmap1, fmap2)
        batch_size, h1, w1, _, h2, w2 = tf.unstack(tf.shape(corr))
        corr = tf.reshape(corr, (batch_size * h1 * w1, h2, w2, 1))

        self.corr_pyramid = [corr]
        for _ in range(num_levels - 1):
            corr = tf.nn.avg_pool2d(corr, ksize=2, strides=2, padding="VALID")
            self.corr_pyramid.append(corr)

    def retrieve(self, _coords_t0, coords_t1):
        r = self.radius
        batch_size, num_rows, num_cols, _ = tf.unstack(tf.shape(coords_t1))

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            d = tf.range(-r, r + 1, dtype=tf.float32)
            delta_rows, delta_cols = tf.meshgrid(d, d, indexing="ij")
            delta = tf.stack([delta_rows, delta_cols], axis=-1)

            centroid_lvl = (
                tf.reshape(coords_t1, (batch_size * num_rows * num_cols, 1, 1, 2))
                / 2**i
            )
            delta_lvl = tf.reshape(delta, (1, 2 * r + 1, 2 * r + 1, 2))
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_img_sampling(corr, coords_lvl)
            corr = tf.reshape(
                corr, (batch_size, num_rows, num_cols, (2 * r + 1) * (2 * r + 1))
            )
            out_pyramid.append(corr)
        out = tf.concat(out_pyramid, axis=-1)
        return out

    def correlation(self, fmap1, fmap2):
        batch_size, num_rows, num_cols, num_channels = tf.unstack(tf.shape(fmap1))
        fmap1 = tf.reshape(fmap1, (batch_size, num_rows * num_cols, num_channels))
        fmap2 = tf.reshape(fmap2, (batch_size, num_rows * num_cols, num_channels))
        corr = tf.matmul(fmap1, fmap2, transpose_b=True)
        corr = tf.reshape(corr, (batch_size, num_rows, num_cols, 1, num_rows, num_cols))
        return corr / tf.sqrt(tf.cast(num_channels, dtype=tf.float32))


def main():
    img = tf.random.uniform(shape=[2, 40, 70, 3])
    coords = (tf.random.uniform([2, 40, 70, 2]) - 0.5) * 30
    interpol = bilinear_img_sampling(img, coords)
    print(interpol.shape)
    return 0


if __name__ == "__main__":
    main()
