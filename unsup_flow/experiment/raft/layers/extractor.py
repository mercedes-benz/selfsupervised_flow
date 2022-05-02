# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT

import tensorflow as tf
import tensorflow_addons as tfa


def Norm(norm_type):
    if norm_type == "instance":
        return tfa.layers.InstanceNormalization()
    elif norm_type is None:
        return tf.keras.layers.Lambda(lambda x: x)
    else:
        raise ValueError("Invalid norm_type specified: {0}".format(norm_type))


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, num_in_filters, num_out_filters, norm, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = num_out_filters
        self.strides = strides

        self.conv1 = tf.keras.layers.Conv2D(num_out_filters, 3, strides, "same")
        self.conv2 = tf.keras.layers.Conv2D(num_out_filters, 3, 1, "same")

        self.norm1 = Norm(norm)
        self.norm2 = Norm(norm)

        if strides == 1 and num_in_filters == num_out_filters:
            self.downsample = None
        else:
            self.downsample = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(num_out_filters, 1, strides),
                    Norm(norm),
                ]
            )

    def call(self, inputs, training):
        fx = inputs
        fx = tf.nn.relu(self.norm1(self.conv1(fx), training=training))
        fx = tf.nn.relu(self.norm2(self.conv2(fx), training=training))

        if self.downsample:
            inputs = self.downsample(inputs, training=training)

        return tf.nn.relu(inputs + fx)


class SmallEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        output_dim=128,
        norm=None,
        drop_rate=0.0,
        conv_strides=(1, 2, 2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm_str = norm
        self.drop_rate = drop_rate

        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=7, strides=2, padding="same"
        )
        self.norm1 = Norm(norm, groups=8)

        assert len(conv_strides) == 3
        self.layer1 = self._make_layer(
            num_in_filters=32, num_out_filters=32, conv_stride=conv_strides[0]
        )
        self.layer2 = self._make_layer(
            num_in_filters=32, num_out_filters=64, conv_stride=conv_strides[1]
        )
        self.layer3 = self._make_layer(
            num_in_filters=64, num_out_filters=96, conv_stride=conv_strides[2]
        )

        self.conv2 = tf.keras.layers.Conv2D(output_dim, 1)

        self.dropout = tf.keras.layers.Dropout(drop_rate) if drop_rate > 0 else None

    def _make_layer(self, num_in_filters, num_out_filters, conv_stride):
        seq = tf.keras.Sequential(
            [
                ResidualBlock(
                    num_in_filters, num_out_filters, self.norm_str, conv_stride
                ),
                ResidualBlock(num_in_filters, num_out_filters, self.norm_str, 1),
            ]
        )
        return seq

    def call(self, inputs, training):
        is_list = isinstance(inputs, (tuple, list))
        if is_list:
            inputs = tf.concat(inputs, axis=0)

        x = tf.nn.relu(self.norm1(self.conv1(inputs), training=training))
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.conv2(x)

        if self.dropout:
            x = self.dropout(x, training=training)

        if is_list:
            x = tf.split(x, num_or_size_splits=2, axis=0)

        return x
