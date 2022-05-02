# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT

import tensorflow as tf


class FlowOrClassificationHead(tf.keras.layers.Layer):
    def __init__(self, filters=256, out_dims=2, **kwargs):
        super().__init__(**kwargs)
        assert out_dims in [
            2,
            3,
            4,
        ], "choose out_dims=2 for flow or out_dims=4 for classification or 3 if the paper DL is dangerously close"
        self.filters = filters

        self.conv1 = tf.keras.layers.Conv2D(filters, 3, 1, "same")
        self.conv2 = tf.keras.layers.Conv2D(out_dims, 3, 1, "same")

    def call(self, inputs):
        return self.conv2(tf.nn.relu(self.conv1(inputs)))


class ConvGRU(tf.keras.layers.Layer):
    def __init__(self, filters=128, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        self.convz = tf.keras.layers.Conv2D(filters, 3, 1, "same")
        self.convr = tf.keras.layers.Conv2D(filters, 3, 1, "same")
        self.convq = tf.keras.layers.Conv2D(filters, 3, 1, "same")

    def call(self, h, x):
        hx = tf.concat([h, x], axis=-1)

        z = tf.nn.sigmoid(self.convz(hx))
        r = tf.nn.sigmoid(self.convr(hx))
        q = tf.nn.tanh(self.convq(tf.concat([r * h, x], axis=-1)))

        h = (1 - z) * h + z * q
        return h


class SmallMotionEncoder(tf.keras.layers.Layer):
    def __init__(self, predict_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.conv_stat_corr1 = tf.keras.layers.Conv2D(96, 1, 1, "same")

        self.conv_flow1 = tf.keras.layers.Conv2D(64, 7, 1, "same")
        self.conv_flow2 = tf.keras.layers.Conv2D(32, 3, 1, "same")
        self.predict_logits = predict_logits
        if self.predict_logits:
            self.conv_class1 = tf.keras.layers.Conv2D(64, 7, 1, "same")
            self.conv_class2 = tf.keras.layers.Conv2D(32, 3, 1, "same")
        self.conv = tf.keras.layers.Conv2D(80, 3, 1, "same")

    def call(self, flow, corr, logits):
        corr = tf.nn.relu(self.conv_stat_corr1(corr))

        flow = tf.nn.relu(self.conv_flow1(flow))
        flow = tf.nn.relu(self.conv_flow2(flow))

        concat_vals = [corr, flow]
        if self.predict_logits:
            logits = tf.nn.relu(self.conv_class1(logits))
            logits = tf.nn.relu(self.conv_class2(logits))
            concat_vals.append(logits)
        else:
            assert logits is None

        cor_flo_logits = tf.concat(concat_vals, axis=-1)
        out = tf.nn.relu(self.conv(cor_flo_logits))

        if self.predict_logits:
            return tf.concat([out, logits, flow], axis=-1)
        else:
            return tf.concat([out, flow], axis=-1)


class SmallUpdateBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters=96,
        learn_upsampling_mask=True,
        use_seperate_flow_maps_stat_dyn=True,
        feature_downsampling_factor=8,
        predict_weight_for_static_aggregation=False,
        predict_logits=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.predict_logits = predict_logits

        self.motion_encoder = SmallMotionEncoder(predict_logits=predict_logits)
        self.gru = ConvGRU(filters)
        self.predict_weight_for_static_aggregation = (
            predict_weight_for_static_aggregation is not False
        )
        num_stat_flow_head_channels = (
            3 if predict_weight_for_static_aggregation is not False else 2
        )
        self.static_flow_head = FlowOrClassificationHead(
            128, num_stat_flow_head_channels
        )

        if self.predict_logits:
            self.classification_head = FlowOrClassificationHead(128, 4)
        self.learn_upsampling_mask = learn_upsampling_mask
        if self.learn_upsampling_mask:
            self.upsampling_mask_head = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(256, 3, 1, "same"),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2D(feature_downsampling_factor ** 2 * 9, 1, 1),
                ]
            )
        self.use_seperate_flow_maps_stat_dyn = use_seperate_flow_maps_stat_dyn

    def call(self, net, inp, corr, flow, logits, weight_logits_for_static_aggregation):
        if self.predict_weight_for_static_aggregation:
            motion_features = self.motion_encoder(
                tf.concat([flow, weight_logits_for_static_aggregation], axis=-1),
                corr,
                logits,
            )
        else:
            assert weight_logits_for_static_aggregation is None
            motion_features = self.motion_encoder(flow, corr, logits)

        inp = tf.concat([inp, motion_features], axis=-1)
        net = self.gru(net, inp)

        if self.predict_weight_for_static_aggregation:
            delta = self.static_flow_head(net)
            delta_static_flow = delta[..., 0:2]
            delta_weights = delta[..., -1][..., None]
        else:
            delta_static_flow = self.static_flow_head(net)
            delta_weights = None

        if self.predict_logits:
            delta_logits = self.classification_head(net)
        else:
            delta_logits = None

        if self.learn_upsampling_mask:
            raise NotImplementedError()
        else:
            mask = None

        return (
            net,
            delta_static_flow,
            delta_logits,
            mask,
            delta_weights,
        )
