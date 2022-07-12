# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT

import tensorflow as tf

from unsup_flow.blocks.point_pillars import PointPillarsLayer
from unsup_flow.experiment.decoder import HeadDecoder
from unsup_flow.experiment.losses import (
    SupervisedLoss,
    UnsupervisedLoss,
    plot_some_supervised_metrics,
)
from unsup_flow.experiment.movavg_cls_threshold import MovingAverageThreshold
from unsup_flow.experiment.raft.layers.corr import CorrBlock, initialize_flow
from unsup_flow.experiment.raft.layers.extractor import SmallEncoder
from unsup_flow.experiment.raft.layers.update import SmallUpdateBlock
from unsup_flow.experiment.raft.layers.upsampling import upflow_n, uplogits_n
from unsup_flow.tf import shape


class OurPillarModel(tf.keras.Model):  # type:ignore
    def __init__(self, cfg, label_dict, *args, **kwargs):
        assert tf.__version__[0] == "2"
        super().__init__(*args, autocast=False, **kwargs)
        self.cfg = cfg

        used_modes = {p.mode for p in self.cfg.phases.values()}
        assert used_modes.issubset({"supervised", "unsupervised"})

        self.head_decoder_fw = HeadDecoder(self.cfg, name="head_decoder_forward")
        self.head_decoder_bw = HeadDecoder(self.cfg, name="head_decoder_backward")
        num_still_points = self.cfg.data.train.num_still_points
        if num_still_points is not None:
            num_still_points *= self.cfg.model.num_iters
        self.moving_dynamicness_threshold = MovingAverageThreshold(
            unsupervised="unsupervised" in used_modes,
            num_train_samples=self.cfg.data.nbr_samples.kitti_lidar_raw
            if self.cfg.data.name == "kitti_lidar_raw"
            else self.cfg.data.nbr_samples.train,
            num_moving=self.cfg.data.train.num_moving_points * self.cfg.model.num_iters,
            num_still=num_still_points,
        )
        self.network = RAFT(
            cfg=self.cfg,
            head_decoder_fw=self.head_decoder_fw,
            head_decoder_bw=self.head_decoder_bw,
        )
        self.loss_layers = {}
        self.label_dict = label_dict
        loss_cfg = self.cfg.losses
        if "supervised" in used_modes:
            assert self.cfg.model.dynamic_flow_is_non_rigid_flow == (
                "norig_and_ego" in loss_cfg.supervised.mode
            )
            self.loss_layers["supervised"] = SupervisedLoss(
                cfg=loss_cfg.supervised,
                final_scale=self.cfg.model.u_net.final_scale,
                initial_grid_size=self.cfg.model.point_pillars.nbr_pillars,
            )
        if "unsupervised" in used_modes:
            self.loss_layers["unsupervised"] = UnsupervisedLoss(
                cfg=loss_cfg.unsupervised,
                model_cfg=self.cfg.model,
                bev_extent=self.cfg.data.bev_extent,
                final_scale=self.cfg.model.u_net.final_scale,
            )

    def call(self, el, summaries, compute_losses, training):
        # predictions is list across iterations
        outputs_fw, outputs_bw = self.network(el, summaries, training)

        predictions_fw = []
        predictions_bw = []
        losses = []

        for it, (net_output_0_1, net_output_1_0) in enumerate(
            zip(outputs_fw, outputs_bw)
        ):
            should_write_img_summaries = summaries["imgs_eval"] and (
                it == len(outputs_fw) - 1
            )
            should_write_metric_summaries = summaries["metrics_eval"] and (
                it == len(outputs_fw) - 1
            )
            cur_summaries = {
                "writer": summaries["writer"],
                "imgs_eval": should_write_img_summaries,
                "metrics_eval": should_write_metric_summaries,
                "metrics_label_dict": summaries["metrics_label_dict"],
                "label_mapping": summaries["label_mapping"],
            }

            prediction_fw = self.head_decoder_fw(
                net_output_0_1,
                dynamicness_threshold=self.moving_dynamicness_threshold.value(),
                pointwise_valid_mask=el["pcl_t0"]["pointwise_valid_mask"],
                pointwise_voxel_coordinates=el["pcl_t0"]["pointwise_voxel_coors"],
                pc=el["pcl_t0"]["pc"],
                filled_pillar_mask=el["pcl_t0"]["bev_pillar_fs_mask"],
                odom=el["odom_t0_t1"],
                inv_odom=tf.linalg.inv(el["odom_t0_t1"]),
                summaries=cur_summaries,
                gt_flow_bev=el["pcl_t0"].get("flow_map_bev_gt", None),
                ohe_gt_stat_dyn_ground_label_bev_map=el.get(
                    "ohe_gt_stat_dyn_ground_label_bev_map_t0", None
                ),
                dynamic_flow_is_non_rigid_flow=self.cfg.model.dynamic_flow_is_non_rigid_flow,
            )
            prediction_bw = self.head_decoder_bw(
                net_output_1_0,
                dynamicness_threshold=self.moving_dynamicness_threshold.value(),
                pointwise_valid_mask=el["pcl_t1"]["pointwise_valid_mask"],
                pointwise_voxel_coordinates=el["pcl_t1"]["pointwise_voxel_coors"],
                pc=el["pcl_t1"]["pc"],
                filled_pillar_mask=el["pcl_t1"]["bev_pillar_fs_mask"],
                odom=tf.linalg.inv(el["odom_t0_t1"]),
                inv_odom=el["odom_t0_t1"],
                gt_flow_bev=el["pcl_t1"].get("flow_map_bev_gt", None),
                summaries=cur_summaries,
                ohe_gt_stat_dyn_ground_label_bev_map=el.get(
                    "ohe_gt_stat_dyn_ground_label_bev_map_t1", None
                ),
                dynamic_flow_is_non_rigid_flow=self.cfg.model.dynamic_flow_is_non_rigid_flow,
            )

            predictions_fw.append(prediction_fw)
            predictions_bw.append(prediction_bw)
            num_iters = (
                self.cfg.model.num_iters if training else self.cfg.model.num_pred_iters
            )
            weight = 0.8 ** (num_iters - 1 - it)
            if compute_losses:
                loss = {
                    k: weight
                    * self.loss_layers[k](
                        el,
                        prediction_fw,
                        prediction_bw,
                        self.moving_dynamicness_threshold,
                        cur_summaries,
                        training,
                    )
                    for k in self.loss_layers
                }
                losses.append(loss)

        self.prediction_fw = predictions_fw[-1]
        self.prediction_bw = predictions_bw[-1]

        if summaries["metrics_eval"]:
            plot_some_supervised_metrics(
                el, self.prediction_fw, self.label_dict, summaries, "fw"
            )
            plot_some_supervised_metrics(
                el, self.prediction_bw, self.label_dict, summaries, "bw"
            )

        self.loss = {}
        if compute_losses:
            for key in self.loss_layers:
                self.loss[key] = tf.add_n([level[key] for level in losses])
        return self.loss


class RAFT(tf.keras.Model):
    def __init__(self, cfg, head_decoder_fw, head_decoder_bw, **kwargs):
        super().__init__(autocast=False, **kwargs)
        self.head_decoder_fw = head_decoder_fw
        self.head_decoder_bw = head_decoder_bw

        self.cfg = cfg

        self.drop_rate = cfg.model.dropout_rate
        self.bev_rows_res_meters_per_fs_pixel = (
            (self.cfg.data.bev_extent[2] - self.cfg.data.bev_extent[0])
            / self.cfg.model.point_pillars.nbr_pillars[0]
            * self.cfg.model.u_net.final_scale
        )

        self.bev_cols_res_meters_per_fs_pixel = (
            (self.cfg.data.bev_extent[3] - self.cfg.data.bev_extent[1])
            / self.cfg.model.point_pillars.nbr_pillars[1]
            * self.cfg.model.u_net.final_scale
        )
        assert (
            self.bev_rows_res_meters_per_fs_pixel
            == self.bev_cols_res_meters_per_fs_pixel
        ), "resolutions are different, but I am unsure if they are applied correctly in this case cause of interpretion switch in raft"

        self.pp_layer = PointPillarsLayer(
            self.cfg.model.point_pillars,
            bn_kwargs=dict(self.cfg.layers.batch_normalization),
        )

        self.hidden_dim = hdim = 96
        self.context_dim = cdim = 64
        feat_for_corr_dim = 128
        assert (
            self.cfg.model.feature_downsampling_factor == 8
        ), "you cannot use default CorrBlock without default resolution"
        self.flow_maps_archi = self.cfg.model.flow_maps_archi

        conv_strides = (1, 2, 2)

        self.fnet = SmallEncoder(
            output_dim=feat_for_corr_dim,
            norm="instance",
            drop_rate=self.drop_rate,
            conv_strides=conv_strides,
        )
        self.cnet = SmallEncoder(
            output_dim=hdim + cdim,
            norm=None,
            drop_rate=self.drop_rate,
            conv_strides=conv_strides,
        )
        assert self.flow_maps_archi in [
            "single",
            "vanilla",
        ], "please use flow_maps_archi single or vanilla"
        self.update_block = SmallUpdateBlock(
            filters=hdim,
            learn_upsampling_mask=self.cfg.model.learn_upsampling,
            use_seperate_flow_maps_stat_dyn=False,
            feature_downsampling_factor=self.cfg.model.feature_downsampling_factor,
            predict_weight_for_static_aggregation=self.cfg.model.predict_weight_for_static_aggregation,
            predict_logits=self.flow_maps_archi != "vanilla",
        )

    def call(
        self,
        el,
        summaries,
        training,
    ):

        v_feats_0 = el["pcl_t0"]["voxel_feats"]
        p_feats_0 = el["pcl_t0"]["point_feats"]
        v_coors_0 = el["pcl_t0"]["voxel_coors"]
        v_count_0 = el["pcl_t0"]["voxel_count"]
        v_feats_1 = el["pcl_t1"]["voxel_feats"]
        p_feats_1 = el["pcl_t1"]["point_feats"]
        v_coors_1 = el["pcl_t1"]["voxel_coors"]
        v_count_1 = el["pcl_t1"]["voxel_count"]
        nbr_pcls = el["batch_size"]
        training = training

        self.bev_img_t0 = self.pp_layer(  # bev_enc_0
            [
                v_feats_0,
                p_feats_0["row_splits"],
                p_feats_0["values"],
                v_coors_0,
                v_count_0,
                nbr_pcls,
            ],
            training=training,
        )

        self.bev_img_t1 = self.pp_layer(  # bev_enc_1
            [
                v_feats_1,
                p_feats_1["row_splits"],
                p_feats_1["values"],
                v_coors_1,
                v_count_1,
                nbr_pcls,
            ],
            training=training,
        )

        self.fmap_t0 = self.fnet(self.bev_img_t0, training=training)
        self.fmap_t1 = self.fnet(self.bev_img_t1, training=training)

        assert self.flow_maps_archi in [
            "single",
            "vanilla",
        ], self.flow_maps_archi
        retvals_fw = self.predict_single_flow_map_and_classes(
            self.bev_img_t0,
            self.fmap_t0,
            self.fmap_t1,
            self.head_decoder_fw,
            training,
        )
        retvals_bw = self.predict_single_flow_map_and_classes(
            self.bev_img_t1,
            self.fmap_t1,
            self.fmap_t0,
            self.head_decoder_bw,
            training,
        )

        return retvals_fw, retvals_bw

    def predict_single_flow_map_and_classes(
        self, img_t0, fmap_t0, fmap_t1, decoder, training
    ):
        # coordiantes have behavior [:, w, h, :] = [h, w] where img_t0 shape is [B, H, W, C]
        pixel_coords_t0 = initialize_flow(
            img_t0, downscale_factor=self.cfg.model.feature_downsampling_factor
        )
        pixel_coords_t1 = pixel_coords_t0

        b, h, w, _ = shape(pixel_coords_t0)
        if self.cfg.model.flow_maps_archi == "vanilla":
            logits = None
            upsampled_dummy_logits = tf.zeros(
                (
                    b,
                    h * self.cfg.model.feature_downsampling_factor,
                    w * self.cfg.model.feature_downsampling_factor,
                    4,
                ),
                dtype=tf.float32,
            )
        else:
            logits = tf.zeros((b, h, w, 4), dtype=tf.float32)
        if self.cfg.model.predict_weight_for_static_aggregation is not False:
            assert self.cfg.model.flow_maps_archi != "vanilla"
            weight_logits_for_static_aggregation = tf.zeros(
                (b, h, w, 1), dtype=tf.float32
            )
        else:
            weight_logits_for_static_aggregation = None

        # setup correlation values
        assert self.cfg.model.corr_cfg.module == "all", self.cfg.model.corr_cfg.module
        correlation = CorrBlock(
            fmap_t0,
            fmap_t1,
            num_levels=self.cfg.model.corr_cfg.num_levels,
            radius=self.cfg.model.corr_cfg.search_radius,
        )

        # context network
        cnet = self.cnet(img_t0, training=training)
        net, inp = tf.split(cnet, [self.hidden_dim, self.context_dim], axis=-1)
        net = tf.tanh(net)
        inp = tf.nn.relu(inp)

        iters = self.cfg.model.num_iters if training else self.cfg.model.num_pred_iters
        intermediate_flow_predictions = []
        for _i in range(iters):
            pixel_coords_t1 = tf.stop_gradient(pixel_coords_t1)
            if self.cfg.model.flow_maps_archi != "vanilla":
                logits = tf.stop_gradient(logits)
            if self.cfg.model.predict_weight_for_static_aggregation is not False:
                weight_logits_for_static_aggregation = tf.stop_gradient(
                    weight_logits_for_static_aggregation
                )
            corr = correlation.retrieve(None, pixel_coords_t1)

            flow = pixel_coords_t1 - pixel_coords_t0
            (
                net,
                delta_flow,
                delta_logits,
                mask,
                delta_weight_logits_for_static_aggr,
            ) = self.update_block(
                net, inp, corr, flow, logits, weight_logits_for_static_aggregation
            )

            pixel_coords_t1 += delta_flow
            flow += delta_flow
            assert mask is None
            if self.cfg.model.flow_maps_archi != "vanilla":
                logits += delta_logits
            if self.cfg.model.predict_weight_for_static_aggregation is not False:
                weight_logits_for_static_aggregation += (
                    delta_weight_logits_for_static_aggr
                )

            upsampled_flow = upflow_n(
                flow,
                n=self.cfg.model.feature_downsampling_factor,
            )
            upsampled_flow_usfl_convention = self.change_flow_convention_from_raft2usfl(
                upsampled_flow
            )
            intermediate_flow_predictions.append(
                decoder.concat2network_output(
                    upsampled_dummy_logits
                    if self.cfg.model.flow_maps_archi == "vanilla"
                    else uplogits_n(
                        logits,
                        n=self.cfg.model.feature_downsampling_factor,
                    ),
                    upsampled_flow_usfl_convention,
                    upsampled_flow_usfl_convention,
                    None
                    if weight_logits_for_static_aggregation is None
                    else uplogits_n(
                        weight_logits_for_static_aggregation,
                        n=self.cfg.model.feature_downsampling_factor,
                    ),
                )
            )

        return intermediate_flow_predictions

    def change_flow_convention_from_raft2usfl(self, flow):
        resolution_adapter = tf.constant(
            [
                self.bev_rows_res_meters_per_fs_pixel,  # x-resolution of bev map
                self.bev_cols_res_meters_per_fs_pixel,  # y-resolution of bev map
            ],
            dtype=tf.float32,
        )[None, None, None, ...]
        flow_meters = tf.reverse(flow, axis=[-1]) * resolution_adapter
        return flow_meters
