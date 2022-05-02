#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import tensorflow as tf


def step_decay(global_step, step_length=60_000, decay_ratio=0.5, name="step_decay"):
    with tf.name_scope(name):
        step_length = tf.constant(step_length, tf.float32)
        decay_ratio = tf.constant(decay_ratio, tf.float32)
        global_step_flt = tf.cast(global_step, tf.float32)
        theta = tf.pow(decay_ratio, tf.floor(global_step_flt / step_length))
        if tf.__version__[0] == "2":
            tf.summary.scalar("theta", theta, step=global_step)
        else:
            tf.summary.scalar("theta", theta)
        return theta


def warm_up(global_step, step_length=2_000, initial=1e-5, name="warm_up"):
    with tf.name_scope(name):
        step_length = tf.constant(step_length, tf.float32)
        initial = tf.constant(initial, tf.float32)
        global_step_flt = tf.cast(global_step, tf.float32)
        theta = tf.where(
            global_step_flt < step_length,
            initial / tf.pow(initial, global_step_flt / step_length),
            1.0,
        )
        if tf.__version__[0] == "2":
            tf.summary.scalar("theta", theta, step=global_step)
        else:
            tf.summary.scalar("theta", theta)
        return theta
