#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import tensorflow as tf

from unsup_flow.tf import shapes_broadcastable


def numerically_stable_quotient_lin_comb_exps(
    *, num_exps, num_weights, denom_exps, denom_weights
):
    # evaluates in numerically stable form expression of
    # (sum_i num_weight_i * exp(num_exp_i))  /  (sum_i denom_weight_i * exp(denom_exp_i))
    max_denom_exp = denom_exps[0]
    for i in range(1, len(denom_exps)):
        max_denom_exp = tf.maximum(max_denom_exp, denom_exps[i])
    # divide numerator and denominator by exp(max_denom_exp)
    # by subtracting in every exponent
    for i in range(len(num_exps)):
        num_exps[i] = num_exps[i] - max_denom_exp
    for i in range(len(denom_exps)):
        denom_exps[i] = denom_exps[i] - max_denom_exp
    # now max denom exp is 0 and therefore denominator is well-behaved
    # in the case where all num exps are contained in the denom exps we also have a well-behaved numerator
    # and everything is fine (quite common case, e.g. exp(a)/(exp(a)+exp(b))))
    num_elems = [w * tf.exp(e) for w, e in zip(num_weights, num_exps)]
    denom_elems = [w * tf.exp(e) for w, e in zip(denom_weights, denom_exps)]
    return tf.add_n(num_elems) / tf.add_n(denom_elems)


def numerically_stable_quotient_lin_comb_exps_across_axis(
    *, num_exps, num_weights, denom_exps, denom_weights, axis=-1
):
    assert (
        len(num_exps.shape)
        == len(num_weights.shape)
        == len(denom_exps.shape)
        == len(denom_weights.shape)
    )
    num_shape = num_exps.shape.as_list()
    assert num_shape == num_weights.shape.as_list()
    num_shape[axis] = None
    denom_shape = denom_exps.shape.as_list()
    assert denom_shape == denom_weights.shape.as_list()
    denom_shape[axis] = None
    assert shapes_broadcastable(num_shape, denom_shape)

    # evaluates in numerically stable form expression of
    # (sum_i num_weight_i * exp(num_exp_i))  /  (sum_i denom_weight_i * exp(denom_exp_i))
    weight_masked_denom_exps = tf.where(
        denom_weights != 0.0,
        denom_exps,
        tf.reduce_min(denom_exps, axis=axis, keepdims=True),
    )
    max_denom_exp = tf.reduce_max(weight_masked_denom_exps, axis=axis, keepdims=True)
    # divide numerator and denominator by exp(max_denom_exp)
    # by subtracting in every exponent
    num_exps = num_exps - max_denom_exp
    denom_exps = denom_exps - max_denom_exp
    # now max denom exp is 0 and therefore denominator is well-behaved
    # in the case where all num exps are contained in the denom exps we also have a well-behaved numerator
    # and everything is fine (quite common case, e.g. exp(a)/(exp(a)+exp(b))))
    num = num_weights * tf.exp(num_exps)
    denom = denom_weights * tf.exp(denom_exps)
    return tf.reduce_sum(num, axis=axis) / tf.reduce_sum(denom, axis=axis)


def normalized_sigmoid_sum(logits, mask=None):
    # sigmoid(x) = exp(-relu(-x)) * sigmoid(abs(x))
    neg_logit_part = -tf.nn.relu(-logits)
    weights = tf.nn.sigmoid(tf.abs(logits))
    if mask is not None:
        neg_logit_part = tf.where(mask, neg_logit_part, tf.zeros_like(neg_logit_part))
        weights = tf.where(mask, weights, tf.zeros_like(weights))
    return numerically_stable_quotient_lin_comb_exps_across_axis(
        num_exps=neg_logit_part[..., :, None],
        num_weights=weights[..., :, None],
        denom_exps=neg_logit_part[..., None, :],
        denom_weights=weights[..., None, :],
    )


if __name__ == "__main__":
    logits = tf.random.normal((3, 2, 5)) + 1000
    sigm = tf.nn.sigmoid(logits)
    print(sigm)
    result = sigm / tf.reduce_sum(sigm, axis=-1, keepdims=True)
    print(result)
    print(normalized_sigmoid_sum(logits))
