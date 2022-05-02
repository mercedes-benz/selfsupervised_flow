#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import tensorflow as tf

from unsup_flow.tf.relaxed_tf_function import relaxed_tf_function


# @tf.function
@relaxed_tf_function(
    lambda x: [s if s in [2, 3, 4] and i > 0 else None for i, s in enumerate(x.shape)]
)
def tf_func(dict_of_tensors, step, **kwargs):
    print(
        "####################################### Tracing #########################################"
    )
    print(kwargs)
    print({k: id(v) for k, v in kwargs.items()})
    print(dict_of_tensors)
    print(step)
    print([v for k, v in kwargs.items() if len(k) > 1])
    return (
        dict_of_tensors["c"]
        + dict_of_tensors["ccc"]
        + step
        + sum(v for k, v in kwargs.items() if len(k) > 1)
    )


@relaxed_tf_function(
    lambda x: [s if s in [2, 3, 4] and i > 0 else None for i, s in enumerate(x.shape)]
)
def tf_func2(dict_of_tensors, step, **kwargs):
    print(
        "####################################### Tracing 2 #########################################"
    )
    print(kwargs)
    print({k: id(v) for k, v in kwargs.items()})
    print(dict_of_tensors)
    print(step)
    print([v for k, v in kwargs.items() if len(k) > 1])
    return dict_of_tensors["c"] + step + sum(v for k, v in kwargs.items() if len(k) > 1)


def test_relaxed_tf_function_tracing():
    c = tf.constant(0.0)
    cc = tf.constant([[0.0], [0.0]])
    dd = tf.constant([[1.0]])
    ccc = tf.constant([[[0.0], [0.0]], [[0.0], [0.0]]])
    r1 = tf_func({"c": c, "cc": (cc, ccc), "ccc": ccc}, 1, asdf=3, s="asdf")
    r2 = tf_func({"c": c, "cc": (dd, ccc), "ccc": ccc}, 1, asdf=3, s="asdf")
    r3 = tf_func({"c": c, "cc": (dd, ccc), "ccc": ccc}, 1, jkl=4, s="jkl")
    r4 = tf_func({"c": c, "cc": (cc, ccc), "ccc": ccc}, 4, jkl=4, s="jkl")
    print(r1, r2, r3, r4)
    r1 = tf_func2({"c": c, "cc": (cc, ccc), "ccc": ccc}, 1, asdf=3, s="asdf")
    r2 = tf_func2({"c": c, "cc": (dd, ccc), "ccc": ccc}, 1, asdf=3, s="asdf")
    r3 = tf_func2({"c": c, "cc": (dd, ccc), "ccc": ccc}, 1, jkl=4, s="jkl")
    r4 = tf_func2({"c": c, "cc": (cc, ccc), "ccc": ccc}, 4, jkl=4, s="jkl")
    print(r1, r2, r3, r4)
    r1 = tf_func({"c": c, "cc": (cc, ccc), "ccc": ccc}, 1, asdf=3, s="asdf")
    r2 = tf_func({"c": c, "cc": (dd, ccc), "ccc": ccc}, 1, asdf=3, s="asdf")
    r3 = tf_func({"c": c, "cc": (dd, ccc), "ccc": ccc}, 1, jkl=4, s="jkl")
    r4 = tf_func({"c": c, "cc": (cc, ccc), "ccc": ccc}, 4, jkl=4, s="jkl")
    print(r1, r2, r3, r4)
    # trace_a1 = tf_func.get_concrete_function(
    #     {"c": c, "cc": cc, "ccc": ccc}, c, asdf=3, s="asdf"
    # )
    # trace_a2 = tf_func.get_concrete_function(
    #     {"c": c, "cc": dd, "ccc": ccc}, c, asdf=3, s="asdf"
    # )
    # trace_b1 = tf_func.get_concrete_function(
    #     {"c": c, "cc": dd, "ccc": ccc}, c, jkl=3, s="jkl"
    # )
    # trace_b2 = tf_func.get_concrete_function(
    #     {"c": c, "cc": cc, "ccc": ccc}, c, jkl=3, s="jkl"
    # )
    # assert trace_a1 == trace_a2
    # assert trace_b1 == trace_b2
    # assert trace_a1 != trace_b1
