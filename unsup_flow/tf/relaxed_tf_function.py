#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import functools

import tensorflow as tf

TF_Tensors = (tf.Tensor, tf.Variable)


def all_relaxed(tensor):
    return [None] * len(tensor.shape)


def get_relaxed_shapes_from_nested_args(arg, shape_preservation_function=all_relaxed):
    if isinstance(arg, TF_Tensors):
        return tf.TensorSpec(shape_preservation_function(arg), arg.dtype)
    elif type(arg) is dict:
        assert all(type(k) is str for k in arg.keys())
        return {
            key: get_relaxed_shapes_from_nested_args(val, shape_preservation_function)
            for key, val in arg.items()
        }
    elif type(arg) is tuple:
        return tuple(
            get_relaxed_shapes_from_nested_args(val, shape_preservation_function)
            for val in arg
        )
    elif type(arg) is list:
        return [
            get_relaxed_shapes_from_nested_args(val, shape_preservation_function)
            for val in arg
        ]
    else:
        raise ValueError("unknown arg: {0}".format(arg))


def unpack_nested_tensors(nested_tensors):
    if isinstance(nested_tensors, TF_Tensors):
        return [nested_tensors]
    elif type(nested_tensors) is dict:
        assert all(type(k) is str for k in nested_tensors.keys())
        result = []
        for k in sorted(nested_tensors.keys()):
            result += unpack_nested_tensors(nested_tensors[k])
        return result
    elif type(nested_tensors) in (tuple, list):
        result = []
        for el in nested_tensors:
            result += unpack_nested_tensors(el)
        return result
    elif type(nested_tensors) is set:
        result = []
        for el in sorted(nested_tensors):
            result += unpack_nested_tensors(el)
    else:
        return []


class TemplateTensor:
    pass


def make_template(nested_tensors):
    if isinstance(nested_tensors, TF_Tensors):
        return TemplateTensor()
    elif type(nested_tensors) is dict:
        assert all(type(k) is str for k in nested_tensors.keys())
        return {k: make_template(v) for k, v in nested_tensors.items()}
    elif type(nested_tensors) is tuple:
        return tuple(make_template(v) for v in nested_tensors)
    elif type(nested_tensors) is list:
        return [make_template(v) for v in nested_tensors]
    elif type(nested_tensors) is set:
        return {make_template(v) for v in sorted(nested_tensors)}
    else:
        return nested_tensors


def get_list_nester_from_template(nested_tensors):
    template_nested_tensors = make_template(nested_tensors)

    def nest_list_of_tensors(list_tensors, template_nested_tensors):
        if type(template_nested_tensors) is TemplateTensor:
            assert len(list_tensors) >= 1
            return list_tensors[0], list_tensors[1:]
        elif type(template_nested_tensors) is dict:
            assert all(type(k) is str for k in template_nested_tensors.keys())
            result = {}
            for k in sorted(template_nested_tensors.keys()):
                r, list_tensors = nest_list_of_tensors(
                    list_tensors, template_nested_tensors[k]
                )
                result[k] = r
            return result, list_tensors
        elif type(template_nested_tensors) in (tuple, list):
            is_tuple = type(template_nested_tensors) is tuple
            result = []
            for el in template_nested_tensors:
                r, list_tensors = nest_list_of_tensors(list_tensors, el)
                result.append(r)
            if is_tuple:
                result = tuple(result)
            return result, list_tensors
        elif type(template_nested_tensors) is set:
            result = {}
            for el in sorted(template_nested_tensors):
                r, list_tensors = nest_list_of_tensors(list_tensors, el)
                result.add(r)
            return result, list_tensors
        else:
            return template_nested_tensors, list_tensors

    return functools.partial(
        nest_list_of_tensors, template_nested_tensors=template_nested_tensors
    )


def hashable_tuples_from_non_tensor_nested_struct(nested_struct):
    if isinstance(nested_struct, TF_Tensors):
        return ()
    if type(nested_struct) is dict:
        assert all(type(k) is str for k in nested_struct.keys())
        return tuple(
            (k, hashable_tuples_from_non_tensor_nested_struct(nested_struct[k]))
            for k in sorted(nested_struct.keys())
        )
    if type(nested_struct) in (list, tuple):
        return tuple(
            hashable_tuples_from_non_tensor_nested_struct(v) for v in nested_struct
        )
    if type(nested_struct) is set:
        return tuple(
            hashable_tuples_from_non_tensor_nested_struct(v)
            for v in sorted(nested_struct)
        )
    return id(nested_struct)


def hashable_tuples_from_non_tensor_args_kwargs(args, kwargs):
    return (
        hashable_tuples_from_non_tensor_nested_struct(args),
        hashable_tuples_from_non_tensor_nested_struct(kwargs),
    )


def relaxed_tf_function(shape_preservation_function=all_relaxed):
    """
    shape_preservation_function: check out the default function `all_relaxed` to understand what you want to put here

    Constraints:
        Your function that you want to wrap with tf.function can accept args and kwargs.
        All tf.Tensor and tf.Variable inputs will be detected if they are possibly nested
        but only inside tuples list or dicts (with keywords being strictly strings)!
    """

    def deco(func):
        def wrapped(*args, **kwargs):
            if not hasattr(wrapped, "cached_funcs"):
                wrapped.cached_funcs = {}
            hashable_tuples = hashable_tuples_from_non_tensor_args_kwargs(args, kwargs)
            if hashable_tuples not in wrapped.cached_funcs:
                nest_list_of_tensors = get_list_nester_from_template((args, kwargs))

                def func_with_list_args_tensors(*list_tensors):
                    inner_args, inner_kwargs = nest_list_of_tensors(list_tensors)[0]
                    return func(*inner_args, **inner_kwargs)

                list_tensor_shapes = get_relaxed_shapes_from_nested_args(
                    unpack_nested_tensors((args, kwargs)), shape_preservation_function
                )
                wrapped.cached_funcs[
                    hashable_tuples_from_non_tensor_args_kwargs(args, kwargs)
                ] = tf.function(
                    func_with_list_args_tensors, input_signature=list_tensor_shapes
                )
            return wrapped.cached_funcs[hashable_tuples](
                *unpack_nested_tensors((args, kwargs))
            )

        return wrapped

    return deco


if __name__ == "__main__":
    c = tf.constant(0.0)
    list_tensors = [c, c, c, c, c, c]
    nested_tensors = ({"tupe": (c, c, "inside"), "list": [(c, c, 3), c]}, c, "asd")
    print(unpack_nested_tensors(nested_tensors))
    print(*[id(v) for v in unpack_nested_tensors(nested_tensors)])
    nest_list_of_tensors = get_list_nester_from_template(nested_tensors)
    print(nest_list_of_tensors(list_tensors)[0])
