#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


from typing import Dict

import numpy as np
import tensorflow as tf
from flatten_dict import flatten, unflatten


class terminalcolors:
    WARNING = "\033[93m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


tcolor = terminalcolors


def get_output_shapes(dataset: tf.data.Dataset) -> Dict:
    try:
        return dict(dataset.output_shapes)
    except AttributeError:
        tensors = flatten(dataset.element_spec)
        flat_result = {k: v.shape for k, v in tensors.items()}
        return unflatten(flat_result)


def get_output_types(dataset: tf.data.Dataset) -> Dict:
    try:
        return dict(dataset.output_types)
    except AttributeError:
        tensors = flatten(dataset.element_spec)
        flat_result = {k: v.dtype for k, v in tensors.items()}
        return unflatten(flat_result)


def padded_batch(
    dataset: tf.data.Dataset,
    batch_size: int,
    *,
    padding_values: Dict = None,
    padding_value_int: int = None,
    padding_value_bool: bool = None,
    verbose: bool = False,
) -> tf.data.Dataset:
    flatten_ds_shapes = flatten(get_output_shapes(dataset))
    flatten_ds_types = flatten(get_output_types(dataset))
    assert set(flatten_ds_shapes) == set(flatten_ds_types)
    flatten_padded_shapes = {
        k: cur_shape.as_list() for k, cur_shape in flatten_ds_shapes.items()
    }
    assert any(None in s for s in flatten_padded_shapes.values())

    if verbose:
        print("creating a padded batch with the following padded dimensions:")
        for k, s in flatten_padded_shapes.items():
            if None in s:
                print("\t%s: %s" % ("/".join(k), str(s)))

    padded_shapes = unflatten(flatten_padded_shapes)

    flatten_userdefined_padding_values = flatten(padding_values or {})
    flatten_padding_values = {
        k: ("" if flatten_ds_types[k] == tf.string else tf.cast(0, flatten_ds_types[k]))
        if (None not in flatten_padded_shapes[k])
        else cur_type.as_numpy_dtype(flatten_userdefined_padding_values[k])
        if k in flatten_userdefined_padding_values
        else cur_type.as_numpy_dtype(np.nan)
        if cur_type.is_floating
        else "__padded_batch_default_string_pad__"
        if cur_type == tf.string
        else cur_type.as_numpy_dtype(padding_value_int)
        if cur_type.is_integer and padding_value_int is not None
        else cur_type.as_numpy_dtype(padding_value_bool)
        if cur_type.is_bool and padding_value_bool is not None
        else None
        for k, cur_type in flatten_ds_types.items()
    }
    keys_missing_padding_values = []
    for k, padval in flatten_padding_values.items():
        if padval is None and None in flatten_padded_shapes[k]:
            keys_missing_padding_values.append(
                (k, flatten_ds_types[k], flatten_ds_shapes[k])
            )
    if len(keys_missing_padding_values) > 0:
        raise AssertionError(
            "no padding values given for:\n%s"
            % "\n".join(map(str, keys_missing_padding_values))
        )

    padding_values = unflatten(flatten_padding_values)
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=padded_shapes,
        padding_values=padding_values,
        # padded_shapes=AttrDict(padded_shapes),
        # padding_values=AttrDict(padding_values),
    )
    return dataset
