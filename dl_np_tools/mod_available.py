#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


try:
    import numpy as np
except ModuleNotFoundError:
    pass
try:
    import torch
except ModuleNotFoundError:
    pass
try:
    import tensorflow as tf
except ModuleNotFoundError:
    pass


def np_available():
    try:
        np
        return True
    except NameError:
        return False


def tf_available():
    try:
        tf
        return True
    except NameError:
        return False


def torch_available():
    try:
        torch
        return True
    except NameError:
        return False
