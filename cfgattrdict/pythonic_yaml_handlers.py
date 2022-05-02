#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import functools

import yaml

from .basics import denumpyfy


class PythonicLoader(yaml.SafeLoader):
    pass


PythonicLoader.add_constructor("!tuple", yaml.FullLoader.construct_python_tuple)

yaml_pythonic_load = functools.wraps(yaml.load)(
    functools.partial(yaml.load, Loader=PythonicLoader)
)


class PythonicDumper(yaml.SafeDumper):
    pass


def represent_tuple(dumper, data):
    return dumper.represent_sequence("!tuple", list(data))


PythonicDumper.add_representer(tuple, represent_tuple)
PythonicDumper.add_multi_representer(dict, PythonicDumper.represent_dict)

yaml_pythonic_dump = functools.wraps(yaml.dump)(
    lambda x, *args, **kwargs: yaml.dump(
        denumpyfy(x), *args, Dumper=PythonicDumper, **kwargs
    )
)
