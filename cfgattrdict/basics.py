#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import functools
import hashlib
import os
import re
from datetime import datetime
from typing import Dict

import numpy as np


def is_dict(value):
    if isinstance(value, dict):
        return True
    if not hasattr(value, "update"):
        return False
    if not callable(value.update):
        return False
    if not hasattr(value, "keys"):
        return False
    if not callable(value.keys):
        return False
    if not hasattr(value, "values"):
        return False
    if not callable(value.values):
        return False
    if not hasattr(value, "items"):
        return False
    if not callable(value.items):
        return False
    if not hasattr(value, "get"):
        return False
    if not callable(value.get):
        return False
    return True


def convert_camel2snake(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def hash_string(string):
    return hashlib.md5(string.encode()).hexdigest()


def parse_value(value, opening_bracket=None):
    if opening_bracket is not None:
        closing = ")" if opening_bracket == "(" else "]"
        next_open = min(
            map(
                lambda x: len(value) + 1 if x == -1 else x,
                [value.find("("), value.find("[")],
            )
        )
        closing_bracket = value.index(closing)
        if closing_bracket < next_open:
            if closing_bracket == 0:
                return [], value[1:]
            part = [parse_value(v) for v in value[:closing_bracket].split(",")]
            if opening_bracket == "(":
                part = tuple(part)
            return part, value[closing_bracket + 1 :]
        else:
            inside, after = parse_value(
                value[next_open + 1 :], opening_bracket=value[next_open]
            )
            if next_open == 0:
                part = []
            else:
                part = [parse_value(v) for v in value[: next_open - 1].split(",")]
            part.append(inside)
            final, after = parse_value(after, opening_bracket=opening_bracket)
            part += list(final)
            if opening_bracket == "(":
                part = tuple(part)
            return part, after
    while value[0] in "\"'":
        value = value.strip('"')
        value = value.strip("'")
    if value[0] in "([":
        result, after_closing = parse_value(value[1:], opening_bracket=value[0])
        assert len(after_closing) == 0
        return result
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value in ["True", "False"]:
        return value == "True"
    if value in ["on", "off"]:
        return value == "on"
    if value in ["None", "null"]:
        return None
    return value


def denumpyfy(tuple_list_dict_set_array_number):
    """A nested structure of tuples, lists, dicts, sets and the lowest level numpy
    values gets converted to an object with the same structure but all being
    corresponding native python numbers.
    Parameters
    ----------
    tuple_list_dict_set_array_number : tuple, list, dict, set, numpy array, number
        The object that should be converted.
    Returns
    -------
    tuple, list, dict, set, native number (float, int)
        The object with the same structure but only native python numbers.
    """
    if isinstance(tuple_list_dict_set_array_number, tuple):
        return tuple(denumpyfy(elem) for elem in tuple_list_dict_set_array_number)
    if isinstance(tuple_list_dict_set_array_number, set):
        return {denumpyfy(elem) for elem in tuple_list_dict_set_array_number}
    if isinstance(tuple_list_dict_set_array_number, list):
        return [denumpyfy(elem) for elem in tuple_list_dict_set_array_number]
    if is_dict(tuple_list_dict_set_array_number):
        return {
            denumpyfy(k): denumpyfy(tuple_list_dict_set_array_number[k])
            for k in tuple_list_dict_set_array_number
        }
    if isinstance(tuple_list_dict_set_array_number, np.ndarray):
        assert isinstance(
            tuple_list_dict_set_array_number.flatten()[0],
            (np.bool, np.floating, np.integer),
        )
        return tuple_list_dict_set_array_number.tolist()
    if isinstance(tuple_list_dict_set_array_number, (bool, np.bool)):
        return bool(tuple_list_dict_set_array_number)
    if isinstance(tuple_list_dict_set_array_number, (float, np.floating)):
        return float(tuple_list_dict_set_array_number)
    if isinstance(tuple_list_dict_set_array_number, (int, np.integer)):
        return int(tuple_list_dict_set_array_number)
    return tuple_list_dict_set_array_number


def get_time_stamp(with_date=True, with_delims=False, include_micros=False):
    micros = "-%f" if include_micros else ""
    if with_date:
        if with_delims:
            return datetime.now().strftime("%Y/%m/%d-%H:%M:%S" + micros)
        else:
            return datetime.now().strftime("%Y%m%d-%H%M%S" + micros)
    else:
        if with_delims:
            return datetime.now().strftime("%H:%M:%S" + micros)
        else:
            return datetime.now().strftime("%H%M%S" + micros)


def munge_filename(name, mode="strict"):
    """Remove characters that might not be safe in a filename."""
    if mode == "strict":
        non_alphabetic = re.compile("[^A-Za-z0-9_.]")
    else:
        non_alphabetic = re.compile("[^A-Za-z0-9_\\-.=,:]")
    return non_alphabetic.sub("_", name)


def ask_yn(question, default=-1, timeout=0):
    """Ask interactively a yes/no-question and wait for an answer.
    Parameters
    ----------
    question : string
        Question asked to the user printed in the terminal.
    default : int
        Default answer can be one of (-1, 0, 1) corresponding to no default
        (requires an user response), No, Yes.
    timeout : float
        Timeout (in seconds) after which the default answer is returned. This
        raises an error if there is no default provided (default = -1).
    Returns
    -------
    bool
        Answer to the question trough user or default. (Yes=True, No=False)
    """
    import select
    import sys

    answers = "[y/n]"
    if default == 0:
        answers = "[N/y]"
    elif default == 1:
        answers = "[Y/n]"
    elif default != -1:
        raise Exception("Wrong default parameter (%d) to ask_yn!" % default)

    if timeout > 0:
        if default == -1:
            raise Exception("When using timeout, specify a default answer!")
        answers += " (%.1fs time to answer!)" % timeout
    print(question + " " + answers)

    if timeout == 0:
        ans = input()
    else:
        i, o, e = select.select([sys.stdin], [], [], timeout)
        if i:
            ans = sys.stdin.readline().strip()
        else:
            ans = ""

    if ans == "y" or ans == "Y":
        return True
    elif ans == "n" or ans == "N":
        return False
    elif len(ans) == 0:
        if default == 0:
            return False
        elif default == 1:
            return True
        elif default == -1:
            print("There is no default option given to this y/n-question!")
            return ask_yn(question, default=default, timeout=timeout)
        else:
            raise Exception("Logical error in ask_yn function!")
    else:
        print("Wrong answer to y/n-question! Answer was %s!" % ans)
        return ask_yn(question, default=default, timeout=timeout)
    raise Exception("Logical error in ask_yn function!")


def lazy_property(function):
    """Decorator which adds lazy evaluation to the function and cashing the result.
    Parameters
    ----------
    function : callable
        The function that should be evaluated only once and providing the
        result that gets cached.
    Returns
    -------
    return type of callable
        The cached result from the first and only evaluation.
    """
    attribute = "_cache_" + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def visualize(points, *attributes, pdb=True):
    import pptk
    from scipy.optimize import minimize

    if not hasattr(visualize, "viewer"):
        visualize.viewer = pptk.viewer(np.array([[0.0, 0.0, 0.0]]))

    v = visualize.viewer
    lookat, phi, r, theta = (v.get("lookat"), v.get("phi"), v.get("r"), v.get("theta"))
    v.clear()

    assert len(points.shape) == 2
    if points.shape[0] < points.shape[1]:
        points = points.T

    attributes = [
        attr if attr.shape[0] == points.shape[0] else attr.T for attr in attributes
    ]

    x0 = np.mean(points[:, :3], axis=0, keepdims=True)

    def dist_func(x0):
        return np.sum(np.sqrt(np.sum((points[:, :3] - x0) ** 2, axis=1)))

    res = minimize(
        dist_func, x0, method="nelder-mead", options={"xtol": 1e-8, "disp": False}
    )

    v.load(points[:, :3])
    v.set(phi=phi, theta=theta)
    v.set(lookat=lookat, r=r)
    v.attributes(
        *attributes,
        *[points[:, i] for i in range(3, points.shape[1])],
        np.sqrt(np.sum(points[:, :3] ** 2, axis=-1)),
        np.sqrt(np.sum((points[:, :3] - res.x)[:, :3] ** 2, axis=-1)),
    )
    v.set(point_size=0.005)
    if pdb:
        import pdb  # noqa: T100

        pdb.set_trace()  # noqa: T100
    return v


def dassert(val):
    if not val:
        import pdb  # noqa: T100

        pdb.set_trace()  # noqa: T100
    assert val


def is_power_of_2(n):
    return (n & (n - 1) == 0) and n != 0


def get_current_cpu_memory_usage():
    import psutil

    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2.0 ** 30  # memory use in GB
    return np.asarray(memory_use, np.float32)


def get_current_system_metrics() -> Dict:
    import psutil

    return {
        "cpu/percent": psutil.cpu_percent(),
        "virtual_memory/total": psutil.virtual_memory()[0] / 2.0 ** 30,  # in GB
        "virtual_memory/available": psutil.virtual_memory()[1] / 2.0 ** 30,  # in GB
        "virtual_memory/percent": psutil.virtual_memory()[2],
        "virtual_memory/used": psutil.virtual_memory()[3] / 2.0 ** 30,  # in GB
        "virtual_memory/free": psutil.virtual_memory()[4] / 2.0 ** 30,  # in GB
        "process/cpu/memory": get_current_cpu_memory_usage(),
    }
