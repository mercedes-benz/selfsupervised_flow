#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import os.path as osp
from time import time

from cfgattrdict import yaml_pythonic_dump


class BestMetricLogger:
    def __init__(self, path, default="max", exceptions=None, update_interval=10 * 60):
        self.filepath = osp.join(path, "best_metrics.yml")
        assert not osp.exists(self.filepath)
        self.start_time = time()
        self.log_values = {}
        self.step0 = False
        assert default in ["max", "min"]
        self.default = default
        if exceptions is None:
            self.exceptions = []
        else:
            self.exceptions = exceptions
        self.update_interval = update_interval
        self.last_update_time = time()

    def start(self):
        assert not self.step0
        self.step0 = True
        self.start_time = time()

    def log(self, step, *, save2disk=True, **kwargs):
        cur_time = time()
        if step == 0:
            assert not self.step0
            self.step0 = True
            self.start_time = cur_time
        assert self.step0
        rel_time = cur_time - self.start_time
        significant_change = False
        for key, val in kwargs.items():
            if key not in self.log_values:
                self.log_values[key] = [
                    {
                        "value": val,
                        "wall": cur_time,
                        "relative": rel_time,
                        "step": step,
                    },
                    {
                        "value": val,
                        "wall": cur_time,
                        "relative": rel_time,
                        "step": step,
                    },
                ]
                significant_change = True
            else:
                last_val = self.log_values[key][-1]["value"]
                try:
                    if (self.default == "max") != any(
                        exc in key for exc in self.exceptions
                    ):
                        better = last_val < val
                    else:
                        better = last_val > val
                    better = bool(better)
                except ValueError:
                    print(
                        "Could not determine if value %s improved over previous %s for key %s"
                        % (str(val), str(last_val), key)
                    )
                if better:
                    del self.log_values[key][-1]
                    self.log_values[key].append(
                        {
                            "value": val,
                            "wall": cur_time,
                            "relative": rel_time,
                            "step": step,
                        }
                    )
                    self.log_values[key].append(
                        {
                            "value": val,
                            "wall": cur_time,
                            "relative": rel_time,
                            "step": step,
                        }
                    )
                    significant_change = True
                else:
                    del self.log_values[key][-1]
                    self.log_values[key].append(
                        {
                            "value": last_val,
                            "wall": cur_time,
                            "relative": rel_time,
                            "step": step,
                        }
                    )
        if save2disk:
            if (
                significant_change
                or time() - self.last_update_time >= self.update_interval
            ):
                with open(self.filepath, "w") as fout:
                    fout.write(
                        yaml_pythonic_dump(self.log_values, default_flow_style=False)
                    )
                self.last_update_time = time()


if __name__ == "__main__":
    import numpy as np

    lg = BestMetricLogger("/tmp/test_best_metrics_logger")
    lg.log(0, asdf=3, zulu=-np.inf, right=0)
    lg.log(10, asdf=35, zulu=1, right=0)
    lg.log(30, asdf=3, zulu=0.5, right=0)
    lg.log(30, zulu=3)
