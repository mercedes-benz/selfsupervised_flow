#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


"""Console script for unsup_flow."""
import argparse
import functools as ft
import os
import os.path as osp
import sys

import numpy as np
import tensorflow as tf

from cfgattrdict import ConfigAttrDict, munge_filename, parse_value
from unsup_flow.experiment.experiment import UnsupervisedFlow


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("_", nargs="*")
    parser.add_argument("--prod", action="store_true")
    parser.add_argument("--desc", "-d", default=None, type=str)
    parser.add_argument("--profile", "-p", action="store_true")
    parser.add_argument("--eager", "-e", action="store_true")
    parser.add_argument("--minimal", "-m", action="store_true")
    parser.add_argument("--test", "-t", action="store_true")
    parser.add_argument("--keys_value", "-kv", action="append", nargs="+", default=[])
    parser.add_argument(
        "-cf",
        "--config_file",
        default=osp.join(os.getenv("CFG_DIR", "config"), "config.yml"),
    )
    parser.add_argument(
        "--configs", "-c", action="append", nargs="*", type=str, default=[]
    )

    args = parser.parse_args()
    args.configs = ft.reduce(lambda a, b: a + b, [[]] + args.configs)
    return args


def main_tf2():
    """Console script for unsup_flow."""
    args = parse_args()
    cfg = parse_cli_args_into_cfgattrdict(args)

    np.random.seed(cfg.random_seed)
    tf_seed = (np.random.randint(2 ** 32) + 3907157302) % (2 ** 32)
    if cfg.random_seed == 3141:
        assert tf_seed == 2718, tf_seed
    tf.random.set_seed(tf_seed)

    exp = UnsupervisedFlow(cfg)
    exp.prepare()
    if args.desc is not None:
        desc = munge_filename(args.desc)
        desc_fname = osp.join(osp.dirname(exp.experiment_path), "description.txt")
        if osp.exists(desc_fname):
            with open(desc_fname, "r") as fin:
                lines = fin.readlines()
                assert len(lines) == 1, lines
                assert lines[0].strip() == desc, (lines[0].strip(), desc)
        else:
            with open(desc_fname, "w") as fout:
                fout.write(desc + "\n")

    exp.run()

    return 0


def parse_cli_args_into_cfgattrdict(args):

    base_cfgs = ConfigAttrDict().from_file(args.config_file)

    cfg = ConfigAttrDict()
    if "default" in base_cfgs:
        cfg.rupdate(base_cfgs.default)
    if args.minimal:
        cfg.rupdate(base_cfgs.minimal)
    for config_name in args.configs:
        print("Updating cfg with %s:" % config_name)
        print(base_cfgs[config_name])
        cfg.rupdate(base_cfgs[config_name])
    if args.prod:
        cfg.set_default("debug")
        cfg.debug.prod = True
    if args.profile:
        cfg.set_default("debug")
        cfg.debug.profile = True
    else:
        cfg.set_default("debug")
    if args.eager:
        cfg.set_default("debug")
        cfg.debug.eager = True
    else:
        cfg.set_default("debug")
        cfg.debug.eager = False
    for keys_value in args.keys_value:
        if len(keys_value) <= 1:
            assert len(keys_value) == 0
            continue
        keys = keys_value[:-1]
        value = parse_value(keys_value[-1])
        upd_cfg = ConfigAttrDict()
        upd_cfg.set_default(keys, value)
        cfg.rupdate(upd_cfg)

    long_run = args.prod or args.test
    assert min(args.prod, args.test) == 0

    if not long_run:
        cfg.data.nbr_samples.train = 100
        cfg.data.nbr_samples.kitti = 10
        cfg.data.nbr_samples.valid = 10
        cfg.iterations.pretrain = min(cfg.iterations.pretrain, 100)
        cfg.iterations.train = min(cfg.iterations.train, 100)
        cfg.iterations.eval_every = min(cfg.iterations.eval_every, 5)
        cfg.iterations.full_eval_every = min(cfg.iterations.full_eval_every, 20)

    return cfg


if __name__ == "__main__":
    assert tf.__version__[0] == "2"
    sys.exit(main_tf2())  # pragma: no cover
