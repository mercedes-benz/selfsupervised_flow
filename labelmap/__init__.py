#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import functools
import os
import os.path as osp
import typing as t

import matplotlib
import matplotlib.cm
import numpy as np

from cfgattrdict import ConfigAttrDict


def get_label_map_from_file(
    raw_map_name: str, aggregation_name: str = None, color_map_name: str = None
) -> "LabelMap":
    label_map_cfgs = ConfigAttrDict().from_file(
        osp.join(t.cast(str, os.getenv("CFG_DIR")), "label_mappings.yml")
    )
    label_map = LabelMap(
        label_map_cfgs.label_names[raw_map_name],
        label_map_cfgs.label_aggregation.get(aggregation_name, None),
        label_map_cfgs.label_colors.get(color_map_name, None),
    )
    return label_map


class LabelMap:
    def __init__(
        self,
        ridx_rname_dict: t.Dict[int, str],
        mname_rnames_dict: t.Dict[str, t.List[str]] = None,
        raw_color_dict: t.Dict[int, t.Tuple[int, ...]] = None,
    ) -> None:
        self.ridx_rname_dict = ridx_rname_dict
        # check that no name occures more than once
        assert sorted(set(self.ridx_rname_dict.values())) == sorted(
            self.ridx_rname_dict.values()
        )
        self.rname_ridx_dict = {rn: ri for ri, rn in self.ridx_rname_dict.items()}
        if mname_rnames_dict is None:
            self.mname_rnames_dict: t.Dict[str, t.List[str]] = {"ignore": []}
        else:
            self.mname_rnames_dict = mname_rnames_dict
        if raw_color_dict is None:
            self.ridx_color_dict = {}
            cmap = matplotlib.cm.get_cmap("jet")
            for i, ridx in enumerate(sorted(self.ridx_rname_dict.keys())):
                if len(self.ridx_rname_dict) > 1:
                    self.ridx_color_dict[ridx] = cmap(
                        i / (len(self.ridx_rname_dict) - 1)
                    )
                else:
                    self.ridx_color_dict[ridx] = cmap(0.5)
        else:
            self._set_ridx_color_dict_from_raw_color_dict(raw_color_dict)
        assert set(self.ridx_color_dict.keys()).issubset(
            set(self.ridx_rname_dict.keys())
        )

        self._compute_rnames()
        self._fill_mname_rname_dict_with_defaults()
        self._compute_mnames()
        self._compute_mname_ridx_dict()
        self._compute_midx_ridx_dict()
        self._normalize_color_dict()
        self._compute_mcolors()
        self._compute_rname_color_dict()
        self._compute_mname_color_dict()
        self._compute_ridx_color_arr()
        self._compute_ridx_midx_dict()

    def chain(self, rhs: "LabelMap") -> "LabelMap":
        mname_rnames_dict = dict(rhs.mname_rnames_dict)
        for mname in mname_rnames_dict:
            original_rnames = []
            for rname in mname_rnames_dict[mname]:
                original_rnames += self.mname_rnames_dict[rname]
            mname_rnames_dict[mname] = original_rnames
        return LabelMap(
            ridx_rname_dict=dict(self.ridx_rname_dict),
            mname_rnames_dict=mname_rnames_dict,
            ridx_color_dict=dict(self.ridx_color_dict),
        )

    def _set_ridx_color_dict_from_raw_color_dict(
        self, raw_color_dict: t.Dict[int, t.Tuple[int, ...]]
    ) -> None:
        raw_set = set(raw_color_dict.keys())
        ridx_set = set(self.ridx_rname_dict.keys())
        rname_set = set(self.ridx_rname_dict.values())
        if raw_set.issubset(ridx_set):
            assert not raw_set.issubset(rname_set), (
                "color keys\n%s\ncould be raw idxs\n%s\nor raw names\n%s\nnot distinguishable and should be avoided"
                % (str(raw_set), str(ridx_set), str(rname_set))
            )
            self.ridx_color_dict = dict(raw_color_dict)
        elif raw_set.issubset(rname_set):
            assert not raw_set.issubset(ridx_set)
            self.ridx_color_dict = {}
            for ridx, rname in self.ridx_rname_dict.items():
                if rname in raw_color_dict:
                    self.ridx_color_dict[ridx] = raw_color_dict[rname]
        else:
            raise ValueError(
                "color keys\n%s\ndid not match either raw idxs\n%s\nor raw names\n%s"
                % (str(raw_set), str(ridx_set), str(rname_set))
            )

    def _compute_rnames(self) -> None:
        self.rnames = sorted(set(self.ridx_rname_dict.values()))

    def _fill_mname_rname_dict_with_defaults(self) -> None:
        mapped_rnames = functools.reduce(
            lambda a, b: a + b, self.mname_rnames_dict.values()
        )
        # check that each name is mapped not more than once
        assert sorted(mapped_rnames) == sorted(
            set(mapped_rnames)
        ), "there were names mapped more than once: %s" % sorted(mapped_rnames)
        # check that mapped raw labels actually exist
        assert sorted(set(self.rnames) | set(mapped_rnames)) == self.rnames
        unmapped_rnames = set(self.rnames) - set(mapped_rnames)
        # check that none of the unmapped names are already taken by group names
        assert len(unmapped_rnames & set(self.mname_rnames_dict.keys())) == 0
        self.mname_rnames_dict.update(
            {unmapped_rname: [unmapped_rname] for unmapped_rname in unmapped_rnames}
        )
        self.rname_mname_dict = {
            rname: mname
            for mname in self.mname_rnames_dict
            for rname in self.mname_rnames_dict[mname]
        }
        self.ridx_mname_dict = {
            self.rname_ridx_dict[rname]: mname
            for rname, mname in self.rname_mname_dict.items()
        }

    def _compute_mnames(self) -> None:
        self.mnames = set(self.mname_rnames_dict.keys())
        assert (
            "ignore" in self.mnames
        ), "aggregated labels should provide a possibly empty ignore group"
        self.mnames = ["ignore"] + sorted(self.mnames - {"ignore"})
        self.midx_mname_dict = {i: mn for i, mn in enumerate(self.mnames)}

    def _compute_mname_ridx_dict(self) -> None:
        self.mname_ridx_dict = {}
        for mname in self.mnames:
            if mname == "ignore" and len(self.mname_rnames_dict["ignore"]) == 0:
                continue
            assert len(self.mname_rnames_dict[mname]) > 0, (
                "labelmap had for a mapped name %s no corresponding raw names mapped:\n%s"
                % (mname, str(self.mname_rnames_dict))
            )
            rname = self.mname_rnames_dict[mname][0]
            self.mname_ridx_dict[mname] = self.rname_ridx_dict[rname]

    def _compute_midx_ridx_dict(self) -> None:
        self.midx_ridx_dict = {}
        for mname in self.mname_ridx_dict:
            self.midx_ridx_dict[self.mnames.index(mname)] = self.mname_ridx_dict[mname]

    def _normalize_color_dict(self) -> None:
        all_color_vals = []
        for ridx in self.ridx_color_dict:
            all_color_vals += list(self.ridx_color_dict[ridx])
        if all(
            map(lambda x: isinstance(x, int) and x >= 0 and x <= 255, all_color_vals)
        ):
            for ridx in self.ridx_color_dict:
                self.ridx_color_dict[ridx] = tuple(
                    map(lambda x: x / 255.0, self.ridx_color_dict[ridx])
                )
        for ridx in self.ridx_color_dict:
            if len(self.ridx_color_dict[ridx]) == 3:
                self.ridx_color_dict[ridx] = (*self.ridx_color_dict[ridx], 1.0)

            assert all(
                map(
                    lambda x: isinstance(x, float) and x >= 0.0 and x <= 1.0,
                    self.ridx_color_dict[ridx],
                )
            )

            assert len(self.ridx_color_dict[ridx]) == 4

    def _compute_mcolors(self) -> None:
        self.mcolors = [
            self.ridx_color_dict.get(self.mname_ridx_dict[mname], (0.0, 0.0, 0.0, 0.0))
            if mname in self.mname_ridx_dict
            else (0.0, 0.0, 0.0, 0.0)
            for mname in self.mnames
        ]

    def _compute_rname_color_dict(self) -> None:
        self.rname_color_dict = {
            self.ridx_rname_dict[ridx]: color
            for ridx, color in self.ridx_color_dict.items()
        }

    def _compute_mname_color_dict(self) -> None:
        self.mname_color_dict = {
            mname: color for mname, color in zip(self.mnames, self.mcolors)
        }

    def _compute_ridx_color_arr(self) -> None:
        ridxs = sorted(self.ridx_rname_dict.keys())
        minridx = ridxs[0]
        maxridx = ridxs[-1]
        assert minridx <= 0 or maxridx <= 2 * minridx, (minridx, maxridx)
        self.ridx_color_arr = -np.ones((max(maxridx + 1, maxridx + 1 - minridx), 4))
        for ridx, color in self.ridx_color_dict.items():
            self.ridx_color_arr[ridx] = color

    def _compute_ridx_midx_dict(self) -> None:
        self.ridx_midx_dict = {}
        for ridx, rname in self.ridx_rname_dict.items():
            for mname in self.mname_rnames_dict:
                if rname in self.mname_rnames_dict[mname]:
                    break
            assert rname in self.mname_rnames_dict[mname]
            self.ridx_midx_dict[ridx] = self.mnames.index(mname)
