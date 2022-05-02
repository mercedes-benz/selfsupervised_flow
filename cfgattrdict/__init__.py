#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


__all__ = [
    "ask_yn",
    "convert_camel2snake",
    "dassert",
    "denumpyfy",
    "get_current_cpu_memory_usage",
    "get_current_system_metrics",
    "get_time_stamp",
    "hash_string",
    "is_dict",
    "is_power_of_2",
    "lazy_property",
    "munge_filename",
    "parse_value",
    "visualize",
    "PythonicDumper",
    "yaml_pythonic_dump",
    "yaml_pythonic_load",
]

import os
import os.path as osp
import subprocess
import traceback as tb
import typing as t

import numpy as np

from .basics import (
    ask_yn,
    convert_camel2snake,
    dassert,
    denumpyfy,
    get_current_cpu_memory_usage,
    get_current_system_metrics,
    get_time_stamp,
    hash_string,
    is_dict,
    is_power_of_2,
    lazy_property,
    munge_filename,
    parse_value,
    visualize,
)
from .pythonic_yaml_handlers import (
    PythonicDumper,
    yaml_pythonic_dump,
    yaml_pythonic_load,
)


class NoDiff:
    def __str__(self):
        return ""


class AttrDict(dict):
    """An AttrDict with strings as keys can also be accessed through
    attrdict.key = value notation."""

    # __getattr__ = dict.__getitem__
    def __getattr__(self, *args, **kwargs):
        if args[0] in self.keys():
            return super().__getitem__(*args, **kwargs)
        elif hasattr(super(), "__getattr__"):
            return super().__getattr__(*args, **kwargs)
        else:
            raise AttributeError("AttrDict has no key %s" % str(args[0]))

    # __setattr__ = t.cast(t.Callable[[object, str, t.Any], None], AttrDict.__setitem__)
    def __setattr__(self, key: str, value):
        self.__setitem__(key, value)

    def __setitem__(self, key: str, value):
        assert key not in ["update", "values", "keys"], (
            "__setitem__ on AttrDict was called with illegal key name %s and value %s:\n%s"
            % (key, str(value), str(self))
        )
        super().__setitem__(key, value)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key in self:
            self[key] = self._recursive_attr_dict_conversion(self[key])

    @classmethod
    def _recursive_attr_dict_conversion(cls, nested_struct):
        if is_dict(nested_struct):
            return cls(nested_struct)
        elif (
            isinstance(nested_struct, tuple)
            or isinstance(nested_struct, list)
            and isinstance(nested_struct, set)
        ):
            istuple = isinstance(nested_struct, tuple)
            isset = isinstance(nested_struct, set)
            if istuple or isset:
                nested_struct = list(nested_struct)
            for i in range(len(nested_struct)):
                nested_struct[i] = cls._recursive_attr_dict_conversion(nested_struct[i])
            if istuple:
                nested_struct = tuple(nested_struct)
            if isset:
                nested_struct = set(nested_struct)
            return nested_struct
        else:
            return nested_struct

    def recursive_dict_conversion(self, nested_struct=None, top_level=True):
        if nested_struct is None and top_level:
            nested_struct = self
        if is_dict(nested_struct):
            return {
                k: self.recursive_dict_conversion(v, top_level=False)
                for k, v in nested_struct.items()
            }
        elif (
            isinstance(nested_struct, tuple)
            or isinstance(nested_struct, list)
            or isinstance(nested_struct, set)
        ):
            istuple = isinstance(nested_struct, tuple)
            isset = isinstance(nested_struct, set)
            result = []
            if istuple or isset:
                nested_struct_as_list = list(nested_struct)
            else:
                nested_struct_as_list = nested_struct
            for i in range(len(nested_struct_as_list)):
                result.append(
                    self.recursive_dict_conversion(
                        nested_struct_as_list[i], top_level=False
                    )
                )
            if istuple:
                result = tuple(result)
            if isset:
                result = set(result)
            return result
        else:
            return nested_struct

    def inert(self):
        class InertAttrDict:
            def __init__(self, attrdict):
                self.attrdict = attrdict

            def active(self):
                return self.attrdict

        return InertAttrDict(self)


def apply_dfs_recursive_call_on_structs_inplace(operation: t.Callable, struct_to_apply):
    if is_dict(struct_to_apply):
        for key in sorted(struct_to_apply):
            apply_dfs_recursive_call_on_structs_inplace(
                operation, struct_to_apply=struct_to_apply[key]
            )
    elif isinstance(struct_to_apply, (set, tuple, list)):
        for elem in struct_to_apply:
            apply_dfs_recursive_call_on_structs_inplace(operation, struct_to_apply=elem)
    operation(struct_to_apply)


class ConfigAttrDict(AttrDict):
    """A ConfigAttrDict with strings as keys can also be accessed through
    attrdict.key = value notation.

    Rules for recursive updates:
    - dict <- dict: union over keys, for same key: recursive update of values
    - dict <- set: keys with values to be removed
    - list <- list: concatenating or replacing (see flag in rupdate)
    - list <- int: updating list to be list[:int] (-x notation works)
    - list <- tuple(int, list): first performing list <- int, then list <- list (concatenating, not replacing regardless of selected mode)
    - tuple <- tuple: recursively updating
    - tuple <- dict: dict keys as index to recursively update single elements
    - set: union
    """

    def __init__(self, *args, **kwargs):
        # finalized should not be added to the keys of the dict
        # therefore the __setattr__ from the original dict class needs to be called
        self.set_attribute("finalized", False)
        self.set_attribute("_initialized_path", None)
        self.set_description(None)
        super().__init__(*args, **kwargs)

    @classmethod
    def _recursive_finalization(cls, nested_struct):
        if isinstance(nested_struct, cls):
            nested_struct.finalize()
            return
        assert not is_dict(
            nested_struct
        ), "There was a non-ConfigAttrDict inside a ConfigAttrDict! %s" % str(
            nested_struct
        )
        if isinstance(nested_struct, (tuple, list, set)):
            for v in nested_struct:
                cls._recursive_finalization(v)

    def set_attribute(self, name: str, value):
        super(dict, self).__setattr__(name, value)

    def set_description(self, description: str):
        self.set_attribute("description", description)

    def finalize(self):
        self.set_attribute("finalized", True)
        for v in self.values():
            self._recursive_finalization(v)

    @classmethod
    def _recursive_unfinalization(cls, nested_struct):
        if isinstance(nested_struct, cls):
            nested_struct.unfinalize()
            return
        assert not is_dict(
            nested_struct
        ), "There was a non-ConfigAttrDict inside a ConfigAttrDict! %s" % str(
            nested_struct
        )
        if isinstance(nested_struct, (tuple, list, set)):
            for v in nested_struct:
                cls._recursive_unfinalization(v)

    def unfinalize(self):
        self.set_attribute("finalized", False)
        for v in self.values():
            self._recursive_unfinalization(v)

    def __setitem__(self, *args, **kwargs):
        assert not self.finalized, (
            "This ConfigAttrDict was already finalized! "
            "Setting attribute or item with name %s to value %s failed!"
            % (str(args[0]), str(args[1]))
        )
        super().__setitem__(*args, **kwargs)

    __setattr__ = __setitem__

    def set_or_assert_equal(self, keys, value):
        self.set_default(keys, value)
        cur = self
        keys_left = keys
        if isinstance(keys, str):
            keys_left = [keys]
        while len(keys_left) > 0:
            cur = cur[keys_left[0]]
            keys_left = keys_left[1:]
        assert (
            cur == value
        ), "a different value for %s was already set: %s instead of %s\n%s" % (
            keys[-1],
            str(cur),
            str(value),
            str(self),
        )

    def set_default(self, keys, *args):
        if isinstance(keys, str) or len(keys) == 1:
            key = keys if isinstance(keys, str) else keys[0]
            if len(args) == 0:
                value = ConfigAttrDict()
            else:
                assert len(args) == 1
                value = args[0]
            if key in self.keys():
                if is_dict(self[key]) != is_dict(value) or (
                    not is_dict(self[key]) and type(self[key]) != type(value)
                ):
                    if None not in [self[key], value]:
                        raise TypeError(
                            "A default (type %s) was set for a key named %s "
                            "which had already a value of a different type %s assigned!"
                            % (type(value), key, type(self[key]))
                            + "\n"
                            + str(value)
                            + "\n"
                            + str(self[key])
                        )
            else:
                self[key] = value
            return self[key]
        else:
            self.set_default(keys[0])
            return self[keys[0]].set_default(keys[1:], *args)

    def get_train_specific_config(self):
        train_specs = ConfigAttrDict(self)
        train_specs.set_default("debug")
        del train_specs["debug"]
        return train_specs

    def copy(self, _struct_to_copy=None, _top_level=True):
        if _top_level:
            _struct_to_copy = self
        if is_dict(_struct_to_copy):
            return ConfigAttrDict(
                {
                    k: self.copy(_struct_to_copy=v, _top_level=False)
                    for k, v in _struct_to_copy.items()
                }
            )
        elif isinstance(_struct_to_copy, (tuple, list, set)):
            dtype = type(_struct_to_copy)
            result = [
                self.copy(_struct_to_copy=v, _top_level=False) for v in _struct_to_copy
            ]
            return dtype(result)
        else:
            assert not isinstance(_struct_to_copy, np.ndarray)
            return _struct_to_copy

    def from_file(self, filename):
        assert filename[-4:] == ".yml" or filename[-4:] == ".cfg"
        assert len(self.keys()) == 0
        self.clear()
        with open(filename, "r") as f:
            file_cfg = ConfigAttrDict(yaml_pythonic_load(f))
            self.rupdate(file_cfg.copy())

        def replace_meta_cfgs(element):
            if not is_dict(element) or "meta_cfgs" not in element:
                return
            assert isinstance(element["meta_cfgs"], list)
            for meta_cfg_name in element["meta_cfgs"]:
                assert isinstance(meta_cfg_name, str)
                apply_dfs_recursive_call_on_structs_inplace(
                    replace_meta_cfgs, self[meta_cfg_name]
                )
            result = ConfigAttrDict()
            assert isinstance(element["meta_cfgs"], list)
            for meta_cfg_name in element["meta_cfgs"]:
                assert isinstance(meta_cfg_name, str)
                result.rupdate(self[meta_cfg_name].copy())
            del element["meta_cfgs"]
            result.rupdate(element)
            for k in result:
                element[k] = result[k]

        apply_dfs_recursive_call_on_structs_inplace(replace_meta_cfgs, self)
        return self

    def _apply_dfs_recursive_call(self, operation: t.Callable, _struct_to_apply=None):
        if _struct_to_apply is None:
            _struct_to_apply = self
        if is_dict(_struct_to_apply):
            for key in _struct_to_apply:
                _struct_to_apply[key] = self._apply_dfs_recursive_call(
                    operation, _struct_to_apply=_struct_to_apply[key]
                )
        elif isinstance(_struct_to_apply, (set, tuple, list)):
            original_type = type(_struct_to_apply)
            result = []
            for elem in _struct_to_apply:
                result.append(
                    self._apply_dfs_recursive_call(operation, _struct_to_apply=elem)
                )
            _struct_to_apply = original_type(result)
        _struct_to_apply = operation(_struct_to_apply)
        return _struct_to_apply

    def _apply_bfs_recursive_call(
        self, operation: t.Callable, _struct_to_apply=None, _top_level=True
    ):
        if _top_level:
            _struct_to_apply = self
        _struct_to_apply = operation(_struct_to_apply)
        if is_dict(_struct_to_apply):
            for key in _struct_to_apply:
                _struct_to_apply[key] = self._apply_bfs_recursive_call(
                    operation, _struct_to_apply=_struct_to_apply[key], _top_level=False
                )
        elif isinstance(_struct_to_apply, (set, tuple, list)):
            original_type = type(_struct_to_apply)
            result = []
            for elem in _struct_to_apply:
                result.append(
                    self._apply_bfs_recursive_call(
                        operation, _struct_to_apply=elem, _top_level=False
                    )
                )
            _struct_to_apply = original_type(result)
        return _struct_to_apply

    def rupdate(
        self,
        struct_update,
        wrapped_cur_struct=None,
        meta_cfgs=None,
        replace_list_rule=False,
    ):
        """
        Rules for recursive updates:
        - dict <- dict: union over keys, for same key: recursive update of values
        - dict <- set: keys with values to be removed
        - list <- list: concatenating or replacing (see flag in rupdate)
        - list <- int: updating list to be list[:int] (-x notation works)
        - list <- tuple(int, list): first performing list <- int, then list <- list (concatenating, not replacing regardless of selected mode)
        - tuple <- tuple: recursively updating
        - tuple <- dict: dict keys as index to recursively update single elements
        - set: union
        """
        if meta_cfgs is not None:
            assert is_dict(meta_cfgs), "provided meta_cfgs is not a dict: %s" % str(
                meta_cfgs
            )
            assert is_dict(struct_update), (
                "when providing meta_cfgs the rupdate should be with a dict, not %s"
                % str(struct_update)
            )
            if "meta_cfgs" in struct_update:
                assert isinstance(
                    struct_update["meta_cfgs"], list
                ), "meta_cfgs field was not a list (of cfg names): %s" % str(
                    struct_update["meta_cfgs"]
                )
                for meta_cfg_name in struct_update["meta_cfgs"]:
                    assert isinstance(
                        meta_cfg_name, str
                    ), "provided meta_cfgs contained non strings: %s" % str(
                        meta_cfg_name
                    )
                    assert (
                        meta_cfg_name in meta_cfgs
                    ), "meta_cfg_name %s is not implemented in meta_cfgs: %s" % (
                        meta_cfg_name,
                        str(meta_cfgs),
                    )
                    self.rupdate(
                        meta_cfgs[meta_cfg_name],
                        meta_cfgs=meta_cfgs,
                        replace_list_rule=replace_list_rule,
                    )

        if wrapped_cur_struct is None:
            cur_struct = self
        else:
            assert isinstance(wrapped_cur_struct, list)
            assert len(wrapped_cur_struct) == 1
            cur_struct = wrapped_cur_struct[0]
        if is_dict(cur_struct):
            if not (
                is_dict(struct_update)
                or isinstance(struct_update, set)
                or struct_update is None
            ):
                raise ValueError(
                    "The provided updating structure %s is not a dict, a set or None!"
                    % str(struct_update)
                )
            if struct_update is None:
                cur_struct = struct_update
            elif isinstance(struct_update, set):
                for del_key in struct_update:
                    del cur_struct[del_key]
            else:
                all_keys = set(cur_struct.keys()) | set(struct_update.keys())
                all_keys -= {"meta_cfgs"}
                for key in all_keys:
                    if key not in struct_update:
                        continue
                    if key not in cur_struct:
                        cur_struct[key] = struct_update[key]
                    else:  # key is in both dicts
                        cur_struct[key] = self.rupdate(
                            struct_update[key],
                            [cur_struct[key]],
                            meta_cfgs=meta_cfgs,
                            replace_list_rule=replace_list_rule,
                        )
        elif isinstance(cur_struct, list):
            if not isinstance(struct_update, (list, tuple, int, np.integer)):
                raise ValueError(
                    "Can only update list %s with a list, tuple or int! "
                    "You provided %s of type %s!"
                    % (str(cur_struct), str(struct_update), type(struct_update))
                )
            if isinstance(struct_update, list):
                if replace_list_rule:
                    cur_struct = struct_update
                else:
                    cur_struct += struct_update
            elif isinstance(struct_update, (int, np.integer)):
                assert -len(cur_struct) <= struct_update <= len(cur_struct)
                cur_struct = cur_struct[:struct_update]
            elif isinstance(struct_update, tuple):
                assert len(struct_update) == 2
                assert isinstance(struct_update[0], (int, np.integer))
                assert isinstance(struct_update[1], list)
                cur_struct = self.rupdate(
                    struct_update[0], [cur_struct], replace_list_rule=False
                )
                cur_struct = self.rupdate(
                    struct_update[1], [cur_struct], replace_list_rule=False
                )
        elif isinstance(cur_struct, tuple):
            cur_struct = list(cur_struct)
            assert isinstance(struct_update, tuple) or is_dict(
                struct_update
            ), "current tuple structure %s cannot be updated with %s" % (
                str(cur_struct),
                str(struct_update),
            )
            if isinstance(struct_update, tuple):
                assert len(cur_struct) == len(struct_update)
                for i in range(len(cur_struct)):
                    cur_struct[i] = self.rupdate(
                        struct_update[i],
                        [cur_struct[i]],
                        replace_list_rule=replace_list_rule,
                    )
            elif is_dict(struct_update):
                cur_struct = list(cur_struct)
                for key_idx in sorted(struct_update.keys()):
                    assert key_idx < len(cur_struct)
                    cur_struct[key_idx] = self.rupdate(
                        struct_update[key_idx],
                        [cur_struct[key_idx]],
                        replace_list_rule=replace_list_rule,
                    )
            cur_struct = tuple(cur_struct)
        elif isinstance(cur_struct, set):
            assert isinstance(struct_update, set)
            cur_struct |= struct_update
        else:
            cur_struct = struct_update
        return cur_struct

    def _get_denumpyfied_train_spec(self):
        return denumpyfy(self.get_train_specific_config())

    def get_hash_value(self):
        return hash_string(yaml_pythonic_dump(self._get_denumpyfied_train_spec()))

    def get_train_cfg_as_string(self):
        return yaml_pythonic_dump(
            self._get_denumpyfied_train_spec(), default_flow_style=False
        )

    def dump_train_spec(self, filename):
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                f.write(self.get_train_cfg_as_string())

    def __str__(self):
        return yaml_pythonic_dump(
            self._get_denumpyfied_train_spec(), default_flow_style=False
        )

    @classmethod
    def get_nbr_untracked_files(cls):
        src_dir = os.getenv("SRC_DIR")
        nbr_untracked_files = subprocess.check_output(
            'git status --porcelain 2>/dev/null | grep "^??" | wc -l',
            cwd=src_dir,
            shell=True,
        )
        nbr_untracked_files = int(nbr_untracked_files.decode("utf8").strip())
        return nbr_untracked_files

    @classmethod
    def dump_command(cls, path):
        import sys

        with open(osp.join(path, "command.txt"), "w") as f:
            f.write(osp.abspath(sys.argv[0]) + " " + " ".join(sys.argv[1:]) + "\n")

    @classmethod
    def create_description_symlink(cls, path, description):
        assert description == munge_filename(description)
        src = osp.join("..", osp.basename(path))
        dst = osp.join(osp.dirname(path), "description", description)
        os.makedirs(osp.dirname(dst), exist_ok=True)
        if osp.exists(dst):
            assert osp.islink(dst)
            assert os.readlink(dst) == src, (os.readlink(dst), src)
            return
        os.symlink(src, dst)
        os.symlink(src + ".cfg", dst + ".cfg")

    @classmethod
    def dump_description(cls, path, description):
        dst = osp.join(path, "description.txt")
        if osp.exists(dst):
            with open(dst, "r") as fin:
                lines = fin.readlines()
            assert len(lines) == 1
            line = lines[0].strip()
            assert line == description, (line, description)
        else:
            with open(dst, "w") as fout:
                fout.write(description + "\n")

    @classmethod
    def dump_requirements(cls, path):
        src_dir = os.getenv("SRC_DIR")

        frozen_reqs = subprocess.check_output(
            "pip3 freeze", stderr=subprocess.STDOUT, shell=True, cwd=src_dir
        ).decode("utf8")

        with open(osp.join(path, "frozen_requirements.txt"), "w") as fout:
            fout.write(frozen_reqs)

    def dump_git_status(self, path):
        src_dir = os.getenv("SRC_DIR")
        nbr_untracked_files = self.get_nbr_untracked_files()
        if nbr_untracked_files > 0:
            automatic_git_add_intent = ask_yn(
                "You have %d untracked files. For the automatic git status "
                "file creation step it is recommended that you at least mark "
                "all files with an intent to add in git. "
                "Do it with `git add -N .`! Do you want me to do it for you?"
                % nbr_untracked_files
            )
            if automatic_git_add_intent:
                os.system("git add -N .")
                assert self.get_nbr_untracked_files() == 0
            else:
                quit()

        branch_name = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=src_dir
        )
        if isinstance(branch_name, bytes):
            branch_name = branch_name.decode("utf8")
        branch_name = branch_name.strip()
        self.dump_git_status_per_branch(
            os.path.join(path, "git-status-%s.txt" % branch_name),
            ref_branch=branch_name,
        )
        self.dump_git_status_per_branch(
            os.path.join(path, "git-status-master.txt"), ref_branch="master"
        )
        try:
            self.dump_git_status_per_branch(
                os.path.join(path, "git-status-origin-master.txt"),
                ref_branch="origin/master",
            )
        except subprocess.CalledProcessError:
            pass

    def dump_git_status_per_branch(
        self, git_file, ref_branch="origin/master", check_submodules=False
    ):
        src_dir = os.getenv("SRC_DIR")

        # get git status
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", ref_branch], cwd=src_dir
        )
        if check_submodules:
            # first check if there are any changes in submodules
            submodule_diffs = subprocess.check_output(
                "git submodule foreach --quiet --recursive 'git diff HEAD'",
                shell=True,
                cwd=src_dir,
            )
            if len(submodule_diffs) > 0:
                raise EnvironmentError(
                    "There were uncommited changes in the submodules. "
                    "Please commit them before starting a training!"
                )
        # get the diff
        diffs = subprocess.check_output(["git", "diff", ref_branch], cwd=src_dir)

        # write to file
        with open(git_file, "w") as f:
            if hasattr(commit_hash, "decode"):
                commit_hash = commit_hash.decode("ascii")
            if hasattr(diffs, "decode"):
                try:
                    ascii_diffs = diffs.decode("ascii")
                except UnicodeDecodeError as err:
                    errstr = str(err).split(":")[0]
                    pos = errstr.index(" in position ")
                    pos = int(errstr[len(" in position ") + pos :])
                    ascii_diffs = diffs.decode("utf8")
                    warning_msg = (
                        "Warning: The diff has some unicode characters (e.g. %s) "
                        "which could not be decoded to ASCII. Now using UTF8! "
                        "Please look out for those when applying this diff. "
                        "But, in general this should be fine as most applications can handle UTF8!"
                        % ascii_diffs[pos]
                    )
                    f.write(warning_msg)
                    f.write("\n")
                    print(warning_msg)
                    tb.print_exc()
            assert "-di" + "rty" not in ascii_diffs
            f.write(commit_hash)
            f.write("\n")
            f.write(ascii_diffs)

    def get_hashed_path(self) -> str:
        assert self._initialized_path is not None
        return self._initialized_path

    def initialize_hashed_path(
        self,
        base_path,
        *,
        exist_ok=True,
        timestamp: t.Union[str, bool] = False,
        dump_git_status: bool = True,
        dump_command: bool = True,
        dump_requirements: bool = True,
        verbose: str = None,
        description: str = None,
    ):
        """Create and return path based on the hash of this config.
        Parameters
        ----------
        base_path : string
            Base path under which this directory should be constructed.
        Returns
        -------
        string
            Create the directory and returns its path. Also creates a file with
            the same name and '.cfg' as extension were all the parameters are
            listed in yaml format.
        """
        assert self._initialized_path is None
        self.finalize()
        hash_value = self.get_hash_value()
        dirname = osp.join(base_path, hash_value)
        if timestamp is True:
            stamped_dirname = osp.join(dirname, get_time_stamp())
        elif isinstance(timestamp, str):
            stamped_dirname = osp.join(dirname, timestamp)
        else:
            assert timestamp is False
            stamped_dirname = dirname
        dir_already_existed = False
        if osp.exists(stamped_dirname):
            dir_already_existed = True
            if verbose is not None:
                print("%s directory already exists:\t\t%s" % (verbose, stamped_dirname))
        os.makedirs(stamped_dirname, exist_ok=exist_ok)
        if not dir_already_existed and verbose is not None:
            print("%s directory created:\t\t%s" % (verbose, stamped_dirname))
        self.dump_train_spec(dirname + ".cfg")
        if dump_git_status:
            self.dump_git_status(stamped_dirname)
        if dump_command:
            self.dump_command(stamped_dirname)
        if dump_requirements:
            self.dump_requirements(stamped_dirname)
        self.set_attribute("_initialized_path", stamped_dirname)
        if description is None:
            description = self.description
        elif self.description is None:
            self.description = description
        assert description == self.description
        if description is not None:
            self.create_description_symlink(dirname, description)
            self.dump_description(dirname, description)
        return stamped_dirname

    def flatten(self, _cfg_dict=None, _inside=False):
        if _inside:
            cfg_dict = _cfg_dict
        else:
            assert _cfg_dict is None
            cfg_dict = self
        if is_dict(cfg_dict):
            result = {}
            for k in cfg_dict:
                flatten_sub = self.flatten(_cfg_dict=cfg_dict[k], _inside=True)
                if is_dict(flatten_sub):
                    for fk in flatten_sub:
                        result[osp.join(k, fk)] = flatten_sub[fk]
                else:
                    result[k] = flatten_sub
            return result
        elif isinstance(cfg_dict, tuple):
            result = {}
            for i, val in enumerate(cfg_dict):
                if isinstance(val, NoDiff):
                    continue
                k = "!t%d" % i
                flatten_sub = self.flatten(_cfg_dict=val, _inside=True)
                if is_dict(flatten_sub):
                    for fk in flatten_sub:
                        result[osp.join(k, fk)] = flatten_sub[fk]
                else:
                    result[k] = flatten_sub
            return result
        else:
            return cfg_dict

    def get_tb_str_repr(self) -> str:
        import sys

        return "Configuration-Hash: %s\n\nCommand: `$ %s`\n\n    %s" % (
            self.get_hash_value(),
            osp.abspath(sys.argv[0]) + " " + " ".join(sys.argv[1:]),
            self.get_train_cfg_as_string().replace("\n", "\n    "),
        )
