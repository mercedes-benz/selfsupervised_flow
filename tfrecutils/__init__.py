#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


import os
import os.path as osp
import shutil
import typing as t
from glob import glob

import numpy as np
import tensorflow as tf
from flatten_dict import flatten, unflatten

from cfgattrdict import ask_yn, yaml_pythonic_dump, yaml_pythonic_load


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def get_feature(value, dtype):
    if dtype == "string":
        if isinstance(value, np.ndarray):
            value = value.reshape(-1)
        return _bytes_feature([v.encode() for v in value.tolist()])
    elif dtype == "int64":
        return _int64_feature(value)
    else:
        raise ValueError("Unhandable dtype: %s" % dtype)


def user_is_happy_with_data(data_dict) -> bool:
    print()
    print("This seems to be a newly created version of a dataset.")
    print(
        "Please take your time to check if the values of your first sample make sense to you!"
    )
    if ask_yn("Do you want to skip this step anyway?", default=0):
        return True
    for key in data_dict:
        print()
        print("Manual inpsection of    %s" % key)
        view = data_dict[key]
        if isinstance(view, np.ndarray):
            slice_shape = np.minimum(view.shape, 10)
            for i in range(len(view.shape)):
                view = view.take(range(slice_shape[i]), axis=i)
        print(view)
        if not ask_yn(
            "Are you happy with the data of %s %s %s?"
            % (
                key,
                data_dict[key].shape if isinstance(view, np.ndarray) else "(,)",
                data_dict[key].dtype
                if isinstance(view, np.ndarray)
                else type(data_dict[key]).__name__,
            ),
            default=1,
        ):
            return False
    return True


def write_tfrecord(
    data_dict,
    filename,
    *,
    verbose=False,
    dimension_dict=None,
    meta=None,
    ask_user_to_check_data=True,
) -> None:
    # #region normalize
    if dimension_dict is None:
        dimension_dict = {}

    if meta is None:
        meta = {}

    endswitch = ".tfrecords"
    if filename[-len(endswitch) :] != endswitch:
        filename += endswitch

    def reducer(a, b):
        assert b is not None
        assert "/" not in b
        if a is None:
            return b
        return a + "/" + b

    data_dict = flatten(data_dict, reducer=reducer)
    dimension_dict = flatten(dimension_dict, reducer=reducer)
    # #endregion normalize

    data_dict["filename"] = filename

    written_feature_format = {}
    feature = {}
    for key in data_dict:
        data = np.array(data_dict[key])
        dtype = data.dtype.name
        if dtype[:3] == "str":
            dtype = "string"
        written_feature_format[key] = dtype
        dims = data.shape
        if key in dimension_dict:
            fixed_dims = dimension_dict[key]
            assert isinstance(
                fixed_dims, (tuple, list)
            ), "You did not provide valid fixed dims for %s: %s" % (
                key,
                str(fixed_dims),
            )
            assert len(fixed_dims) == len(dims), (
                "You provided fixed %s dims for %s which were incompatible with found dims: %s"
                % (str(fixed_dims), key, str(dims))
            )
            for d, fd in zip(dims, fixed_dims):
                if fd not in [-1, None]:
                    if d != fd:
                        raise ValueError(
                            "For %s array dimensions %s do not work with "
                            "provided dimensions %s!" % (key, dims, fixed_dims)
                        )
                    assert d == fd
        else:
            fixed_dims = [None for d in dims]
        if dtype != "string":
            data = data.tobytes()
        written_feature_format[key + "_raw"] = "string"
        if dtype == "string":
            feature[key + "_raw"] = get_feature(data, "string")
        else:
            feature[key + "_raw"] = _bytes_feature([data])
        for i, dim in enumerate(dims):
            if fixed_dims[i] in [-1, None]:
                written_feature_format[key + "_dim%d" % i] = -1
                feature[key + "_dim%d" % i] = _int64_feature([dim])
            else:
                written_feature_format[key + "_dim%d" % i] = fixed_dims[i]

    # add meta information to feature format
    if len(meta) > 0:
        assert "meta" not in written_feature_format
        written_feature_format["meta"] = meta

    dirname = osp.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    format_filename = osp.join(dirname, "tfrecords_format.yml")
    if verbose:
        from tqdm import tqdm

        tqdm.write("\t".join(["Writing", filename]))

    if osp.exists(format_filename):
        with open(format_filename, "r") as fin:
            existing_feature_format = yaml_pythonic_load(fin)
        if existing_feature_format != written_feature_format:
            from datadiff import diff

            print(diff(existing_feature_format, written_feature_format))
            override = ask_yn(
                "New feature format detected! Do you want to override? "
                "Warning: Deletes complete content of folder!"
            )
            if not override:
                quit()
            else:
                shutil.rmtree(osp.dirname(format_filename))
                os.makedirs(osp.dirname(format_filename))

    if not osp.exists(format_filename):
        if (not ask_user_to_check_data) or user_is_happy_with_data(data_dict):
            with open(format_filename, "w") as fout:
                fout.write(yaml_pythonic_dump(written_feature_format))
        else:
            quit()

    with tf.io.TFRecordWriter(filename) as writer:
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())


def get_parsing_func_from_feature_format(feature_format):
    multi_dims = {}
    encoding_features = {}
    for key in feature_format:
        if key == "meta":
            continue
        if key + "_raw" in feature_format:
            dims = []
            while key + "_dim%d" % len(dims) in feature_format:
                dims += [feature_format[key + "_dim%d" % len(dims)]]
            multi_dims[key] = dims
        else:
            if key[-4:] == "_raw":
                encoding_features[key] = tf.io.FixedLenSequenceFeature(
                    [], getattr(tf, feature_format[key]), allow_missing=True
                )
            else:
                feature_type = feature_format[key]
                if "_dim" in key:
                    if feature_type == -1:
                        feature_type = "int64"
                    else:
                        continue
                encoding_features[key] = tf.io.FixedLenFeature(
                    [], getattr(tf, feature_type)
                )

    def parsing_func(proto):
        parsed = tf.io.parse_single_example(proto, encoding_features)
        for key in multi_dims:
            if feature_format[key] != "string":
                if feature_format[key] == "bool":
                    parsed[key] = tf.cast(
                        tf.io.decode_raw(parsed[key + "_raw"], out_type=tf.uint8),
                        tf.bool,
                    )
                else:
                    parsed[key] = tf.io.decode_raw(
                        parsed[key + "_raw"], out_type=feature_format[key]
                    )
            else:
                parsed[key] = parsed[key + "_raw"]
            parsed[key] = tf.reshape(
                parsed[key],
                [
                    parsed[key + "_dim%d" % i] if dim == -1 else dim
                    for i, dim in enumerate(multi_dims[key])
                ],
            )
            del parsed[key + "_raw"]
            for i, dim in enumerate(multi_dims[key]):
                if dim == -1:
                    del parsed[key + "_dim%d" % i]

        parsed = unflatten(parsed, splitter="path")
        return parsed

    return parsing_func


def tfrecord_parser(
    filenames_or_dirname,
    feature_format=None,
    *,
    keep_plain=False,
    parallel_prefetching=5,
    buffer_size=None,
    count=None,
) -> t.Union[tf.data.Dataset, t.Tuple[tf.data.Dataset, t.Dict[str, t.Any]]]:
    if feature_format is None:
        if isinstance(filenames_or_dirname, str):
            cur_filenames, feature_format = get_filenames_and_feature_format(
                filenames_or_dirname
            )
            filenames = [{"default": f} for f in cur_filenames]
            feature_format = {"default": feature_format}
        else:
            assert isinstance(filenames_or_dirname, dict)
            filenames_dict: t.Dict[str, t.List[str]] = {}
            feature_format = {}
            for key, dirname in filenames_or_dirname.items():
                cur_filenames, cur_feature_format = get_filenames_and_feature_format(
                    dirname
                )
                filenames_dict[key] = cur_filenames
                feature_format[key] = cur_feature_format
                some_key = key
            # filenames = list(t.cast(t.Iterator[t.Tuple[str]], zip(*filenames_dict)))
            filenames = [
                {k: filenames_dict[k][i] for k in filenames_dict}
                for i in range(len(filenames_dict[some_key]))
            ]
            for fnames in filenames:
                assert all(
                    osp.basename(fnames[some_key]) == osp.basename(f)
                    for f in fnames.values()
                ), fnames
    else:
        assert isinstance(filenames_or_dirname, list)
        if len(filenames_or_dirname) > 0 and isinstance(filenames_or_dirname[0], str):
            filenames = [{"default": f} for f in filenames_or_dirname]
            feature_format = {"default": feature_format}
        else:
            filenames = filenames_or_dirname

    assert isinstance(filenames, list)
    assert len(filenames_or_dirname) == 0 or isinstance(filenames[0], dict)
    assert isinstance(feature_format, dict)
    N = len(feature_format)
    assert all(N == len(f) for f in filenames)
    sorted_keys = sorted(feature_format.keys())

    if buffer_size is None:
        buffer_size = max(2 * len(filenames), 1)

    parsing_funcs = {
        k: get_parsing_func_from_feature_format(ff) for k, ff in feature_format.items()
    }
    if not keep_plain:
        np.random.shuffle(filenames)
    dataset = tf.data.Dataset.from_tensor_slices(
        np.array([[fnames[k] for k in sorted_keys] for fnames in filenames]).astype(str)
    )
    if not keep_plain:
        if tf.__version__[0] == "1":
            dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size=buffer_size, count=count)
            )
        else:
            dataset = dataset.shuffle(buffer_size=buffer_size).repeat(count=count)

    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    if tf.__version__[0] == "1":
        dataset = dataset.apply(
            tf.data.experimental.parallel_interleave(
                tf.data.TFRecordDataset, cycle_length=parallel_prefetching
            )
        )
    else:
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=parallel_prefetching,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    dataset = dataset.batch(N, drop_remainder=True)

    dataset = dataset.prefetch(parallel_prefetching)

    def agg_parsing_func(sample):
        assert sample.shape == (N,)
        if N > 1:
            return {k: parsing_funcs[k](sample[i]) for i, k in enumerate(sorted_keys)}
        else:
            return parsing_funcs["default"](sample[0])

    dataset = dataset.map(agg_parsing_func)

    metas = {k: ff.get("meta", None) for k, ff in feature_format.items()}
    if not all(m is None for m in metas.values()):
        if N == 1:
            return dataset, metas["default"]
        return dataset, metas
    return dataset


def tfrecord_single_file_reader(feature_format) -> t.Callable[[str], t.Any]:
    parsing_func = get_parsing_func_from_feature_format(feature_format)
    filename_ph = tf.placeholder(dtype=tf.string, shape=())
    filename_ds = tf.data.Dataset.from_tensors(filename_ph)
    parsed = (
        filename_ds.apply(tf.data.TFRecordDataset)
        .map(parsing_func)
        .make_initializable_iterator()
    )
    init = parsed.initializer
    parsed = parsed.get_next()
    s = tf.Session()

    def reader(filename):
        s.run(init, {filename_ph: filename})
        result, _ = s.run([parsed, init], {filename_ph: filename})
        return result

    return reader


def get_filenames_and_feature_format(
    dirname: str,
) -> t.Tuple[t.List[str], t.Dict[str, t.Any]]:
    filenames = sorted(glob(osp.join(dirname, "*.tfrecords")))
    with open(osp.join(dirname, "tfrecords_format.yml"), "r") as fin:
        feature_format = yaml_pythonic_load(fin)
    return filenames, feature_format


def get_filenames_and_tfrecord_file_reader(
    dirname,
) -> t.Union[
    t.Tuple[t.List[str], t.Callable[[str], t.Any]],
    t.Tuple[t.List[str], t.Callable[[str], t.Any], t.Dict[str, t.Any]],
]:
    filenames, feature_format = get_filenames_and_feature_format(dirname)
    reader = tfrecord_single_file_reader(feature_format)
    if "meta" in feature_format:
        return filenames, reader, feature_format["meta"]
    return filenames, reader


if __name__ == "__main__":
    dirname = "/tmp/testtfrecords"
    filename = osp.join(dirname, "test1")
    filename2 = osp.join(dirname, "test2")
    data_dict = {
        "simple_string": "hello world",
        "list_of_strings": ["hi", "ho0"],
        "array_of_strings": np.array([["hi1", "ho2"], ["hi3", "ho40"]]),
        "testfloat": 1.0,
        "testfloat32": np.array(1.0, dtype=np.float32),
        "testfloat64": np.array(1.0, dtype=np.float64),
        "testint": 1,
        "testint32": np.array(1, dtype=np.int32),
        "testint64": np.array(1, dtype=np.int64),
        "testfloatarray": np.eye(3),
        "testintarray": np.eye(3, dtype=np.int32),
        "testlistfloat": [1.0, 2.0],
        "testlistint": [[1, 0], [2, 3]],
        "testuint8": np.array(1, dtype=np.uint8),
        "testuint16": np.array(1, dtype=np.uint16),
        "testint16": np.array(1, dtype=np.int16),
        "testemptyarray": np.empty((0, 18, 23)),
        "testbool": np.eye(100, dtype=np.bool),
    }
    fixed_dims = {
        "array_of_strings": (None, 2),
        "testemptyarray": (None, 18, 23),
        "testlistint": (2, -1),
    }
    write_tfrecord(data_dict, filename, dimension_dict=fixed_dims)
    data_dict["add_info"] = "info"
    write_tfrecord(data_dict, filename2, dimension_dict=fixed_dims)
    with open(osp.join(dirname, "tfrecords_format.yml"), "r") as fin:
        print("tfrecords_format.yml")
        print(yaml_pythonic_load(fin))
    fnames, feature_format = get_filenames_and_feature_format(dirname)
    tf_dataset = tfrecord_parser(fnames, feature_format)
    iterator_get_next = tf_dataset.make_one_shot_iterator().get_next()
    with tf.Session() as session:
        for _ in range(len(fnames)):
            result = session.run(iterator_get_next)
            for k in result:
                if isinstance(result[k], np.ndarray):
                    print(
                        k,
                        iterator_get_next[k],
                        type(result[k]),
                        result[k].shape,
                        result[k],
                    )
                else:
                    print(k, iterator_get_next[k], type(result[k]), result[k])
