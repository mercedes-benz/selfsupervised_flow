#!/usr/bin/env python
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT


"""The setup script."""

from setuptools import find_packages, setup

setup(
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Package for collaboration on unsupervised flow.",
    entry_points={
        "console_scripts": [
            "unsup_flow=unsup_flow.cli:main",
        ],
    },
    include_package_data=True,
    keywords="unsup_flow",
    name="unsup_flow",
    packages=find_packages(
        include=[
            "unsup_flow",
            "tfrecutils",
            "usfl_io",
            "npimgtools",
            "dl_np_tools",
            "cfgattrdict",
            "unsup_flow.*",
        ]
    ),
    test_suite="tests",
    version="0.1.0",
    zip_safe=False,
)
