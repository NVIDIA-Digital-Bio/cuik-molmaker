# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import configparser
import glob
import os
import shutil
import subprocess
import sys
import sysconfig

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    def run(self):
        # Ensure build directory exists
        build_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "build")
        os.makedirs(build_dir, exist_ok=True)

        # Prepare CMake command
        cmake_args = [
            "cmake",
            f"-DCMAKE_PREFIX_PATH={os.environ['CONDA_PREFIX']}/lib/"
            f"python{PYTHON_VERSION}/site-packages/torch/share/cmake;"
            f"{os.environ['CONDA_PREFIX']}",
            os.path.abspath(os.path.dirname(__file__)),
        ]

        # Run CMake
        print("Running CMake:", " ".join(cmake_args))
        subprocess.check_call(cmake_args, cwd=build_dir)

        # Run Make
        print("Running make -j4")
        subprocess.check_call(["make", "-j4"], cwd=build_dir)

        # Call the original build_ext to copy .so files, etc.
        super().run()


# Get torch and rdkit versions with priority: config settings > env vars > default
TORCH_VERSION = os.environ.get("TORCH_VERSION")
RDKIT_VERSION = os.environ.get("RDKIT_VERSION")
PYTHON_VERSION = os.environ.get("PYTHON_VERSION")


if TORCH_VERSION is None:
    print("Error: PyTorch version is not set.")
    print("Please specify it as follows using environment variables:")
    print(
        "TORCH_VERSION=2.6.0 RDKIT_VERSION=2024.03.4 PYTHON_VERSION=3.11 pip install ."
    )
    sys.exit(1)

if RDKIT_VERSION is None:
    print("Error: RDKit version is not set.")
    print("Please specify it as follows using environment variables:")
    print(
        "TORCH_VERSION=2.6.0 RDKIT_VERSION=2024.03.4 PYTHON_VERSION=3.11 pip install ."
    )
    sys.exit(1)

if PYTHON_VERSION is None:
    print("Error: Python version is not set.")
    print("Please specify it as follows using environment variables:")
    print(
        "TORCH_VERSION=2.6.0 RDKIT_VERSION=2024.03.4 PYTHON_VERSION=3.11 pip install ."
    )
    sys.exit(1)


# Update setup.cfg with the Python tag
PYTHON_DIGIT_ONLY_VERSION = PYTHON_VERSION.replace(".", "")

config = configparser.ConfigParser()
config.read("setup.cfg")
if "bdist_wheel" not in config:
    config["bdist_wheel"] = {}
config["bdist_wheel"]["python-tag"] = f"py{PYTHON_DIGIT_ONLY_VERSION}"
config["bdist_wheel"]["plat_name"] = sysconfig.get_platform()
with open("setup.cfg", "w") as f:
    config.write(f)

print(f"Environment TORCH_VERSION: {os.environ.get('TORCH_VERSION')}")
print(f"Environment RDKIT_VERSION: {os.environ.get('RDKIT_VERSION')}")

print(
    f"Building with TORCH_VERSION={TORCH_VERSION}, RDKIT_VERSION={RDKIT_VERSION}, "
    f"PYTHON_VERSION={PYTHON_VERSION}"
)


# Create package directory structure first
dest_dir = os.path.join("cuik_molmaker")
lib_dir = os.path.join(dest_dir, "lib")
utils_dir = os.path.join(dest_dir, "utils")
data_dir = os.path.join(dest_dir, "data")
os.makedirs(dest_dir, exist_ok=True)
os.makedirs(lib_dir, exist_ok=True)
os.makedirs(utils_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Copy all Python files from src/ to package directory
src_py_files = glob.glob(os.path.join("src", "**", "*.py"), recursive=True)
for src_file in src_py_files:
    # Get relative path from src/
    rel_path = os.path.relpath(src_file, "src")
    # Create destination path
    dest_path = os.path.join(dest_dir, rel_path)
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    # Copy the file
    print(f"Copying {src_file} to {dest_path}")
    shutil.copy2(src_file, dest_path)

# Copy data files
data_files = [
    "best_normalization_params.json",
    "fast_normalization_params.json",
    "descriptastorus_normalization_params.json",
    "README.md",
]
for data_file in data_files:
    src_path = os.path.join("data", data_file)
    dest_path = os.path.join(data_dir, data_file)
    if os.path.exists(src_path):
        print(f"Copying {src_path} to {dest_path}")
        shutil.copy2(src_path, dest_path)
    else:
        print(f"WARNING: {src_path} not found")


# Create an empty __init__.py in the lib directory to make it a package
lib_init_file = os.path.join(lib_dir, "__init__.py")
if not os.path.exists(lib_init_file):
    print(f"Creating {lib_init_file}")
    with open(lib_init_file, "w") as f:
        f.write("# This file makes the lib directory a Python package\n")


setup(
    name="cuik_molmaker",
    version="0.1",
    author="S. Veccham",
    author_email="sveccham@nvidia.com",
    description="C++ module for featurizing molecules",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    # Include both src directory and cuik_molmaker_py package
    packages=find_packages(
        include=[
            "cuik_molmaker",
            "cuik_molmaker.lib",
            "cuik_molmaker.utils",
            "cuik_molmaker.data",
        ]
    ),
    package_data={
        "cuik_molmaker": [
            "*.so",
            "*.py",
            "data/*.json",
            "data/*.md",
        ],  # Include Python extension and Python files
        "cuik_molmaker.lib": [
            "*.so",
            "__init__.py",
        ],  # Include shared libraries and __init__.py
        "cuik_molmaker.utils": ["*.py"],  # Include Python files
    },
    include_package_data=True,
    cmdclass={
        "build_ext": CMakeBuild,
    },
    install_requires=[
        f"rdkit=={RDKIT_VERSION}",
        f"torch=={TORCH_VERSION}",
        "scipy",
    ],
    build_requires=[
        f"rdkit=={RDKIT_VERSION}",
        f"torch=={TORCH_VERSION}",
    ],
    tests_require=["pytest"],
    extras_require={
        "dev": [
            "black>=24.2.0",
            "flake8>=7.3.0",
            "isort>=5.13.2",
            "pre-commit>=3.6.0",
        ],
    },
    python_requires=f"=={PYTHON_VERSION}.*",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        f"Programming Language :: Python :: {PYTHON_VERSION}",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    entry_points={
        "console_scripts": [
            "cuik-molmaker-fit-distribution=cuik_molmaker.utils.fit_distribution:main",
            "cuik-molmaker-mol-features=cuik_molmaker.mol_features:main",
        ],
    },
)

# Check if .so file exists, display helpful message if not
so_file = os.path.join(
    "build",
    f"cuik_molmaker_cpp.cpython-{PYTHON_DIGIT_ONLY_VERSION}-x86_64-linux-gnu.so",
)
print(f"Looking for compiled extension at: {so_file}")

if os.path.exists(so_file):
    print(f"Found compiled extension, copying to {dest_dir}")
    shutil.copy2(so_file, dest_dir)
else:
    print(
        "WARNING: Compiled extension not found."
        "You need to build the C++ extension first."
    )
    sys.exit(1)

# Check for the shared library
lib_file = os.path.join("build", "libcuik_molmaker_core.so")
if os.path.exists(lib_file):
    print(f"Found shared library, copying to {lib_dir}")
    shutil.copy2(lib_file, lib_dir)
else:
    print(
        "WARNING: Shared library not found. You need to build the C++ extension first."
    )
    print("Run cmake and make in the build directory first.")
    sys.exit(1)
