# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup
import os
import shutil

# Create package directory structure first
dest_dir = os.path.join('cuik_molmaker')
os.makedirs(dest_dir, exist_ok=True)

# Check if .so file exists, display helpful message if not
so_file = os.path.join('build', 'cuik_molmaker.cpython-311-x86_64-linux-gnu.so')
print(f"Looking for compiled extension at: {so_file}")

if os.path.exists(so_file):
    print(f"Found compiled extension, copying to {dest_dir}")
    shutil.copy2(so_file, dest_dir)
else:
    print("WARNING: Compiled extension not found. You need to build the C++ extension first.")
    print("Try running: python setup.py build_ext --inplace")
    # Uncomment to abort if .so is missing:
    # sys.exit(1)

# Ensure __init__.py exists
init_file = os.path.join(dest_dir, '__init__.py')
if not os.path.exists(init_file):
    print(f"Creating {init_file}")
    with open(init_file, 'w') as f:
        f.write("# Import compiled extension\n")
        f.write("from pathlib import Path\n")
        f.write("import os\n")
        f.write("import sys\n")
        f.write("\n")
        f.write("# Find the .so file in this directory\n")
        f.write("_module_dir = Path(__file__).parent\n")
        f.write("for file in os.listdir(_module_dir):\n")
        f.write("    if file.endswith('.so'):\n")
        f.write("        # Add the extension module directly\n")
        f.write("        from importlib.machinery import ExtensionFileLoader\n")
        f.write("        from importlib.util import spec_from_loader, module_from_spec\n")
        f.write("        \n")
        f.write("        _loader = ExtensionFileLoader('cuik_molmaker', str(_module_dir / file))\n") 
        f.write("        _spec = spec_from_loader('cuik_molmaker', _loader)\n")
        f.write("        _module = module_from_spec(_spec)\n")
        f.write("        _loader.exec_module(_module)\n")
        f.write("        \n")
        f.write("        # Import all attributes from the module\n")
        f.write("        for attr in dir(_module):\n")
        f.write("            if not attr.startswith('_'):\n") 
        f.write("                globals()[attr] = getattr(_module, attr)\n")
        f.write("        break\n")

setup(
    name="cuik_molmaker",
    version="0.1",
    author="S. Veccham",
    author_email="sveccham@nvidia.com",
    description="C++ module for featurizing molecules",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    # Explicitly list the package instead of using find_packages()
    packages=["cuik_molmaker"],
    package_data={
        'cuik_molmaker': ['*.so'],  # Include .so files in the package
    },
    install_requires=[
        'rdkit==2024.03.4',
        'torch==2.6.0',
    ],
    tests_require=['pytest'],
    python_requires='>=3.11',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
) 
