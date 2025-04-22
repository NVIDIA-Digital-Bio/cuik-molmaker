# cuik-molmaker
`cuik-molmaker` is a specialized package designed for molecular featurization, converting chemical structures into formats that can be effectively used as inputs for deep learning models, particularly graph neural networks (GNNs).
## Setup conda environment
```
# Create conda env
conda create -n cuik_molmaker_build python=3.11 conda-forge::rdkit==2024.03.4 conda-forge::pybind11==2.13.6 conda-forge::pytorch-cpu==2.6.0 conda-forge::libboost-devel==1.84.0 conda-forge::libboost-python-devel==1.84.0 conda-forge::numpy==1.26.4

# Activate conda env
conda activate cuik_molmaker_build
```

## Compile and run
```
cd repo/
mkdir -p build && cd build
# ensure that build/ directory is empty

# HACK: This is to accommodate NumPy<=2.0. In NumPy 2.0, `numpy/core/include` was moved to `numpy/_core/include`
ln -s $CONDA_PREFIX/lib/python3.11/site-packages/numpy/_core/include $CONDA_PREFIX/lib/python3.11/site-packages/numpy/core/include

cmake -DCMAKE_PREFIX_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/torch/share/cmake;$CONDA_PREFIX" ..

make -j4

# .so should be created

## Optional: Install as PyPI package
cd path/to/repo
pip install .

# this should create a cuik_molmaker directory with .so file and a __init__.py file.

# Test that install works
pytest -s tests/python/test_featurize_dims.py
```

## Usage
```
python
>>> import torch
>>> import cuik_molmaker
>>> atom_props_onehot = ["atomic-number", "total-degree", "formal-charge", "chirality", "num-hydrogens", "hybridization"]
>>> atom_property_list_onehot = cuik_molmaker.atom_onehot_feature_names_to_tensor(atom_props_onehot)
>>> print(f"{atom_property_list_onehot}")
tensor([ 0,  2,  9,  6, 10,  5])
```

## Minimal conda env for import and running
```
# Create minimal conda env
conda create -n cuik_molmaker_import python=3.11 conda-forge::rdkit==2024.03.4 conda-forge::pytorch==2.6.0

conda activate cuik_molmaker_import

cd path/to/repo
pip install .
python -c "import cuik_molmaker; print(dir(cuik_molmaker))"

```

## Testing
### Running C++ tests using Catch2
```
# Step 1: Build with test flag set to on
cmake -DCUIKMOLMAKER_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/torch/share/cmake;$CONDA_PREFIX" ..

# Optional: List tests/tags
./catch2_tests --list-tests
./catch2_tests --list-tags

# Step 2: Run all C++ tests
cd /path/to/build
./catch2_tests
```

### Running python tests using pytest
```
# Step 1: Install cuik-molmaker using pip
cd path/to/repo
pip install .

# Step 2: Run pytest
pytest -s tests/python
```