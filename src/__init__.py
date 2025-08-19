# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .utils.fit_distribution import best_fit_distribution, get_fast_distribution
from .utils.descriptor_normalization import get_normalization_functions
from .mol_features import MoleculeFeaturizer
# Import compiled extension
from pathlib import Path
import os

# Find the .so file in this directory
_module_dir = Path(__file__).parent
for file in os.listdir(_module_dir):
    if file.endswith('.so') and 'cpython' in file:
        # Add the extension module directly
        from importlib.machinery import ExtensionFileLoader
        from importlib.util import spec_from_loader, module_from_spec

        _loader = ExtensionFileLoader('cuik_molmaker_cpp', str(_module_dir / file))
        _spec = spec_from_loader('cuik_molmaker_cpp', _loader)
        _module = module_from_spec(_spec)
        _loader.exec_module(_module)

        # Import all attributes from the module
        for attr in dir(_module):
            if not attr.startswith('_'):
                globals()[attr] = getattr(_module, attr)
        break
    

__all__ = [
    # mol_features functions
    "MoleculeFeaturizer",
    # fit_distribution functions
    "best_fit_distribution",
    "get_fast_distribution",
    # descriptor_normalization functions
    "get_normalization_functions"
]