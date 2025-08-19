# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import os

import pandas as pd
import pytest


@pytest.fixture
def sample_smiles1():
    # Test molecule with multiple rings and substituents
    return "CSc1ccc(-c2cnc(N)nc2-c2ccccc2O)cc1"


@pytest.fixture
def sf6_smiles():
    return "FS(F)(F)(F)(F)F"


@pytest.fixture
def pf5_smiles():
    return "FP(F)(F)(F)F"


@pytest.fixture
def ccl3i_smiles():
    return "ClC(Cl)(Cl)I"


@pytest.fixture
def xef2_smiles():
    return "F[Xe]F"


@pytest.fixture
def test_data_path():
    # Get the path relative to this test file
    curr_file_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(curr_file_dir, "..", "data")
    return data_path


@pytest.fixture
def smiles_list_100(test_data_path):
    df = pd.read_csv(os.path.join(test_data_path, "sample_smiles_100.csv"))
    return df["canonical_smiles"].tolist()


@pytest.fixture
def rdkit2D_desc_df(test_data_path):
    df = pd.read_csv(os.path.join(test_data_path, "rdkit2D_desc.csv"))
    return df
