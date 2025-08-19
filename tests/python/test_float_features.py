# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import cuik_molmaker
import torch
import rdkit
import pandas as pd

def test_hydrogen_bond_donor_feature():
    smiles = "C(C(=O)O)C(=O)O"
    atom_onehot_features_names = []
    atom_float_features_names = [
        "hydrogen-bond-donor",
                                  "hydrogen-bond-acceptor", 
                                  "acidic", 
                                  "basic"]
    atom_onehot_features_tensor = cuik_molmaker.atom_onehot_feature_names_to_tensor(atom_onehot_features_names)
    atom_float_features_tensor = cuik_molmaker.atom_float_feature_names_to_tensor(atom_float_features_names)
    bond_property_tensor = cuik_molmaker.bond_feature_names_to_tensor(["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"])

    explicit_H, offset_carbon, duplicate_edges, add_self_loop = False, False, True, False
    atom_feats, bond_feats, edge_index, rev_edge_index, _ = cuik_molmaker.mol_featurizer(smiles, atom_onehot_features_tensor, atom_float_features_tensor, bond_property_tensor, explicit_H, offset_carbon, duplicate_edges, add_self_loop)
    print(f"{atom_feats.shape=}")
    print(torch.round(atom_feats, decimals=1))


def test_ring_size_feature():
    smiles = "Nc1nc(N2CCC(n3cc(C(=O)O)cn3)CC2)nc2ccccc12"

    atom_onehot_features_names = ["ring-size"]
    atom_float_features_names = []
    atom_onehot_features_tensor = cuik_molmaker.atom_onehot_feature_names_to_tensor(atom_onehot_features_names)
    atom_float_features_tensor = cuik_molmaker.atom_float_feature_names_to_tensor(atom_float_features_names)
    bond_property_tensor = cuik_molmaker.bond_feature_names_to_tensor(["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"])

    explicit_H, offset_carbon, duplicate_edges, add_self_loop = False, False, True, False
    atom_feats, bond_feats, edge_index, rev_edge_index, _ = cuik_molmaker.mol_featurizer(smiles, atom_onehot_features_tensor, atom_float_features_tensor, bond_property_tensor, explicit_H, offset_carbon, duplicate_edges, add_self_loop)
    print(f"{atom_feats.shape=}")
    print(torch.round(atom_feats, decimals=1))
