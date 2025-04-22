# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cuik_molmaker
import lzma
import pickle as pkl
import torch
import torch.nn.functional as F
import os
import pytest

# Helper function for pytests
def get_atom_feature_list(atom_featurizer_version):
    if atom_featurizer_version == "V1":
        return ["atomic-number", "total-degree","formal-charge", "chirality", "num-hydrogens", "hybridization"]
    elif atom_featurizer_version == "V2":
        return ["atomic-number-common", "total-degree","formal-charge", "chirality", "num-hydrogens", "hybridization-expanded"]
    elif atom_featurizer_version == "ORGANIC":
        return ["atomic-number-organic", "total-degree","formal-charge", "chirality", "num-hydrogens", "hybridization-organic"]
    else:
        raise ValueError(f"Invalid atom featurizer version: {atom_featurizer_version}")
    
@pytest.mark.parametrize("atom_featurizer_version", ["V1", "V2", "ORGANIC"])
def test_mol_featurizer(test_data_path, atom_featurizer_version):
    
    atom_feature_list = get_atom_feature_list(atom_featurizer_version)
    atom_onehot_property_list = cuik_molmaker.atom_onehot_feature_names_to_tensor(atom_feature_list)
    atom_float_property_list = cuik_molmaker.atom_float_feature_names_to_tensor(["aromatic", "mass"])
    bond_property_list = cuik_molmaker.bond_feature_names_to_tensor(["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"])

    explicit_H, offset_carbon, duplicate_edges, add_self_loop = False, False, True, False

    ref_file = f"sample_smiles_100_{atom_featurizer_version}_ref.xz"

    ref_file_path = os.path.join(test_data_path, ref_file)
    with lzma.open(ref_file_path, "rb") as f:
        features_ref = pkl.load(f)

    for smiles, features in features_ref.items():
        all_feats = cuik_molmaker.mol_featurizer(smiles, atom_onehot_property_list, atom_float_property_list, bond_property_list, explicit_H, offset_carbon, duplicate_edges, add_self_loop)
        atom_feats, bond_feats, edge_index, rev_edge_index, _ = all_feats

        assert torch.allclose(torch.from_numpy(features[0]), atom_feats), f"{smiles} atom feats diff: {torch.abs(torch.from_numpy(features[0]) - atom_feats).sum()}"
        assert torch.allclose(torch.from_numpy(features[1]).float(), bond_feats), f"{smiles} bond feats diff: {torch.abs(torch.from_numpy(features[1]) - bond_feats).sum()}"
        assert torch.allclose(torch.from_numpy(features[2]), edge_index), f"{smiles} edge index diff: {torch.abs(torch.from_numpy(features[2]) - edge_index).sum()}"
        assert torch.allclose(torch.from_numpy(features[3]), rev_edge_index), f"{smiles} rev edge index diff: {torch.abs(torch.from_numpy(features[3]) - rev_edge_index).sum()}"


def test_mol_featurizer_v2_special_cases(pf5_smiles, sf6_smiles, ccl3i_smiles, xef2_smiles):

    # Setup
    atom_onehot_property_list = cuik_molmaker.atom_onehot_feature_names_to_tensor(["hybridization-expanded"])
    atom_float_property_list = torch.Tensor([])
    bond_property_list = torch.Tensor([])

    explicit_H, offset_carbon, duplicate_edges, add_self_loop = False, False, True, False
    
    # SF6 for SP3D2 hybridization
    sf6_feats = cuik_molmaker.mol_featurizer(sf6_smiles, atom_onehot_property_list, atom_float_property_list, bond_property_list, explicit_H, offset_carbon, duplicate_edges, add_self_loop)
    atom_feats = sf6_feats[0]
    atom_feats_ref = F.one_hot(torch.tensor([4, 6, 4, 4, 4, 4, 4]), num_classes=8).float()
    assert torch.allclose(atom_feats, atom_feats_ref), f"atom feats diff: {torch.abs(atom_feats - atom_feats_ref).sum()}"

    # PF5 for SP3D hybridization
    pf5_feats = cuik_molmaker.mol_featurizer(pf5_smiles, atom_onehot_property_list, atom_float_property_list, bond_property_list, explicit_H, offset_carbon, duplicate_edges, add_self_loop)
    atom_feats = pf5_feats[0]
    atom_feats_ref = F.one_hot(torch.tensor([4, 5, 4, 4, 4, 4]), num_classes=8).float()
    assert torch.allclose(atom_feats, atom_feats_ref), f"atom feats diff: {torch.abs(atom_feats - atom_feats_ref).sum()}"

    # CCl3I for common atomic numbers
    atom_onehot_property_list = cuik_molmaker.atom_onehot_feature_names_to_tensor(["atomic-number-common"])
    ccl3i_feats = cuik_molmaker.mol_featurizer(ccl3i_smiles, atom_onehot_property_list, atom_float_property_list, bond_property_list, explicit_H, offset_carbon, duplicate_edges, add_self_loop)
    atom_feats = ccl3i_feats[0]
    atom_feats_ref = F.one_hot(torch.tensor([16, 5, 16, 16, 36]), num_classes=38).float()
    assert torch.allclose(atom_feats, atom_feats_ref), f"atom feats diff: {torch.abs(atom_feats - atom_feats_ref).sum()}"

    # XeF2 for common atomic numbers
    xef2_feats = cuik_molmaker.mol_featurizer(xef2_smiles, atom_onehot_property_list, atom_float_property_list, bond_property_list, explicit_H, offset_carbon, duplicate_edges, add_self_loop)
    atom_feats = xef2_feats[0]
    atom_feats_ref = F.one_hot(torch.tensor([8, 37, 8]), num_classes=38).float()
    assert torch.allclose(atom_feats, atom_feats_ref), f"atom feats diff: {torch.abs(atom_feats - atom_feats_ref).sum()}"

@pytest.mark.parametrize("atom_featurizer_version", ["V1", "V2", "ORGANIC"])
def test_batch_mol_featurizer(test_data_path, atom_featurizer_version):
    atom_feature_list = get_atom_feature_list(atom_featurizer_version)
    atom_onehot_property_list = cuik_molmaker.atom_onehot_feature_names_to_tensor(atom_feature_list)
    atom_float_property_list = cuik_molmaker.atom_float_feature_names_to_tensor(["aromatic", "mass"])
    bond_property_list = cuik_molmaker.bond_feature_names_to_tensor(["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"])

    explicit_H, offset_carbon, duplicate_edges, add_self_loop = False, False, True, False

    ref_file = f"sample_smiles_100_batch_{atom_featurizer_version}_ref.xz"
    ref_file_path = os.path.join(test_data_path, ref_file)  
    with lzma.open(ref_file_path, "rb") as f:
        features_ref = pkl.load(f)

    smiles_list = features_ref["smiles"]
    all_feats = cuik_molmaker.batch_mol_featurizer(smiles_list, atom_onehot_property_list, atom_float_property_list, bond_property_list, explicit_H, offset_carbon, duplicate_edges, add_self_loop)
    atom_feats, bond_feats, edge_index, rev_edge_index, batch = all_feats

    assert torch.allclose(features_ref['V'], atom_feats), f"atom feats diff: {torch.abs(features_ref['V'] - atom_feats).sum()}"
    assert torch.allclose(features_ref['E'], bond_feats), f"bond feats diff: {torch.abs(features_ref['E'] - bond_feats).sum()}"
    assert torch.allclose(features_ref['edge_index'], edge_index), f"edge index diff: {torch.abs(features_ref['edge_index'] - edge_index).sum()}"
    assert torch.allclose(features_ref['rev_edge_index'], rev_edge_index), f"rev edge index diff: {torch.abs(features_ref['rev_edge_index'] - rev_edge_index).sum()}"
    assert torch.allclose(features_ref['batch'], batch), f"batch diff: {torch.abs(features_ref['batch'] - batch).sum()}"
