// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include "../../../src/features.h"
#include "../../../src/one_hot.h"
#include <iostream>

TEST_CASE("Molecule parsing works correctly", "[utils]") {
    std::string smiles = R"(CC1=C(CC(=O)O)c2cc(F)ccc2/C1=C\c1ccc([S+](C)[O-])cc1)";
    
    SECTION("With implicit hydrogens") {
        bool explicit_H = false;
        std::unique_ptr<RDKit::RWMol> mol = parse_mol(smiles, explicit_H);
        
        REQUIRE(mol != nullptr);
        CHECK(mol->getNumAtoms() == 25);
        CHECK(mol->getNumBonds() == 27);
    }
    
    SECTION("With explicit hydrogens") {
        bool explicit_H = true;
        std::unique_ptr<RDKit::RWMol> mol_with_h = parse_mol(smiles, explicit_H);
        
        REQUIRE(mol_with_h != nullptr);
        CHECK(mol_with_h->getNumAtoms() == 42);
    }
}

TEST_CASE("Feature name to tensor conversion", "[features]") {
    SECTION("Atom float feature names") {
        std::vector<std::string> atom_float_feature_names = {
            "atomic-number",  "mass", "valence", "implicit-valence", "hybridization", 
            "chirality", "aromatic", "in-ring", "min-ring", "max-ring", "num-ring", 
            "degree", "radical-electron", "formal-charge", "group", "period", 
            "single-bond", "aromatic-bond", "double-bond", "triple-bond", "is-carbon",
            "unknown-placeholder-feature" // represents unknown feature
        };

        at::Tensor atom_float_feature_tensor = atom_float_feature_names_to_tensor(atom_float_feature_names);
        at::Tensor atom_float_feature_tensor_ref = at::range(0, 21, at::dtype(at::kInt));
        int atom_float_diff = at::sum(at::abs(atom_float_feature_tensor - atom_float_feature_tensor_ref)).item<int>();
        
        CHECK(atom_float_diff == 0);
    }

    SECTION("Atom onehot feature names") {
        std::vector<std::string> atom_onehot_feature_names = {
            "atomic-number", "atomic-number-common", "atomic-number-organic", "degree", "total-degree", "valence", "implicit-valence", 
            "hybridization", "hybridization-expanded", "hybridization-organic", "chirality", "group", "period", "formal-charge", "num-hydrogens",
            "unknown-placeholder-feature" // represents unknown feature
        };

        at::Tensor atom_onehot_feature_tensor = atom_onehot_feature_names_to_tensor(atom_onehot_feature_names);
        at::Tensor atom_onehot_feature_tensor_ref = at::range(0, 15, at::dtype(at::kInt));
        int atom_onehot_diff = at::sum(at::abs(atom_onehot_feature_tensor - atom_onehot_feature_tensor_ref)).item<int>();
        
        CHECK(atom_onehot_diff == 0);
    }

    SECTION("Bond feature names") {
        std::vector<std::string> bond_feature_names = {
            "is-null", "bond-type-float", "bond-type-onehot", "in-ring", "conjugated", 
            "stereo", "conformer-bond-length", "estimated-bond-length", 
            "unknown-placeholder-feature" // represents unknown feature
        };

        at::Tensor bond_feature_tensor = bond_feature_names_to_tensor(bond_feature_names);
        at::Tensor bond_feature_tensor_ref = at::range(0, 8, at::dtype(at::kInt));
        int bond_feature_diff = at::sum(at::abs(bond_feature_tensor - bond_feature_tensor_ref)).item<int>();
        
        CHECK(bond_feature_diff == 0);
    }
}

TEST_CASE("Feature size calculation", "[features]") {
    SECTION("Atom onehot feature sizes") {
        // Test atomic number feature size
        size_t atomic_num_size = get_one_hot_atom_feature_size(AtomOneHotFeature::ATOMIC_NUM);
        CHECK(atomic_num_size == 101);
        
        // Test degree feature size
        size_t degree_size = get_one_hot_atom_feature_size(AtomOneHotFeature::DEGREE);
        CHECK(degree_size == 6);
        
        // Test formal charge feature size
        size_t formal_charge_size = get_one_hot_atom_feature_size(AtomOneHotFeature::FORMAL_CHARGE);
        CHECK(formal_charge_size == 6);
    }
    
    SECTION("Bond onehot feature sizes") {
        // Test bond type feature size
        size_t bond_type_size = get_one_hot_bond_feature_size(BondFeature::TYPE_ONE_HOT);
        CHECK(bond_type_size == 4);
        
        // Test bond stereo feature size
        size_t bond_stereo_size = get_one_hot_bond_feature_size(BondFeature::STEREO_ONE_HOT);
        CHECK(bond_stereo_size == 7);
    }
}

TEST_CASE("Multiple feature sizes", "[features]") {
    SECTION("Atom dimension calculation") {
        // Setup for atom dimension calculation
        std::vector<std::string> atom_onehot_feature_names = {"atomic-number", "total-degree","formal-charge", "chirality", "num-hydrogens", "hybridization"};
        at::Tensor atom_onehot_feature_tensor = atom_onehot_feature_names_to_tensor(atom_onehot_feature_names);
        std::vector<std::string> atom_float_feature_names = {"aromatic", "mass"};
        at::Tensor atom_float_feature_tensor = atom_float_feature_names_to_tensor(atom_float_feature_names);
        
        // Calculate and check the dimension
        size_t atom_dim = compute_atom_dim(atom_onehot_feature_tensor, atom_float_feature_tensor);
        CHECK(atom_dim == 133);
    }
    
    SECTION("Bond dimension calculation") {
        // Setup for bond dimension calculation
        std::vector<std::string> bond_feature_names = {"is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"};
        at::Tensor bond_feature_tensor = bond_feature_names_to_tensor(bond_feature_names);
        
        // Calculate and check the dimension
        size_t bond_dim = compute_bond_dim(bond_feature_tensor);
        CHECK(bond_dim == 14);
    }
} 
