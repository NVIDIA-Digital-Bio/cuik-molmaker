// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! @file This header file declares feature-related enums, functions, and structs,
//!       some of which are defined in features.cpp and exported to Python.

#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <stdint.h>
#include <type_traits>
#include <vector>

// Torch tensor headers
#include <ATen/ATen.h>
#include <ATen/Functions.h>

#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>

//! Levels at which features or labels can be associated
//! String names are in `feature_level_to_enum` in features.cpp
enum class FeatureLevel {
    NODE,       //!< Values for each node (atom)
    EDGE,       //!< Values for each edge (bond)
    NODEPAIR,   //!< Values for each pair of nodes (pair of atoms), even if no edge (bond)
    GRAPH       //!< Values for whole molecule
};

//! Features for use by `get_atom_float_feature` in float_features.cpp
//! String names are in `atom_float_name_to_enum` in features.cpp
enum class AtomFloatFeature {
    ATOMIC_NUMBER,
    MASS,
    VALENCE,
    IMPLICIT_VALENCE,
    HYBRIDIZATION,
    CHIRALITY,
    AROMATIC,
    IN_RING,
    MIN_RING,
    MAX_RING,
    NUM_RING,
    DEGREE,
    RADICAL_ELECTRON,
    FORMAL_CHARGE,
    GROUP,
    PERIOD,
    SINGLE_BOND,
    AROMATIC_BOND,
    DOUBLE_BOND,
    TRIPLE_BOND,
    IS_CARBON,
    HYDROGEN_BOND_DONOR,
    HYDROGEN_BOND_ACCEPTOR,
    ACIDIC,
    BASIC,
    UNKNOWN
};

//! Features for use by `get_one_hot_atom_feature` in one_hot.cpp
//! String names are in `atom_onehot_name_to_enum` in features.cpp
enum class AtomOneHotFeature {
    ATOMIC_NUM,         //!< All atomic numbers from 1 - 100
    ATOMIC_NUM_COMMON,  //!< First 4 rows of periodic table and Iodine
    ATOMIC_NUM_ORGANIC, //!< Organic chemistry elements only
    DEGREE,             //!< Number of explicit neighboring atoms
    TOTAL_DEGREE,       //!< Number of neighboring atoms including hydrogens
    VALENCE,            //!< Total valence of the atom
    IMPLICIT_VALENCE,   //!< Implicit valence of the atom
    HYBRIDIZATION,      //!< Hybridizations specified in `hybridizationList` in one_hot.cpp
    HYBRIDIZATION_EXPANDED, //!< Hybridizations expanded to include all possible values
    HYBRIDIZATION_ORGANIC, //!< Hybridizations of organic elements
    CHIRALITY,          //!< "R", anything other value ("S") or no value, and an extra
                        //!< chirality-related value (independent of the other two, so can
                        //!< have a 2nd one value)
    GROUP,              //!< Specified by `atomicNumToGroupTable` in float_features.h
    PERIOD,             //!< Specified by `atomicNumToPeriodTable` in float_features.h
    FORMAL_CHARGE,      //!< Formal charge on atom
    NUM_HYDROGENS,      //!< Total number of hydrogens (explicit and implicit) on an atom
    RING_SIZE,          //!< Number of rings the atom is in
    UNKNOWN             //!< Sentinel value.  Do not use.
};

//! Features for use by `get_one_hot_bond_feature` in one_hot.cpp (if ends in `ONE_HOT`), and
//! `get_bond_float_feature` in float_features.cpp
//! String names are in `bond_name_to_enum` in features.cpp
enum class BondFeature {
    IS_NULL,         //!< 1 if Bond is nullptr else 0
    TYPE_FLOAT,         //!< Bond type as a float, e.g. 2.0 for double, 1.5 for aromatic
    TYPE_ONE_HOT,       //!< Selected bond types specified in `bondTypeList` in one_hot.cpp
    IN_RING,            //!< 1.0 if the bond is in at least one ring, else 0.0
    CONJUGATED,         //!< 1.0 if the bond is conjugated, else 0.0
    STEREO_ONE_HOT,     //!< Selected bond stereo values specified in `bondStereoList` in
                        //!< one_hot.cpp
    CONFORMER_BOND_LENGTH,//!< Length of the bond from a conformer (either first or computed)
    ESTIMATED_BOND_LENGTH,//!< Length of the bond estimated with a fast heuristic
    UNKNOWN             //!< Sentinel value.  Do not use.
};


//! Options for handling NaN or infinite values, passed from Python to `mol_featurizer` in
//! features.cpp.  Masking is done in `mask_nans` in features.h
enum class MaskNaNStyle {
    NONE,   //!< Ignore (keep) NaN values
    REPORT, //!< (default behaviour) Count NaN values and report that with the index of the
            //!< first tensor that contained NaNs
    REPLACE //!< Replace NaN values with a specific value (defaults to zero)
};


//! Class to help supporting `int16_t` as if it's a 16-bit floating-point (FP16) type,
//! while still supporting `float` (FP32) and `double` (FP64).
template<typename T>
struct FeatureValues {};

//! Explicit instantiation of `FeatureValues` for `int16_t` as if it's a 16-bit
//! floating-point (FP16) type.
template<> struct FeatureValues<int16_t> {
    static constexpr int16_t zero = 0x0000;
    static constexpr int16_t one = 0x3C00;
    static constexpr int16_t nan_value = 0x7C01;

    template<typename T>
    static int16_t convertToFeatureType(T inputType) {
        static_assert(std::is_floating_point_v<T>);
        return c10::detail::fp16_ieee_from_fp32_value(float(inputType));
    }

    static constexpr bool is_finite(int16_t v) {
        // If the exponent bits are the maximum value, v is infinite or NaN
        return (v & 0x7C00) != 0x7C00;
    }

    using MathType = float;
};
//! Explicit instantiation of `FeatureValues` for `float` (FP32)
template<> struct FeatureValues<float> {
    static constexpr float zero = 0.0f;
    static constexpr float one = 1.0f;
    static constexpr float nan_value = std::numeric_limits<float>::quiet_NaN();

    template<typename T>
    static float convertToFeatureType(T inputType) {
        static_assert(std::is_floating_point_v<T>);
        return float(inputType);
    }

    static bool is_finite(float v) {
        return std::isfinite(v);
    }

    using MathType = float;
};
//! Explicit instantiation of `FeatureValues` for `double` (FP64)
template<> struct FeatureValues<double> {
    static constexpr double zero = 0.0;
    static constexpr double one = 1.0;
    static constexpr double nan_value = std::numeric_limits<double>::quiet_NaN();

    template<typename T>
    static double convertToFeatureType(T inputType) {
        static_assert(std::is_floating_point_v<T>);
        return double(inputType);
    }

    static constexpr bool is_finite(double v) {
        return std::isfinite(v);
    }

    using MathType = double;
};

//! Handling for NaN or infinite values in an array, `data`,  of `n` values.
//! @see MaskNaNStyle
template<typename T>
constexpr int64_t mask_nans(T* data, size_t n, MaskNaNStyle style, T value) {
    if (style == MaskNaNStyle::NONE) {
        return 0;
    }
    if (style == MaskNaNStyle::REPLACE) {
        for (size_t i = 0; i < n; ++i) {
            if (!FeatureValues<T>::is_finite(data[i])) {
                data[i] = value;
            }
        }
        return 0;
    }

    // assert(mask_nan_style == MaskNaNStyle::REPORT);
    int64_t num_nans = 0;
    for (size_t i = 0; i < n; ++i) {
        num_nans += (!FeatureValues<T>::is_finite(data[i]));
    }
    return num_nans;
}


// This is just a function to provide to torch, so that we don't have to copy
// the tensor data to put it in a torch tensor, and torch can delete the data
// when it's no longer needed.
template<typename T>
void deleter(void* p) {
    delete[](T*)p;
}

//! Helper function to construct a torch `Tensor` from a C++ array.
//! The `Tensor` takes ownership of the memory owned by `source`.
template<typename T>
at::Tensor torch_tensor_from_array(std::unique_ptr<T[]>&& source, const int64_t* dims, size_t num_dims, c10::ScalarType type) {
    return at::from_blob(
        source.release(),
        at::IntArrayRef(dims, num_dims),
        deleter<T>, c10::TensorOptions(type));
}

//! Most of the data needed about an atom
struct CompactAtom {
    uint8_t atomicNum;
    uint8_t totalDegree;
    int8_t formalCharge;
    uint8_t chiralTag;
    uint8_t totalNumHs;
    uint8_t hybridization;
    bool isAromatic;
    float mass;
};

//! Most of the data needed about a bond
struct CompactBond {
    uint8_t bondType;
    bool isConjugated;
    bool isInRing;
    uint8_t stereo;
    uint32_t beginAtomIdx;
    uint32_t endAtomIdx;
};

//! Data representing a molecule before featurization
struct GraphData {
    const size_t num_atoms;
    std::unique_ptr<CompactAtom[]> atoms;
    const size_t num_bonds;
    std::unique_ptr<CompactBond[]> bonds;

    std::unique_ptr<RDKit::RWMol> mol;
};

//! Computes the total dimension of atom features based on the property lists
size_t compute_atom_dim(const at::Tensor& atom_property_list_onehot, const at::Tensor& atom_property_list_float);

//! Computes the total dimension of bond features based on the property list
size_t compute_bond_dim(const at::Tensor& bond_property_list);

//! This is called from Python to list atom one-hot features in a format that will be faster
//! to interpret inside `mol_featurizer`, passed in the `atom_property_list_onehot` parameter.
//! Implemented in features.cpp, but declared here so that cuik_molmaker.cpp can expose them to
//! Python via pybind.
at::Tensor atom_onehot_feature_names_to_tensor(const std::vector<std::string>& features);

//! This is called from Python to list all atom one-hot features.
//! Implemented in features.cpp, but declared here so that cuik_molmaker.cpp can expose them to
//! Python via pybind.
std::vector<std::string> list_all_atom_onehot_features();

//! This is called from Python to list atom float features in a format that will be faster
//! to interpret inside `mol_featurizer`, passed in the `atom_property_list_float` parameter.
//! Implemented in features.cpp, but declared here so that cuik_molmaker.cpp can expose them to
//! Python via pybind.
at::Tensor atom_float_feature_names_to_tensor(const std::vector<std::string>& features);

//! This is called from Python to list all atom float features.
//! Implemented in features.cpp, but declared here so that cuik_molmaker.cpp can expose them to
//! Python via pybind.
std::vector<std::string> list_all_atom_float_features();

//! This is called from Python to list bond features in a format that will be faster
//! to interpret inside `mol_featurizer`, passed in the `bond_property_list` parameter.
//! Implemented in features.cpp, but declared here so that cuik_molmaker.cpp can expose them to
//! Python via pybind.
at::Tensor bond_feature_names_to_tensor(const std::vector<std::string>& features);

//! This is called from Python to list all bond features.
//! Implemented in features.cpp, but declared here so that cuik_molmaker.cpp can expose them to
//! Python via pybind.
std::vector<std::string> list_all_bond_features();

//! `mol_featurizer` is called from Python to get feature tensors for `smiles_string`.
//!
//! @param smiles_string SMILES string of the molecule to featurize
//! @param atom_property_list_onehot Torch `Tensor` returned by
//!                                  `atom_onehot_feature_names_to_tensor` representing the
//!                                  list of one-hot atom features to create.
//! @param atom_property_list_float Torch `Tensor` returned by
//!                                 `atom_float_feature_names_to_tensor` representing the
//!                                 list of float atom features to create.
//! @param bond_property_list Torch `Tensor` returned by `bond_feature_names_to_tensor`
//!                           representing the list of bond features to create.
//! @param explicit_H If true, implicit hydrogen atoms will be added explicitly
//!                   before featurizing.
//! @param duplicate_edges If true, bond features will have values stored for
//!                        both edge directions.
//! @param add_self_loop If true, bond features will have values stored for
//!                      self-edges.
//! @param offset_carbon If true, some atom float features will subtract a
//!                      value representing carbon, so that carbon atoms would have value zero.
//! @return A vector of torch `Tensor`s for the features.  The first tensor is the atom features 
//!         tensor, `num_atoms` by the number of values required for all one-hot and float atom
//!         features.  The second tensor is the bond features tensor, `num_edges` (or 
//!         `2*num_edges` if `duplicate_edges` is true) by the number of values required for all 
//!         bond features. The third tensor is a 2 by `num_edges` (or `2*num_edges` if
//!         `duplicate_edges` is true) representing the indices of the nodes each edge is
//!         connected to. The fourth tensor is a 1 by `num_edges` tensor representing the reverse
//!         of the third tensor. The fifth tensor is a 1 by `num_atoms` tensor containing 0s.
std::vector<at::Tensor> mol_featurizer(const std::string& smiles_string, 
    const at::Tensor& atom_property_list_onehot,
    const at::Tensor& atom_property_list_float,
    const at::Tensor& bond_property_list,
    bool explicit_H,
    bool offset_carbon,
    bool duplicate_edges,
    bool add_self_loop);

//! Creates an RWMol from a SMILES string.
//!
//! If `ordered` is true, and the string contains atom classes, called "bookmarks" in RDKit,
//! that form a complete (0-based) ordering of the atoms, the atoms will be reordered according
//! to this explicit order, and the bookmarks will be removed, so that canonical orders
//! can be correctly compared later.
//!
//! This is implemented in cuik_molmaker.cpp, but is declared in this header so
//! that both labels.cpp and features.cpp can call it.
std::unique_ptr<RDKit::RWMol> parse_mol(
    const std::string& smiles_string,
    bool explicit_H,
    bool ordered = true);

//! `batch_mol_featurizer` is called from Python to get feature tensors for `smiles_list`.
//!
//! @param smiles_list List of SMILES strings of molecules to featurize
//! @param atom_property_list_onehot Torch `Tensor` returned by
//!                                  `atom_onehot_feature_names_to_tensor` representing the
//!                                  list of one-hot atom features to create.
//! @param atom_property_list_float Torch `Tensor` returned by
//!                                 `atom_float_feature_names_to_tensor` representing the
//!                                 list of float atom features to create.
//! @param bond_property_list Torch `Tensor` returned by `bond_feature_names_to_tensor`
//!                           representing the list of bond features to create.
//! @param explicit_H If true, implicit hydrogen atoms will be added explicitly
//!                   before featurizing.
//! @param duplicate_edges If true, bond features will have values stored for
//!                        both edge directions.
//! @param offset_carbon If true, some atom float features will subtract a
//!                      value representing carbon, so that carbon atoms would have value zero.
//! @param add_self_loop If true, bond features will have values stored for
//!                      self-edges.
//! @return A vector of torch `Tensor`s for the features.  The first tensor is the atom features 
//!         tensor, total number of atoms  by the number of values required for all one-hot and 
//!         float atom features.  The second tensor is the bond features tensor, total number of 
//!         edges (or `2*total_num_edges` if `duplicate_edges` is true) by the number of values 
//!         required for all bond features. The third tensor is a 2 by `total_num_edges` (or 
//!         `2*total_num_edges` if `duplicate_edges` is true) representing the indices of the 
//!         nodes each edge is connected to. The fourth tensor is a 1 by `total_num_edges` 
//!         tensor representing the reverse of the third tensor. The fifth tensor is a 1 by 
//!         `total_num_atoms` tensor containing the index of the molecule each atom belongs to.
std::vector<at::Tensor> batch_mol_featurizer(const std::vector<std::string>& smiles_list,
    const at::Tensor& atom_property_list_onehot,
    const at::Tensor& atom_property_list_float,
    const at::Tensor& bond_property_list,
    bool explicit_H,
    bool offset_carbon,
    bool duplicate_edges,
    bool add_self_loop);
