// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! @file This file defines functions for float-valued atom and bond features,
//!       declared in float_features.h and called from features.cpp

#include "float_features.h"

#include "features.h"

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <RDGeneral/types.h>

#include <stdint.h>
#include <cmath>

static constexpr double qNaN = std::numeric_limits<double>::quiet_NaN();

// Fills in a particular atom float `feature` into `data`, for all atoms.
// See the declaration in float_features.h for more details.
template<typename T>
void get_atom_float_feature(const GraphData& graph, T* data, AtomFloatFeature feature, size_t stride, bool offset_carbon) {
    const uint32_t num_atoms = graph.num_atoms;
    constexpr uint32_t carbon_atomic_num = 6;
    using MT = typename FeatureValues<T>::MathType;
    switch (feature) {
    case AtomFloatFeature::ATOMIC_NUMBER: {
        const MT offset = offset_carbon ? carbon_atomic_num : 0;
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType((MT(graph.atoms[i].atomicNum) - offset) / MT(5));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::MASS: {
        const RDKit::ROMol& mol = *graph.mol.get();
        constexpr MT carbon_mass = MT(12.011);
        const MT offset = offset_carbon ? carbon_mass : 0;
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType((MT(mol.getAtomWithIdx(i)->getMass()) - offset) / MT(100));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::VALENCE: {
        const RDKit::ROMol& mol = *graph.mol.get();
        const MT offset = offset_carbon ? 4 : 0;
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(mol.getAtomWithIdx(i)->getTotalValence()) - offset);
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::IMPLICIT_VALENCE: {
        const RDKit::ROMol& mol = *graph.mol.get();
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(mol.getAtomWithIdx(i)->getImplicitValence()));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::HYBRIDIZATION: {
        const RDKit::ROMol& mol = *graph.mol.get();
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(mol.getAtomWithIdx(i)->getHybridization()));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::CHIRALITY: {
        const RDKit::ROMol& mol = *graph.mol.get();
        for (uint32_t i = 0; i < num_atoms; ++i) {
            const RDKit::Atom* atom = mol.getAtomWithIdx(i);
            std::string prop;
            bool has_prop = atom->getPropIfPresent(RDKit::common_properties::_CIPCode, prop);
            *data = FeatureValues<T>::convertToFeatureType(has_prop ? MT(prop.length() == 1 && prop[0] == 'R') : MT(2));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::AROMATIC: {
        const RDKit::ROMol& mol = *graph.mol.get();
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(mol.getAtomWithIdx(i)->getIsAromatic()));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::IN_RING: {
        const RDKit::ROMol& mol = *graph.mol.get();
        const RDKit::RingInfo* ring_info = mol.getRingInfo();
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(ring_info->numAtomRings(i) != 0));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::MIN_RING: {
        const RDKit::ROMol& mol = *graph.mol.get();
        const RDKit::RingInfo* ring_info = mol.getRingInfo();
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(ring_info->minAtomRingSize(i)));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::MAX_RING: {
        const RDKit::ROMol& mol = *graph.mol.get();
        for (uint32_t i = 0; i < num_atoms; ++i) {
            data[i * stride] = FeatureValues<T>::zero;
        }
        const RDKit::RingInfo* ring_info = mol.getRingInfo();
        const auto& rings = ring_info->atomRings();
        for (const auto& ring : rings) {
            const T size = FeatureValues<T>::convertToFeatureType(MT(ring.size()));
            for (const auto atom_index : ring) {
                if (size > data[atom_index * stride]) {
                    data[atom_index * stride] = size;
                }
            }
        }
        return;
    }
    case AtomFloatFeature::NUM_RING: {
        const RDKit::ROMol& mol = *graph.mol.get();
        const RDKit::RingInfo* ring_info = mol.getRingInfo();
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(ring_info->numAtomRings(i)));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::DEGREE: {
        const RDKit::ROMol& mol = *graph.mol.get();
        const MT offset = offset_carbon ? 2 : 0;
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(mol.getAtomWithIdx(i)->getTotalDegree()) - offset);
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::RADICAL_ELECTRON: {
        const RDKit::ROMol& mol = *graph.mol.get();
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(mol.getAtomWithIdx(i)->getNumRadicalElectrons()));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::FORMAL_CHARGE: {
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(graph.atoms[i].formalCharge));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::GROUP: {
        const MT offset = offset_carbon ? MT(atomicNumToGroupTable[carbon_atomic_num - 1]) : MT(0);
        for (uint32_t i = 0; i < num_atoms; ++i) {
            const uint32_t atomic_num = graph.atoms[i].atomicNum;
            *data = (atomic_num <= 0 || atomic_num > 118) ? FeatureValues<T>::nan_value : FeatureValues<T>::convertToFeatureType(MT(atomicNumToGroupTable[atomic_num - 1]) - offset);
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::PERIOD: {
        const MT offset = offset_carbon ? MT(atomicNumToPeriodTable[carbon_atomic_num - 1]) : MT(0);
        for (uint32_t i = 0; i < num_atoms; ++i) {
            const uint32_t atomic_num = graph.atoms[i].atomicNum;
            *data = (atomic_num <= 0 || atomic_num > 118) ? FeatureValues<T>::nan_value : FeatureValues<T>::convertToFeatureType(MT(atomicNumToPeriodTable[atomic_num - 1]) - offset);
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::SINGLE_BOND:
    case AtomFloatFeature::AROMATIC_BOND:
    case AtomFloatFeature::DOUBLE_BOND:
    case AtomFloatFeature::TRIPLE_BOND:
    {
        const RDKit::ROMol& mol = *graph.mol.get();
        const RDKit::Bond::BondType type =
            (feature == AtomFloatFeature::SINGLE_BOND) ? RDKit::Bond::SINGLE : (
                (feature == AtomFloatFeature::AROMATIC_BOND) ? RDKit::Bond::AROMATIC : (
                (feature == AtomFloatFeature::DOUBLE_BOND) ? RDKit::Bond::DOUBLE : (
                RDKit::Bond::TRIPLE)));
        for (uint32_t i = 0; i < num_atoms; ++i) {
            auto [begin, end] = mol.getAtomBonds(mol.getAtomWithIdx(i));
            uint32_t count = 0;
            for (; begin != end; ++begin) {
                count += (mol[*begin]->getBondType() == type);
            }
            *data = FeatureValues<T>::convertToFeatureType(MT(count));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::IS_CARBON: {
        const MT offset = offset_carbon ? MT(1) : MT(0);
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(graph.atoms[i].atomicNum == carbon_atomic_num) - offset);
            data += stride;
        }
        return;
    }
    default:
        break;
    }

    // Missing implementation
    assert(0);
    for (uint32_t i = 0; i < num_atoms; ++i) {
        *data = FeatureValues<T>::nan_value;
        data += stride;
    }
}

// Explicit instantiations, so that the function can be templated
// but still be used from other cpp files.
template void get_atom_float_feature<int16_t>(const GraphData& graph, int16_t* data, AtomFloatFeature feature, size_t stride, bool offset_carbon);
template void get_atom_float_feature<float>(const GraphData& graph, float* data, AtomFloatFeature feature, size_t stride, bool offset_carbon);
template void get_atom_float_feature<double>(const GraphData& graph, double* data, AtomFloatFeature feature, size_t stride, bool offset_carbon);


// Fills in a particular bond float `feature` into `data`, for all bonds.
// See the declaration in float_features.h for more details.
template<typename T>
void get_bond_float_feature(const GraphData& graph, T* data, BondFeature feature, size_t stride) {
    const uint32_t num_bonds = graph.num_bonds;
    switch (feature) {
    case BondFeature::IS_NULL: {
        const RDKit::ROMol& mol = *graph.mol.get();
        for (size_t i = 0; i < num_bonds; ++i, data += stride) {
            *data = (mol.getBondWithIdx(i) == nullptr) ? FeatureValues<T>::one : FeatureValues<T>::zero;
        }
        return;
    }
    case BondFeature::TYPE_FLOAT: {
        const RDKit::ROMol& mol = *graph.mol.get();
        for (size_t i = 0; i < num_bonds; ++i, data += stride) {
            auto type = graph.bonds[i].bondType;
            double value = 0;
            switch (type) {
            case RDKit::Bond::BondType::SINGLE: value = 1.0; break;
            case RDKit::Bond::BondType::DOUBLE: value = 2.0; break;
            case RDKit::Bond::BondType::TRIPLE: value = 3.0; break;
            case RDKit::Bond::BondType::AROMATIC: value = 1.5; break;
            default: value = mol.getBondWithIdx(i)->getBondTypeAsDouble();
            }
            *data = FeatureValues<T>::convertToFeatureType(value);
        }
        return;
    }
    case BondFeature::IN_RING: {
        const RDKit::ROMol& mol = *graph.mol.get();
        for (size_t i = 0; i < num_bonds; ++i, data += stride) {
            bool is_in_ring = mol.getRingInfo()->numBondRings(i) != 0;
            *data = is_in_ring ? FeatureValues<T>::one : FeatureValues<T>::zero;
        }
        return;
    }
    case BondFeature::CONJUGATED: {
        for (size_t i = 0; i < num_bonds; ++i, data += stride) {
            bool is_conjugated = graph.bonds[i].isConjugated;
            *data = is_conjugated ? FeatureValues<T>::one : FeatureValues<T>::zero;
        }
        return;
    }
    default:
        // Missing implementation
        assert(0);
        for (uint32_t i = 0; i < num_bonds; ++i, data += stride) {
            *data = FeatureValues<T>::nan_value;
        }
        return;
    }
}

// Explicit instantiations, so that the function can be templated
// but still be used from other cpp files.
template void get_bond_float_feature<int16_t>(const GraphData& graph, int16_t* data, BondFeature feature, size_t stride);
template void get_bond_float_feature<float>(const GraphData& graph, float* data, BondFeature feature, size_t stride);
template void get_bond_float_feature<double>(const GraphData& graph, double* data, BondFeature feature, size_t stride);
